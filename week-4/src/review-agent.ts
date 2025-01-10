/**
 * @module review-agent
 * Core functionality for reviewing and processing pull requests, generating code suggestions, and managing file reviews.
 */

import { Octokit } from "@octokit/rest";
import { WebhookEventMap } from "@octokit/webhooks-definitions/schema";
import { ChatCompletionMessageParam } from "groq-sdk/resources/chat/completions";
import * as xml2js from "xml2js";
import type {
  BranchDetails,
  BuilderResponse,
  Builders,
  CodeSuggestion,
  PRFile,
  PRSuggestion,
} from "./constants";
import { PRSuggestionImpl } from "./data/PRSuggestionImpl";
import { generateChatCompletion } from "./llms/chat";
import {
  PR_SUGGESTION_TEMPLATE,
  buildPatchPrompt,
  constructPrompt,
  getReviewPrompt,
  getTokenLength,
  getXMLReviewPrompt,
  isConversationWithinLimit,
} from "./prompts";
import {
  INLINE_FIX_FUNCTION,
  getInlineFixPrompt,
} from "./prompts/inline-prompt";
import { getGitFile } from "./reviews";

/**
 * Reviews a diff using the chat completion API
 * @param messages - Array of chat messages to send to the API
 * @returns The content of the chat completion response
 */
export const reviewDiff = async (messages: ChatCompletionMessageParam[]) => {
  const message = await generateChatCompletion({
    messages,
  });
  return message.content;
};

/**
 * Reviews multiple files by combining their patches and getting feedback
 * @param files - Array of PR files to review
 * @param patchBuilder - Function to build a patch string from a file
 * @param convoBuilder - Function to build conversation messages from a diff
 * @returns The review feedback string
 */
export const reviewFiles = async (
  files: PRFile[],
  patchBuilder: (file: PRFile) => string,
  convoBuilder: (diff: string) => ChatCompletionMessageParam[]
) => {
  const patches = files.map((file) => patchBuilder(file));
  const messages = convoBuilder(patches.join("\n"));
  const feedback = await reviewDiff(messages);
  return feedback;
};

/**
 * Filters out files that should not be reviewed based on extension and filename
 * @param file - PR file to check
 * @returns Boolean indicating if file should be reviewed
 */
const filterFile = (file: PRFile) => {
  const extensionsToIgnore = new Set<string>([
    "pdf",
    "png",
    "jpg",
    "jpeg",
    "gif",
    "mp4",
    "mp3",
    "wav",
    "ogg",
    "webm",
    "md",
    "json",
    "env",
    "toml",
    "svg",
    "ico",
    "webp",
    "bmp",
    "tiff",
    "yaml",
    "yml",
    "lock",
    "csv",
    "xlsx",
    "xls",
    "doc",
    "docx",
    "ppt",
    "pptx",
    "zip",
    "rar",
    "tar",
    "gz",
  ]);
  const filesToIgnore = new Set<string>([
    "package-lock.json",
    "yarn.lock", 
    ".gitignore",
    "package.json",
    "tsconfig.json",
    "poetry.lock",
    "readme.md",
    "changelog.md",
    "license",
    "license.md",
    "license.txt",
    "contributing.md",
    "code_of_conduct.md",
    "requirements.txt",
    "pipfile",
    "pipfile.lock",
    "composer.json",
    "composer.lock",
    "gemfile",
    "gemfile.lock",
    ".env.example",
    ".prettierrc",
    ".eslintrc",
    ".editorconfig",
    "jest.config.js",
    "babel.config.js",
    "webpack.config.js",
    "rollup.config.js",
    "vite.config.js",
    "next.config.js"
  ]);
  const filename = file.filename.toLowerCase().split("/").pop();
  if (filename && filesToIgnore.has(filename)) {
    console.log(`Filtering out ignored file: ${file.filename}`);
    return false;
  }
  const splitFilename = file.filename.toLowerCase().split(".");
  if (splitFilename.length <= 1) {
    console.log(`Filtering out file with no extension: ${file.filename}`);
    return false;
  }
  const extension = splitFilename.pop()?.toLowerCase();
  if (extension && extensionsToIgnore.has(extension)) {
    console.log(`Filtering out file with ignored extension: ${file.filename} (.${extension})`);
    return false;
  }
  return true;
};

/**
 * Groups PR files by their file extension
 * @param files - Array of PR files to group
 * @returns Map of file extensions to arrays of files
 */
const groupFilesByExtension = (files: PRFile[]): Map<string, PRFile[]> => {
  const filesByExtension: Map<string, PRFile[]> = new Map();

  files.forEach((file) => {
    const extension = file.filename.split(".").pop()?.toLowerCase();
    if (extension) {
      if (!filesByExtension.has(extension)) {
        filesByExtension.set(extension, []);
      }
      filesByExtension.get(extension)?.push(file);
    }
  });

  return filesByExtension;
};

/**
 * Processes files that are within the model's token limits by grouping them appropriately
 * @param files - Array of PR files to process
 * @param patchBuilder - Function to build patch string from file
 * @param convoBuilder - Function to build conversation from diff
 * @returns Array of file groups that can be processed together
 */
const processWithinLimitFiles = (
  files: PRFile[],
  patchBuilder: (file: PRFile) => string,
  convoBuilder: (diff: string) => ChatCompletionMessageParam[]
) => {
  const processGroups: PRFile[][] = [];
  const convoWithinModelLimit = isConversationWithinLimit(
    constructPrompt(files, patchBuilder, convoBuilder)
  );

  console.log(`Within model token limits: ${convoWithinModelLimit}`);
  if (!convoWithinModelLimit) {
    const grouped = groupFilesByExtension(files);
    for (const [extension, filesForExt] of grouped.entries()) {
      const extGroupWithinModelLimit = isConversationWithinLimit(
        constructPrompt(filesForExt, patchBuilder, convoBuilder)
      );
      if (extGroupWithinModelLimit) {
        processGroups.push(filesForExt);
      } else {
        // extension group exceeds model limit
        console.log(
          "Processing files per extension that exceed model limit ..."
        );
        let currentGroup: PRFile[] = [];
        filesForExt.sort((a, b) => a.patchTokenLength - b.patchTokenLength);
        filesForExt.forEach((file) => {
          const isPotentialGroupWithinLimit = isConversationWithinLimit(
            constructPrompt([...currentGroup, file], patchBuilder, convoBuilder)
          );
          if (isPotentialGroupWithinLimit) {
            currentGroup.push(file);
          } else {
            processGroups.push(currentGroup);
            currentGroup = [file];
          }
        });
        if (currentGroup.length > 0) {
          processGroups.push(currentGroup);
        }
      }
    }
  } else {
    processGroups.push(files);
  }
  return processGroups;
};

/**
 * Removes lines starting with '-' from a file's patch
 * @param originalFile - PR file to process
 * @returns New PR file object with removed lines stripped from patch
 */
const stripRemovedLines = (originalFile: PRFile) => {
  const originalPatch = String.raw`${originalFile.patch}`;
  const strippedPatch = originalPatch
    .split("\n")
    .filter((line) => !line.startsWith("-"))
    .join("\n");
  return { ...originalFile, patch: strippedPatch };
};

/**
 * Processes files that exceed the model's token limits
 * @param files - Array of PR files to process
 * @param patchBuilder - Function to build patch string from file
 * @param convoBuilder - Function to build conversation from diff
 * @returns Array of file groups that can be processed
 */
const processOutsideLimitFiles = (
  files: PRFile[],
  patchBuilder: (file: PRFile) => string,
  convoBuilder: (diff: string) => ChatCompletionMessageParam[]
) => {
  const processGroups: PRFile[][] = [];
  if (files.length == 0) {
    return processGroups;
  }
  files = files.map((file) => stripRemovedLines(file));
  const convoWithinModelLimit = isConversationWithinLimit(
    constructPrompt(files, patchBuilder, convoBuilder)
  );
  if (convoWithinModelLimit) {
    processGroups.push(files);
  } else {
    const exceedingLimits: PRFile[] = [];
    const withinLimits: PRFile[] = [];
    files.forEach((file) => {
      const isFileConvoWithinLimits = isConversationWithinLimit(
        constructPrompt([file], patchBuilder, convoBuilder)
      );
      if (isFileConvoWithinLimits) {
        withinLimits.push(file);
      } else {
        exceedingLimits.push(file);
      }
    });
    const withinLimitsGroup = processWithinLimitFiles(
      withinLimits,
      patchBuilder,
      convoBuilder
    );
    withinLimitsGroup.forEach((group) => {
      processGroups.push(group);
    });
    if (exceedingLimits.length > 0) {
      console.log("TODO: Need to further chunk large file changes.");
      // throw "Unimplemented"
    }
  }
  return processGroups;
};

/**
 * Processes XML suggestions from review feedback
 * @param feedbacks - Array of XML feedback strings
 * @returns Array of parsed PR suggestions
 */
const processXMLSuggestions = async (feedbacks: string[]) => {
  const xmlParser = new xml2js.Parser();
  const parsedSuggestions = await Promise.all(
    feedbacks.map((fb) => {
      fb = fb
        .split("<code>")
        .join("<code><![CDATA[")
        .split("</code>")
        .join("]]></code>");
      console.log(fb);
      return xmlParser.parseStringPromise(fb);
    })
  );
  const allSuggestions = parsedSuggestions
    .map((sug) => sug.review.suggestion)
    .flat(1);
  const suggestions: PRSuggestion[] = allSuggestions.map((rawSuggestion) => {
    const lines = rawSuggestion.code[0].trim().split("\n");
    lines[0] = lines[0].trim();
    lines[lines.length - 1] = lines[lines.length - 1].trim();
    const code = lines.join("\n");

    return new PRSuggestionImpl(
      rawSuggestion.describe[0],
      rawSuggestion.type[0],
      rawSuggestion.comment[0],
      code,
      rawSuggestion.filename[0]
    );
  });
  return suggestions;
};

/**
 * Generates a GitHub issue URL with the given parameters
 * @param owner - Repository owner
 * @param repoName - Repository name
 * @param title - Issue title
 * @param body - Issue body
 * @param codeblock - Optional code block to include
 * @returns Markdown formatted issue link
 */
const generateGithubIssueUrl = (
  owner: string,
  repoName: string,
  title: string,
  body: string,
  codeblock?: string
) => {
  const encodedTitle = encodeURIComponent(title);
  const encodedBody = encodeURIComponent(body);
  const encodedCodeBlock = codeblock
    ? encodeURIComponent(`\n${codeblock}\n`)
    : "";

  let url = `https://github.com/${owner}/${repoName}/issues/new?title=${encodedTitle}&body=${encodedBody}${encodedCodeBlock}`;

  if (url.length > 2048) {
    url = `https://github.com/${owner}/${repoName}/issues/new?title=${encodedTitle}&body=${encodedBody}`;
  }
  return `[Create Issue](${url})`;
};

/**
 * Removes duplicate suggestions based on their identity
 * @param suggestions - Array of PR suggestions
 * @returns Deduplicated array of suggestions
 */
export const dedupSuggestions = (
  suggestions: PRSuggestion[]
): PRSuggestion[] => {
  const suggestionsMap = new Map<string, PRSuggestion>();
  suggestions.forEach((suggestion) => {
    suggestionsMap.set(suggestion.identity(), suggestion);
  });
  return Array.from(suggestionsMap.values());
};

/**
 * Converts PR suggestions into formatted comment strings
 * @param owner - Repository owner
 * @param repo - Repository name
 * @param suggestions - Array of PR suggestions
 * @returns Array of formatted comment strings
 */
const convertPRSuggestionToComment = (
  owner: string,
  repo: string,
  suggestions: PRSuggestion[]
): string[] => {
  const suggestionsMap = new Map<string, PRSuggestion[]>();
  suggestions.forEach((suggestion) => {
    if (!suggestionsMap.has(suggestion.filename)) {
      suggestionsMap.set(suggestion.filename, []);
    }
    suggestionsMap.get(suggestion.filename).push(suggestion);
  });
  const comments: string[] = [];
  for (let [filename, suggestions] of suggestionsMap) {
    const temp = [`## ${filename}\n`];
    suggestions.forEach((suggestion: PRSuggestion) => {
      const issueLink = generateGithubIssueUrl(
        owner,
        repo,
        suggestion.describe,
        suggestion.comment,
        suggestion.code
      );
      temp.push(
        PR_SUGGESTION_TEMPLATE.replace("{COMMENT}", suggestion.comment)
          .replace("{CODE}", suggestion.code)
          .replace("{ISSUE_LINK}", issueLink)
      );
    });
    comments.push(temp.join("\n"));
  }
  return comments;
};

/**
 * Builds a response from XML feedback
 * @param owner - Repository owner
 * @param repoName - Repository name
 * @param feedbacks - Array of XML feedback strings
 * @returns Builder response with comments and structured suggestions
 */
const xmlResponseBuilder = async (
  owner: string,
  repoName: string,
  feedbacks: string[]
): Promise<BuilderResponse> => {
  console.log("IN XML RESPONSE BUILDER");
  const parsedXMLSuggestions = await processXMLSuggestions(feedbacks);
  const comments = convertPRSuggestionToComment(
    owner,
    repoName,
    dedupSuggestions(parsedXMLSuggestions)
  );
  const commentBlob = comments.join("\n");
  return { comment: commentBlob, structuredComments: parsedXMLSuggestions };
};

/**
 * Creates a curried XML response builder function
 * @param owner - Repository owner
 * @param repoName - Repository name
 * @returns Function that builds XML responses
 */
const curriedXmlResponseBuilder = (owner: string, repoName: string) => {
  return (feedbacks: string[]) =>
    xmlResponseBuilder(owner, repoName, feedbacks);
};

/**
 * Builds a basic response from feedback strings
 * @param feedbacks - Array of feedback strings
 * @returns Builder response with combined comments
 */
const basicResponseBuilder = async (
  feedbacks: string[]
): Promise<BuilderResponse> => {
  console.log("IN BASIC RESPONSE BUILDER");
  const commentBlob = feedbacks.join("\n");
  return { comment: commentBlob, structuredComments: [] };
};

/**
 * Reviews changes in PR files
 * @param files - Array of PR files to review
 * @param convoBuilder - Function to build conversation from diff
 * @param responseBuilder - Function to build response from feedback
 * @returns Builder response with review comments
 */
export const reviewChanges = async (
  files: PRFile[],
  convoBuilder: (diff: string) => ChatCompletionMessageParam[],
  responseBuilder: (responses: string[]) => Promise<BuilderResponse>
) => {
  const patchBuilder = buildPatchPrompt;
  const filteredFiles = files.filter((file) => filterFile(file));
  filteredFiles.map((file) => {
    file.patchTokenLength = getTokenLength(patchBuilder(file));
  });
  const patchesWithinModelLimit: PRFile[] = [];
  const patchesOutsideModelLimit: PRFile[] = [];

  filteredFiles.forEach((file) => {
    const patchWithPromptWithinLimit = isConversationWithinLimit(
      constructPrompt([file], patchBuilder, convoBuilder)
    );
    if (patchWithPromptWithinLimit) {
      patchesWithinModelLimit.push(file);
    } else {
      patchesOutsideModelLimit.push(file);
    }
  });

  console.log(`files within limits: ${patchesWithinModelLimit.length}`);
  const withinLimitsPatchGroups = processWithinLimitFiles(
    patchesWithinModelLimit,
    patchBuilder,
    convoBuilder
  );
  const exceedingLimitsPatchGroups = processOutsideLimitFiles(
    patchesOutsideModelLimit,
    patchBuilder,
    convoBuilder
  );
  console.log(`${withinLimitsPatchGroups.length} within limits groups.`);
  console.log(
    `${patchesOutsideModelLimit.length} files outside limit, skipping them.`
  );

  const groups = [...withinLimitsPatchGroups, ...exceedingLimitsPatchGroups];

  const feedbacks = await Promise.all(
    groups.map((patchGroup) => {
      return reviewFiles(patchGroup, patchBuilder, convoBuilder);
    })
  );
  try {
    return await responseBuilder(feedbacks);
  } catch (exc) {
    console.log("XML parsing error");
    console.log(exc);
    throw exc;
  }
};

/**
 * Indents code to match the indentation of a target line
 * @param file - File contents as string
 * @param code - Code to indent
 * @param lineStart - Starting line number
 * @returns Indented code string
 */
const indentCodeFix = (
  file: string,
  code: string,
  lineStart: number
): string => {
  const fileLines = file.split("\n");
  const firstLine = fileLines[lineStart - 1];
  const codeLines = code.split("\n");
  const indentation = firstLine.match(/^(\s*)/)[0];
  const indentedCodeLines = codeLines.map((line) => indentation + line);
  return indentedCodeLines.join("\n");
};

/**
 * Checks if a code suggestion is different from existing code
 * @param contents - File contents
 * @param suggestion - Code suggestion to check
 * @returns Boolean indicating if suggestion is new
 */
const isCodeSuggestionNew = (
  contents: string,
  suggestion: CodeSuggestion
): boolean => {
  const fileLines = contents.split("\n");
  const targetLines = fileLines
    .slice(suggestion.line_start - 1, suggestion.line_end)
    .join("\n");
  if (targetLines.trim() == suggestion.correction.trim()) {
    return false;
  }
  return true;
};

/**
 * Generates inline comments for a PR suggestion
 * @param suggestion - PR suggestion to process
 * @param file - PR file containing the code
 * @returns Code suggestion or null if generation fails
 */
export const generateInlineComments = async (
  suggestion: PRSuggestion,
  file: PRFile
): Promise<CodeSuggestion> => {
  try {
    const messages = getInlineFixPrompt(file.current_contents, suggestion);
    const { function_call } = await generateChatCompletion({
      messages,
      functions: [INLINE_FIX_FUNCTION],
      function_call: { name: INLINE_FIX_FUNCTION.name },
    });
    if (!function_call) {
      throw new Error("No function call found");
    }
    const args = JSON.parse(function_call.arguments);
    const initialCode = String.raw`${args["code"]}`;
    const indentedCode = indentCodeFix(
      file.current_contents,
      initialCode,
      args["lineStart"]
    );
    const codeFix = {
      file: suggestion.filename,
      line_start: args["lineStart"],
      line_end: args["lineEnd"],
      correction: indentedCode,
      comment: args["comment"],
    };
    if (isCodeSuggestionNew(file.current_contents, codeFix)) {
      return codeFix;
    }
    return null;
  } catch (exc) {
    console.log(exc);
    return null;
  }
};

/**
 * Preprocesses a PR file by fetching its contents
 * @param octokit - Octokit instance
 * @param payload - Pull request webhook payload
 * @param file - PR file to preprocess
 */
const preprocessFile = async (
  octokit: Octokit,
  payload: WebhookEventMap["pull_request"],
  file: PRFile
) => {
  const { base, head } = payload.pull_request;
  const baseBranch: BranchDetails = {
    name: base.ref,
    sha: base.sha,
    url: payload.pull_request.url,
  };
  const currentBranch: BranchDetails = {
    name: head.ref,
    sha: head.sha,
    url: payload.pull_request.url,
  };
  const [oldContents, currentContents] = await Promise.all([
    getGitFile(octokit, payload, baseBranch, file.filename),
    getGitFile(octokit, payload, currentBranch, file.filename),
  ]);

  if (oldContents.content != null) {
    file.old_contents = String.raw`${oldContents.content}`;
  } else {
    file.old_contents = null;
  }

  if (currentContents.content != null) {
    file.current_contents = String.raw`${currentContents.content}`;
  } else {
    file.current_contents = null;
  }
};

/**
 * Retries reviewing changes with different builders
 * @param files - Array of PR files to review
 * @param builders - Array of builder configurations to try
 * @returns Builder response from successful review
 */
const reviewChangesRetry = async (files: PRFile[], builders: Builders[]) => {
  for (const { convoBuilder, responseBuilder } of builders) {
    try {
      console.log(`Trying with convoBuilder: ${convoBuilder.name}.`);
      return await reviewChanges(files, convoBuilder, responseBuilder);
    } catch (error) {
      console.log(
        `Error with convoBuilder: ${convoBuilder.name}, trying next one. Error: ${error}`
      );
    }
  }
  throw new Error("All convoBuilders failed.");
};

/**
 * Main function to process a pull request
 * @param octokit - Octokit instance
 * @param payload - Pull request webhook payload
 * @param files - Array of PR files to process
 * @param includeSuggestions - Whether to include code suggestions
 * @returns Object containing review and suggestions
 */
export const processPullRequest = async (
  octokit: Octokit,
  payload: WebhookEventMap["pull_request"],
  files: PRFile[],
  includeSuggestions = false
) => {
  console.dir({ files }, { depth: null });
  const filteredFiles = files.filter((file) => filterFile(file));
  console.dir({ filteredFiles }, { depth: null });
  if (filteredFiles.length == 0) {
    console.log("Nothing to comment on, all files were filtered out. The PR Agent does not support the following file types: pdf, png, jpg, jpeg, gif, mp4, mp3, md, json, env, toml, svg, package-lock.json, yarn.lock, .gitignore, package.json, tsconfig.json, poetry.lock, readme.md");
    return {
      review: null,
      suggestions: [],
    };
  }
  await Promise.all(
    filteredFiles.map((file) => {
      return preprocessFile(octokit, payload, file);
    })
  );
  const owner = payload.repository.owner.login;
  const repoName = payload.repository.name;
  const curriedXMLResponseBuilder = curriedXmlResponseBuilder(owner, repoName);
  if (includeSuggestions) {
    const reviewComments = await reviewChangesRetry(filteredFiles, [
      {
        convoBuilder: getXMLReviewPrompt,
        responseBuilder: curriedXMLResponseBuilder,
      },
      {
        convoBuilder: getReviewPrompt,
        responseBuilder: basicResponseBuilder,
      },
    ]);
    let inlineComments: CodeSuggestion[] = [];
    if (reviewComments.structuredComments.length > 0) {
      console.log("STARTING INLINE COMMENT PROCESSING");
      inlineComments = await Promise.all(
        reviewComments.structuredComments.map((suggestion) => {
          const file = files.find(
            (file) => file.filename === suggestion.filename
          );
          if (file == null) {
            return null;
          }
          return generateInlineComments(suggestion, file);
        })
      );
    }
    const filteredInlineComments = inlineComments.filter(
      (comment) => comment !== null
    );
    return {
      review: reviewComments,
      suggestions: filteredInlineComments,
    };
  } else {
    const [review] = await Promise.all([
      reviewChangesRetry(filteredFiles, [
        {
          convoBuilder: getXMLReviewPrompt,
          responseBuilder: curriedXMLResponseBuilder,
        },
        {
          convoBuilder: getReviewPrompt,
          responseBuilder: basicResponseBuilder,
        },
      ]),
    ]);

    return {
      review,
      suggestions: [],
    };
  }
};
