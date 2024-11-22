/**
 * @fileoverview Provides functionality for interacting with GitHub's API to manage reviews, comments, and branches.
 * Contains utilities for posting review comments, managing file contents, and creating branches.
 */

import {
  BranchDetails,
  BuilderResponse,
  CodeSuggestion,
  Review,
  processGitFilepath,
} from "./constants";
import { Octokit } from "@octokit/rest";
import { WebhookEventMap } from "@octokit/webhooks-definitions/schema";

/**
 * Posts a general review comment on a pull request.
 * 
 * @async
 * @param {Octokit} octokit - The authenticated Octokit instance
 * @param {WebhookEventMap["pull_request"]} payload - The pull request webhook payload
 * @param {string} review - The review comment text to post
 * @returns {Promise<void>}
 */
const postGeneralReviewComment = async (
  octokit: Octokit,
  payload: WebhookEventMap["pull_request"],
  review: string
) => {
  try {
    await octokit.request(
      "POST /repos/{owner}/{repo}/issues/{issue_number}/comments",
      {
        owner: payload.repository.owner.login,
        repo: payload.repository.name,
        issue_number: payload.pull_request.number,
        body: review,
        headers: {
          "x-github-api-version": "2022-11-28",
        },
      }
    );
  } catch (exc) {
    console.log(exc);
  }
};

/**
 * Posts an inline code suggestion comment on a specific line or range in a pull request.
 * 
 * @async
 * @param {Octokit} octokit - The authenticated Octokit instance
 * @param {WebhookEventMap["pull_request"]} payload - The pull request webhook payload
 * @param {CodeSuggestion} suggestion - The code suggestion details including location and correction
 * @returns {Promise<void>}
 */
const postInlineComment = async (
  octokit: Octokit,
  payload: WebhookEventMap["pull_request"],
  suggestion: CodeSuggestion
) => {
  try {
    const line = suggestion.line_end;
    let startLine = null;
    if (suggestion.line_end != suggestion.line_start) {
      startLine = suggestion.line_start;
    }
    const suggestionBody = `${suggestion.comment}\n\`\`\`suggestion\n${suggestion.correction}`;

    await octokit.request(
      "POST /repos/{owner}/{repo}/pulls/{pull_number}/comments",
      {
        owner: payload.repository.owner.login,
        repo: payload.repository.name,
        pull_number: payload.pull_request.number,
        body: suggestionBody,
        commit_id: payload.pull_request.head.sha,
        path: suggestion.file,
        line: line,
        ...(startLine ? { start_line: startLine } : {}),
        start_side: "RIGHT",
        side: "RIGHT",
        headers: {
          "X-GitHub-Api-Version": "2022-11-28",
        },
      }
    );
  } catch (exc) {
    console.log(exc);
  }
};

/**
 * Applies a complete review to a pull request, including general comments and inline suggestions.
 * 
 * @async
 * @param {Object} params - The parameters object
 * @param {Octokit} params.octokit - The authenticated Octokit instance
 * @param {WebhookEventMap["pull_request"]} params.payload - The pull request webhook payload
 * @param {Review} params.review - The complete review containing comments and suggestions
 * @returns {Promise<void>}
 */
export const applyReview = async ({
  octokit,
  payload,
  review,
}: {
  octokit: Octokit;
  payload: WebhookEventMap["pull_request"];
  review: Review;
}) => {
  let commentPromise = null;
  const comment = review.review?.comment;
  if (comment != null) {
    commentPromise = postGeneralReviewComment(octokit, payload, comment);
  }
  const suggestionPromises = review.suggestions.map((suggestion) =>
    postInlineComment(octokit, payload, suggestion)
  );
  await Promise.all([
    ...(commentPromise ? [commentPromise] : []),
    ...suggestionPromises,
  ]);
};

/**
 * Adds line numbers to file contents for better readability.
 * 
 * @param {string} contents - The raw file contents
 * @returns {string} The file contents with line numbers prepended
 */
const addLineNumbers = (contents: string) => {
  const rawContents = String.raw`${contents}`;
  const prepended = rawContents
    .split("\n")
    .map((line, idx) => `${idx + 1}: ${line}`)
    .join("\n");
  return prepended;
};

/**
 * Retrieves a file's contents from GitHub repository.
 * 
 * @async
 * @param {Octokit} octokit - The authenticated Octokit instance
 * @param {WebhookEventMap["issues"] | WebhookEventMap["pull_request"]} payload - The webhook event payload
 * @param {BranchDetails} branch - The branch details containing name and SHA
 * @param {string} filepath - Path to the file in the repository
 * @returns {Promise<{content: string | null, sha: string | null}>} The file contents and SHA
 * @throws {Error} When API request fails for reasons other than 404
 */
export const getGitFile = async (
  octokit: Octokit,
  payload: WebhookEventMap["issues"] | WebhookEventMap["pull_request"],
  branch: BranchDetails,
  filepath: string
) => {
  try {
    const response = await octokit.request(
      "GET /repos/{owner}/{repo}/contents/{path}",
      {
        owner: payload.repository.owner.login,
        repo: payload.repository.name,
        path: filepath,
        ref: branch.name,
        headers: {
          "X-GitHub-Api-Version": "2022-11-28",
        },
      }
    );
    const decodedContent = Buffer.from(
      //@ts-ignore
      response.data.content,
      "base64"
    ).toString("utf8");
    //@ts-ignore
    return { content: decodedContent, sha: response.data.sha };
  } catch (exc) {
    if (exc.status === 404) {
      return { content: null, sha: null };
    }
    console.log(exc);
    throw exc;
  }
};

/**
 * Retrieves and formats file contents with line numbers and header.
 * 
 * @async
 * @param {Octokit} octokit - The authenticated Octokit instance
 * @param {WebhookEventMap["issues"]} payload - The issues webhook payload
 * @param {BranchDetails} branch - The branch details
 * @param {string} filepath - Path to the file
 * @returns {Promise<{result: string, functionString: string}>} Formatted file contents and description
 */
export const getFileContents = async (
  octokit: Octokit,
  payload: WebhookEventMap["issues"],
  branch: BranchDetails,
  filepath: string
) => {
  const gitFile = await getGitFile(
    octokit,
    payload,
    branch,
    processGitFilepath(filepath)
  );
  const fileWithLines = `# ${filepath}\n${addLineNumbers(gitFile.content)}`;
  return { result: fileWithLines, functionString: `Opening file: ${filepath}` };
};

/**
 * Posts a comment on a GitHub issue.
 * 
 * @async
 * @param {Octokit} octokit - The authenticated Octokit instance
 * @param {WebhookEventMap["issues"]} payload - The issues webhook payload
 * @param {string} comment - The comment text to post
 * @returns {Promise<void>}
 */
export const commentIssue = async (
  octokit: Octokit,
  payload: WebhookEventMap["issues"],
  comment: string
) => {
  await octokit.rest.issues.createComment({
    owner: payload.repository.owner.login,
    repo: payload.repository.name,
    issue_number: payload.issue.number,
    body: comment,
  });
};

/**
 * Creates a new branch in the repository based on the default branch.
 * Generates a unique branch name using the issue title and random hash.
 * Posts a comment on the issue with the new branch URL.
 * 
 * @async
 * @param {Octokit} octokit - The authenticated Octokit instance
 * @param {WebhookEventMap["issues"]} payload - The issues webhook payload
 * @returns {Promise<BranchDetails | null>} The created branch details or null if creation fails
 */
export const createBranch = async (
  octokit: Octokit,
  payload: WebhookEventMap["issues"]
) => {
  let branchDetails = null;
  try {
    const title = payload.issue.title.replace(/\s/g, "-").substring(0, 15);

    const hash = Math.random().toString(36).substring(2, 7);
    const subName = `${title}-${hash}`.substring(0, 20);
    const branchName = `Code-Bot/${subName}`;
    // Get the default branch for the repository
    const { data: repo } = await octokit.rest.repos.get({
      owner: payload.repository.owner.login,
      repo: payload.repository.name,
    });

    // Get the commit SHA of the default branch
    const { data: ref } = await octokit.rest.git.getRef({
      owner: payload.repository.owner.login,
      repo: payload.repository.name,
      ref: `heads/${repo.default_branch}`,
    });

    // Create a new branch from the commit SHA
    const { data: newBranch } = await octokit.rest.git.createRef({
      owner: payload.repository.owner.login,
      repo: payload.repository.name,
      ref: `refs/heads/${branchName}`,
      sha: ref.object.sha,
    });

    console.log(newBranch);

    branchDetails = {
      name: branchName,
      sha: newBranch.object.sha,
      url: newBranch.url,
    };
    let branchUrl = `https://github.com/${payload.repository.owner.login}/${payload.repository.name}/tree/${branchName}`;
    const branchComment = `Branch created: [${branchName}](${branchUrl})`;
    await commentIssue(octokit, payload, branchComment);

    console.log(`Branch ${branchName} created`);
  } catch (exc) {
    console.log(exc);
  }
  return branchDetails;
};
