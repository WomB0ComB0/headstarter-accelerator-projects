import { Node } from "@babel/traverse";
import { JavascriptParser } from "./context/language/javascript-parser";
import { ChatCompletionMessageParam } from "groq-sdk/resources/chat/completions";

/**
 * Represents a file in a Pull Request with its metadata and contents
 * @interface PRFile
 * @property {string} sha - The SHA hash of the file
 * @property {string} filename - Name of the file
 * @property {('added'|'removed'|'renamed'|'changed'|'modified'|'copied'|'unchanged')} status - Current status of the file in the PR
 * @property {number} additions - Number of lines added
 * @property {number} deletions - Number of lines deleted
 * @property {number} changes - Total number of changes
 * @property {string} blob_url - URL to the file's blob
 * @property {string} raw_url - URL to the raw file content
 * @property {string} contents_url - URL to the file's contents
 * @property {string} [patch] - Optional patch information
 * @property {string} [previous_filename] - Previous filename if renamed
 * @property {number} [patchTokenLength] - Length of the patch in tokens
 * @property {string} [old_contents] - Previous contents of the file
 * @property {string} [current_contents] - Current contents of the file
 */
export interface PRFile {
  sha: string;
  filename: string;
  status:
    | "added"
    | "removed"
    | "renamed"
    | "changed"
    | "modified"
    | "copied"
    | "unchanged";
  additions: number;
  deletions: number;
  changes: number;
  blob_url: string;
  raw_url: string;
  contents_url: string;
  patch?: string;
  previous_filename?: string;
  patchTokenLength?: number;
  old_contents?: string;
  current_contents?: string;
}

/**
 * Response structure from the builder containing comments
 * @interface BuilderResponse
 * @property {string} comment - The main comment text
 * @property {any[]} structuredComments - Array of structured comment objects
 */
export interface BuilderResponse {
  comment: string;
  structuredComments: any[];
}

/**
 * Interface defining builder functions for conversation and response handling
 * @interface Builders
 * @property {function} convoBuilder - Builds conversation messages from a diff
 * @property {function} responseBuilder - Builds a response from feedback array
 */
export interface Builders {
  convoBuilder: (diff: string) => ChatCompletionMessageParam[];
  responseBuilder: (feedbacks: string[]) => Promise<BuilderResponse>;
}

/**
 * Information about code patches/changes
 * @interface PatchInfo
 * @property {Object[]} hunks - Array of code change hunks
 * @property {number} hunks[].oldStart - Starting line number in old version
 * @property {number} hunks[].oldLines - Number of lines in old version
 * @property {number} hunks[].newStart - Starting line number in new version
 * @property {number} hunks[].newLines - Number of lines in new version
 * @property {string[]} hunks[].lines - Array of changed lines
 */
export interface PatchInfo {
  hunks: {
    oldStart: number;
    oldLines: number;
    newStart: number;
    newLines: number;
    lines: string[];
  }[];
}

/**
 * Represents a suggestion for a Pull Request
 * @interface PRSuggestion
 * @property {string} describe - Description of the suggestion
 * @property {string} type - Type of suggestion
 * @property {string} comment - Comment explaining the suggestion
 * @property {string} code - The suggested code
 * @property {string} filename - File where suggestion applies
 * @property {function} toString - Converts suggestion to string
 * @property {function} identity - Returns unique identifier
 */
export interface PRSuggestion {
  describe: string;
  type: string;
  comment: string;
  code: string;
  filename: string;
  toString: () => string;
  identity: () => string;
}

/**
 * Represents a code correction suggestion
 * @interface CodeSuggestion
 * @property {string} file - File path where correction applies
 * @property {number} line_start - Starting line number
 * @property {number} line_end - Ending line number
 * @property {string} correction - The suggested correction
 * @property {string} comment - Explanation of the correction
 */
export interface CodeSuggestion {
  file: string;
  line_start: number;
  line_end: number;
  correction: string;
  comment: string;
}

/**
 * Represents a chat message
 * @interface ChatMessage
 * @property {string} role - Role of the message sender
 * @property {string} content - Content of the message
 */
export interface ChatMessage {
  role: string;
  content: string;
}

/**
 * Contains review information and suggestions
 * @interface Review
 * @property {BuilderResponse} review - The review response
 * @property {CodeSuggestion[]} suggestions - Array of code suggestions
 */
export interface Review {
  review: BuilderResponse;
  suggestions: CodeSuggestion[];
}

/**
 * Details about a git branch
 * @interface BranchDetails
 * @property {string} name - Name of the branch
 * @property {string} sha - SHA hash of the branch
 * @property {string} url - URL to the branch
 */
export interface BranchDetails {
  name: string;
  sha: string;
  url: string;
}

/**
 * Delays execution for specified milliseconds
 * @async
 * @param {number} ms - Number of milliseconds to sleep
 * @returns {Promise<void>}
 */
export const sleep = async (ms: number) => {
  return new Promise((resolve) => setTimeout(resolve, ms));
};

/**
 * Processes and normalizes a git filepath
 * @param {string} filepath - The filepath to process
 * @returns {string} Normalized filepath
 */
export const processGitFilepath = (filepath: string) => {
  // Remove the leading '/' if it exists
  return filepath.startsWith("/") ? filepath.slice(1) : filepath;
};

/**
 * Represents the enclosing context of code
 * @interface EnclosingContext
 * @property {Node|null} enclosingContext - The AST node representing enclosing context
 */
export interface EnclosingContext {
  enclosingContext: Node | null;
}

/**
 * Interface for code parsers
 * @interface AbstractParser
 * @property {function} findEnclosingContext - Finds enclosing context for code
 * @property {function} dryRun - Validates file parsing
 */
export interface AbstractParser {
  findEnclosingContext(
    file: string,
    lineStart: number,
    lineEnd: number
  ): EnclosingContext;
  dryRun(file: string): { valid: boolean; error: string };
}

/**
 * Maps file extensions to their corresponding parsers
 * @constant
 * @type {Map<string, AbstractParser>}
 */
const EXTENSIONS_TO_PARSERS: Map<string, AbstractParser> = new Map([
  ["ts", new JavascriptParser()],
  ["tsx", new JavascriptParser()],
  ["js", new JavascriptParser()],
  ["jsx", new JavascriptParser()],
]);

/**
 * Gets the appropriate parser for a file based on its extension
 * @param {string} filename - Name of the file
 * @returns {AbstractParser|null} Parser instance or null if no parser found
 */
export const getParserForExtension = (filename: string) => {
  const fileExtension = filename.split(".").pop().toLowerCase();
  return EXTENSIONS_TO_PARSERS.get(fileExtension) || null;
};

/**
 * Adds line numbers to content string
 * @param {string} contents - Content to add line numbers to
 * @returns {string} Content with line numbers prepended
 */
export const assignLineNumbers = (contents: string): string => {
  const lines = contents.split("\n");
  let lineNumber = 1;
  const linesWithNumbers = lines.map((line) => {
    const numberedLine = `${lineNumber}: ${line}`;
    lineNumber++;
    return numberedLine;
  });
  return linesWithNumbers.join("\n");
};
