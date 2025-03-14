import { ChatCompletionMessageParam } from "groq-sdk/resources/chat/completions";
import { PRSuggestion } from "../constants";

/**
 * System prompt template for generating inline code fixes. Instructs the LLM on how to process
 * code suggestions and generate appropriate fixes.
 * 
 * The prompt expects an XML-formatted suggestion containing:
 * - describe: Description of the issue
 * - type: Category of the fix
 * - comment: Specific modification instructions
 * - code: Original problematic code
 * - filename: File where fix should be applied
 * 
 * Along with the full file contents where the fix will be applied.
 */
export const INLINE_FIX_PROMPT = `In this task, you are provided with a code suggestion in XML format, along with the corresponding file content. Your task is to radiate from this suggestion and draft a precise code fix. Here's how your input will look:

\`\`\`xml
  <suggestion>
    <describe>Your Description Here</describe>
    <type>Your Type Here</type>
    <comment>Your Suggestions Here</comment>
    <code>Original Code Here</code>
    <filename>File Name Here</filename>
  </suggestion>
\`\`\`

{file}

The 'comment' field contains specific code modification instructions. Based on these instructions, you're required to formulate a precise code fix. Bear in mind that the fix must include only the lines between the starting line (linestart) and ending line (lineend) where the changes are applied.

The adjusted code doesn't necessarily need to be standalone valid code, but when incorporated into the corresponding file, it must result in valid, functional code, without errors. Ensure to include only the specific lines affected by the modifications. Avoid including placeholders such as 'rest of code...'

Please interpret the given directions and apply the necessary changes to the provided suggestion and file content. Make the modifications unambiguous and appropriate for utilizing in an inline suggestion on GitHub.`;

/**
 * Function definition for the LLM to structure its code fix response.
 * Specifies the expected format and required fields for the fix.
 */
export const INLINE_FIX_FUNCTION = {
  name: "fix",
  description: "The code fix to address the suggestion and rectify the issue",
  parameters: {
    type: "object",
    properties: {
      comment: {
        type: "string",
        description: "Why this change improves the code",
      },
      code: {
        type: "string",
        description: "Modified Code Snippet",
      },
      lineStart: {
        type: "number",
        description: "Starting Line Number",
      },
      lineEnd: {
        type: "number",
        description: "Ending Line Number",
      },
    },
  },
  required: ["action"],
};

/**
 * Template for constructing the user message that will be sent to the LLM.
 * Combines the suggestion and file contents in a structured format.
 */
const INLINE_USER_MESSAGE_TEMPLATE = `{SUGGESTION}

{FILE}`;

/**
 * Adds line numbers to each line of the provided file contents.
 * This helps with precise line referencing in the LLM's response.
 * 
 * @param contents - The file contents as a single string
 * @returns The file contents with line numbers prefixed to each line
 * @example
 * ```ts
 * const numbered = assignFullLineNumers("line1\nline2");
 * // Returns:
 * // "1: line1
 * //  2: line2"
 * ```
 */
const assignFullLineNumers = (contents: string): string => {
  const lines = contents.split("\n");
  let lineNumber = 1;
  const linesWithNumbers = lines.map((line) => {
    const numberedLine = `${lineNumber}: ${line}`;
    lineNumber++;
    return numberedLine;
  });
  return linesWithNumbers.join("\n");
};

/**
 * Generates the complete prompt messages for requesting an inline code fix from the LLM.
 * Combines the system prompt with a user message containing the suggestion and numbered file contents.
 * 
 * @param fileContents - The full contents of the file where the fix will be applied
 * @param suggestion - The PRSuggestion object containing the issue details and fix instructions
 * @returns An array of chat messages ready to be sent to the LLM
 * @example
 * ```ts
 * const messages = getInlineFixPrompt(fileContents, suggestion);
 * const response = await llm.chat(messages);
 * ```
 */
export const getInlineFixPrompt = (
  fileContents: string,
  suggestion: PRSuggestion
): ChatCompletionMessageParam[] => {
  const userMessage = INLINE_USER_MESSAGE_TEMPLATE.replace(
    "{SUGGESTION}",
    suggestion.toString()
  ).replace("{FILE}", assignFullLineNumers(fileContents));
  return [
    { role: "system", content: INLINE_FIX_PROMPT },
    { role: "user", content: userMessage },
  ];
};
