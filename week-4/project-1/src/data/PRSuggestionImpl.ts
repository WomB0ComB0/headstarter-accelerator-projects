import { PRSuggestion } from "../constants";

/**
 * Implementation of the PRSuggestion interface that represents a code review suggestion
 * for a pull request. Contains metadata and content about the suggestion.
 * @implements {PRSuggestion}
 */
export class PRSuggestionImpl implements PRSuggestion {
  /** Description of what the suggestion addresses or fixes */
  describe: string;

  /** Category/classification of the suggestion (e.g. "bug", "improvement", etc) */
  type: string;

  /** The actual suggestion comment text */
  comment: string;

  /** Code snippet relevant to the suggestion */
  code: string;

  /** Path to the file this suggestion relates to */
  filename: string;

  /**
   * Creates a new PRSuggestionImpl instance
   * @param describe - Description of what the suggestion addresses
   * @param type - Category/classification of the suggestion
   * @param comment - The suggestion comment text
   * @param code - Relevant code snippet
   * @param filename - Path to the related file
   */
  constructor(
    describe: string,
    type: string,
    comment: string,
    code: string,
    filename: string
  ) {
    this.describe = describe;
    this.type = type;
    this.comment = comment;
    this.code = code;
    this.filename = filename;
  }

  /**
   * Converts the suggestion into an XML string representation
   * @returns XML formatted string containing all suggestion fields
   * @example
   * ```xml
   * <suggestion>
   *   <describe>Fix null check</describe>
   *   <type>bug</type>
   *   <comment>Add null check for user param</comment>
   *   <code>if (user === null) return;</code>
   *   <filename>src/user.ts</filename>
   * </suggestion>
   * ```
   */
  toString(): string {
    const xmlElements = [
      `<suggestion>`,
      `  <describe>${this.describe}</describe>`,
      `  <type>${this.type}</type>`,
      `  <comment>${this.comment}</comment>`,
      `  <code>${this.code}</code>`,
      `  <filename>${this.filename}</filename>`,
      `</suggestion>`,
    ];
    return xmlElements.join("\n");
  }

  /**
   * Generates a unique identifier for this suggestion
   * @returns String combining filename and comment for uniqueness
   * @example
   * ```ts
   * // Returns "src/user.ts:Add null check for user param"
   * suggestion.identity()
   * ```
   */
  identity(): string {
    return `${this.filename}:${this.comment}`;
  }
}
