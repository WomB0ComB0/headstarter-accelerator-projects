import { AbstractParser, EnclosingContext } from "../../constants";
import * as parser from "@babel/parser";
import traverse, { NodePath, Node } from "@babel/traverse";

/**
 * Processes an AST node to find the largest enclosing context for a given line range
 * @param path - The current node path being traversed
 * @param lineStart - Starting line number to check for enclosing context
 * @param lineEnd - Ending line number to check for enclosing context 
 * @param largestSize - Current largest node size found
 * @param largestEnclosingContext - Current largest enclosing node found
 * @returns Object containing updated largestSize and largestEnclosingContext
 */
const processNode = (
  path: NodePath<Node>,
  lineStart: number,
  lineEnd: number,
  largestSize: number,
  largestEnclosingContext: Node | null
) => {
  const { start, end } = path.node.loc;
  if (start.line <= lineStart && lineEnd <= end.line) {
    const size = end.line - start.line;
    if (size > largestSize) {
      largestSize = size;
      largestEnclosingContext = path.node;
    }
  }
  return { largestSize, largestEnclosingContext };
};

/**
 * Parser implementation for JavaScript/TypeScript files
 * Uses @babel/parser to parse source code and traverse AST to find enclosing contexts
 * @implements {AbstractParser}
 */
export class JavascriptParser implements AbstractParser {
  /**
   * Finds the largest enclosing context (function, interface etc) that contains the given line range
   * @param file - Source code content as string
   * @param lineStart - Starting line number to find context for
   * @param lineEnd - Ending line number to find context for
   * @returns EnclosingContext containing the found node, or null if none found
   * @throws Will throw if parsing fails
   */
  findEnclosingContext(
    file: string,
    lineStart: number,
    lineEnd: number
  ): EnclosingContext {
    const ast = parser.parse(file, {
      sourceType: "module",
      plugins: ["jsx", "typescript"],
    });
    let largestEnclosingContext: Node | null = null;
    let largestSize = 0;
    traverse(ast, {
      Function(path: NodePath<Node>) {
        ({ largestSize, largestEnclosingContext } = processNode(
          path,
          lineStart,
          lineEnd,
          largestSize,
          largestEnclosingContext
        ));
      },
      TSInterfaceDeclaration(path: NodePath<Node>) {
        ({ largestSize, largestEnclosingContext } = processNode(
          path,
          lineStart,
          lineEnd,
          largestSize,
          largestEnclosingContext
        ));
      },
    });
    return {
      enclosingContext: largestEnclosingContext,
    } as EnclosingContext;
  }

  /**
   * Validates if a file can be successfully parsed
   * @param file - Source code content to validate
   * @returns Object containing:
   *   - valid: boolean indicating if parse was successful
   *   - error: empty string if valid, error message if invalid
   */
  dryRun(file: string): { valid: boolean; error: string } {
    try {
      const ast = parser.parse(file, {
        sourceType: "module",
        plugins: ["jsx", "typescript"],
      });
      return {
        valid: true,
        error: "",
      };
    } catch (exc) {
      return {
        valid: false,
        error: exc,
      };
    }
  }
}
