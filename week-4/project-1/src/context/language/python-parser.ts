import Parser = require("web-tree-sitter");

/**
 * Interface defining required parser behavior
 * @interface AbstractParser
 * @
 */
interface AbstractParser {
  findEnclosingContext(
    file: string,
    lineStart: number,
    lineEnd: number
  ): Promise<EnclosingContext>;
  dryRun(file: string): Promise<ParserResult>;
}


/**
 * Represents the context that encloses a section of code
 */
interface EnclosingContext {
  /** The node representing the enclosing context, or null if none found */
  enclosingContext: CustomNode | null;
}

/**
 * Custom node representation with essential properties
 */
interface CustomNode {
  /** The type of the node (e.g. 'function_definition', 'class_definition', etc) */
  type: string;
  /** The starting line number (1-based) */
  start: number;
  /** The ending line number (1-based) */
  end: number;
  /** Whether this node delegates to another */
  delegate: boolean;
}

/**
 * Result of attempting to parse Python code
 */
interface ParserResult {
  /** Whether the parse was successful */
  valid: boolean;
  /** Error message if parsing failed, empty string otherwise */
  error: string;
}

/**
 * Parser for Python source code using tree-sitter
 */
export class PythonParser implements AbstractParser {
  /** The underlying tree-sitter Parser instance */
  private parser: Parser | null = null;
  /** Whether the parser has been initialized */
  private initialized = false;

  /**
   * Initializes the tree-sitter parser with Python grammar
   * Loads the WASM binary and sets up the parser for Python
   * Only initializes once, subsequent calls are no-ops
   */
  private async initialize() {
    if (this.initialized) return;
    
    await Parser.init();
    this.parser = new Parser();
    
    // Load the Python grammar
    const pythonWasm = await fetch('tree-sitter-python.wasm');
    const wasmBuffer = await pythonWasm.arrayBuffer();
    const Language = await Parser.Language.load(new Uint8Array(wasmBuffer));
    
    this.parser.setLanguage(Language);
    this.initialized = true;
  }

  /**
   * Converts a tree-sitter SyntaxNode to our CustomNode format
   * @param node - The tree-sitter node to convert
   * @returns A CustomNode representation of the input node
   */
  private nodeToCustomNode(node: Parser.SyntaxNode): CustomNode {
    return {
      type: node.type,
      start: node.startPosition.row + 1,
      end: node.endPosition.row + 1,
      delegate: false
    };
  }

  /**
   * Determines if a node type is relevant for our analysis
   * @param type - The type of the node to check
   * @returns True if the node type is one we care about
   */
  private isRelevantNode(type: string): boolean {
    return [
      'function_definition',
      'class_definition',
      'with_statement',
      'for_statement',
      'while_statement',
      'if_statement',
      'try_statement'
    ].includes(type);
  }

  /**
   * Finds the largest node that completely contains the given line range
   * @param root - The root node to start searching from
   * @param lineStart - The starting line number (1-based)
   * @param lineEnd - The ending line number (1-based)
   * @returns The largest enclosing node, or null if none found
   */
  private findLargestEnclosingNode(
    root: Parser.SyntaxNode,
    lineStart: number,
    lineEnd: number
  ): Parser.SyntaxNode | null {
    let largestNode: Parser.SyntaxNode | null = null;
    let largestSize = 0;

    const cursor = root.walk();
    
    const visit = (): void => {
      const node = cursor.currentNode;
      
      if (
        this.isRelevantNode(node.type) &&
        node.startPosition.row + 1 <= lineStart &&
        node.endPosition.row + 1 >= lineEnd
      ) {
        const size = node.endPosition.row - node.startPosition.row;
        if (size > largestSize) {
          largestSize = size;
          largestNode = node;
        }
      }

      if (cursor.gotoFirstChild()) {
        do {
          visit();
        } while (cursor.gotoNextSibling());
        cursor.gotoParent();
      }
    };

    visit();
    return largestNode;
  }

  /**
   * Finds the largest enclosing context for a given line range in Python source code
   * @param file - The Python source code as a string
   * @param lineStart - The starting line number (1-based)
   * @param lineEnd - The ending line number (1-based)
   * @returns Promise resolving to an EnclosingContext
   * @throws Error if parser initialization fails
   */
  async findEnclosingContext(
    file: string,
    lineStart: number,
    lineEnd: number
  ): Promise<EnclosingContext> {
    try {
      await this.initialize();
      if (!this.parser) throw new Error('Parser not initialized');

      const tree = this.parser.parse(file);
      const largestNode = this.findLargestEnclosingNode(tree.rootNode, lineStart, lineEnd);

      return {
        enclosingContext: largestNode ? this.nodeToCustomNode(largestNode) : null
      };
    } catch (error) {
      console.error('Error parsing Python file:', error);
      return { enclosingContext: null };
    }
  }

  /**
   * Attempts to parse Python source code to verify it's valid
   * @param file - The Python source code to validate
   * @returns Promise resolving to a ParserResult indicating success/failure
   * @throws Error if parser initialization fails
   */
  async dryRun(file: string): Promise<ParserResult> {
    try {
      await this.initialize();
      if (!this.parser) throw new Error('Parser not initialized');

      this.parser.parse(file);
      return {
        valid: true,
        error: ''
      };
    } catch (error) {
      return {
        valid: false,
        error: error instanceof Error ? error.message : String(error)
      };
    }
  }
}