/**
 * @fileoverview Provides functionality for handling and formatting code diffs and patches
 * with intelligent context expansion and function-level scope awareness.
 */

import {
  AbstractParser,
  PRFile,
  PatchInfo,
  getParserForExtension,
} from "../constants";
import * as diff from "diff";
import { JavascriptParser } from "./language/javascript-parser";
import { Node } from "@babel/traverse";

/**
 * Expands a diff hunk to include additional context lines above and below
 * @param contents - The full file contents as a string
 * @param hunk - The diff hunk to expand
 * @param linesAbove - Number of context lines to add above the hunk (default: 5)
 * @param linesBelow - Number of context lines to add below the hunk (default: 5)
 * @returns The expanded hunk as a string with added context
 */
const expandHunk = (
  contents: string,
  hunk: diff.Hunk,
  linesAbove: number = 5,
  linesBelow: number = 5
) => {
  const fileLines = contents.split("\n");
  const curExpansion: string[] = [];
  const start = Math.max(0, hunk.oldStart - 1 - linesAbove);
  const end = Math.min(
    fileLines.length,
    hunk.oldStart - 1 + hunk.oldLines + linesBelow
  );

  for (let i = start; i < hunk.oldStart - 1; i++) {
    curExpansion.push(fileLines[i]);
  }

  curExpansion.push(
    `@@ -${hunk.oldStart},${hunk.oldLines} +${hunk.newStart},${hunk.newLines} @@`
  );
  hunk.lines.forEach((line) => {
    if (!curExpansion.includes(line)) {
      curExpansion.push(line);
    }
  });

  for (let i = hunk.oldStart - 1 + hunk.oldLines; i < end; i++) {
    curExpansion.push(fileLines[i]);
  }
  return curExpansion.join("\n");
};

/**
 * Expands all hunks in a file's patch to include additional context lines
 * @param file - The PR file containing the patch and original contents
 * @param linesAbove - Number of context lines to add above each hunk (default: 5)
 * @param linesBelow - Number of context lines to add below each hunk (default: 5)
 * @returns Array of expanded hunks, each as an array of strings
 */
const expandFileLines = (
  file: PRFile,
  linesAbove: number = 5,
  linesBelow: number = 5
) => {
  const fileLines = file.old_contents.split("\n");
  const patches: PatchInfo[] = diff.parsePatch(file.patch);
  const expandedLines: string[][] = [];
  patches.forEach((patch) => {
    patch.hunks.forEach((hunk) => {
      const curExpansion: string[] = [];
      const start = Math.max(0, hunk.oldStart - 1 - linesAbove);
      const end = Math.min(
        fileLines.length,
        hunk.oldStart - 1 + hunk.oldLines + linesBelow
      );

      for (let i = start; i < hunk.oldStart - 1; i++) {
        curExpansion.push(fileLines[i]);
      }

      curExpansion.push(
        `@@ -${hunk.oldStart},${hunk.oldLines} +${hunk.newStart},${hunk.newLines} @@`
      );
      hunk.lines.forEach((line) => {
        if (!curExpansion.includes(line)) {
          curExpansion.push(line);
        }
      });

      for (let i = hunk.oldStart - 1 + hunk.oldLines; i < end; i++) {
        curExpansion.push(fileLines[i]);
      }
      expandedLines.push(curExpansion);
    });
  });

  return expandedLines;
};

/**
 * Formats a PR file's changes with expanded context around each hunk
 * @param file - The PR file to format
 * @returns Formatted string containing filename and expanded patch
 */
export const expandedPatchStrategy = (file: PRFile) => {
  const expandedPatches = expandFileLines(file);
  const expansions = expandedPatches
    .map((patchLines) => patchLines.join("\n"))
    .join("\n\n");
  return `## ${file.filename}\n\n${expansions}`;
};

/**
 * Formats a PR file's changes using the raw patch without additional context
 * @param file - The PR file to format
 * @returns Formatted string containing filename and raw patch
 */
export const rawPatchStrategy = (file: PRFile) => {
  return `## ${file.filename}\n\n${file.patch}`;
};

/**
 * Trims a diff hunk to only include lines with actual changes
 * @param hunk - The diff hunk to trim
 * @returns New hunk object containing only changed lines
 */
const trimHunk = (hunk: diff.Hunk): diff.Hunk => {
  const startIdx = hunk.lines.findIndex(
    (line) => line.startsWith("+") || line.startsWith("-")
  );
  const endIdx = hunk.lines
    .slice()
    .reverse()
    .findIndex((line) => line.startsWith("+") || line.startsWith("-"));
  const editLines = hunk.lines.slice(startIdx, hunk.lines.length - endIdx);
  return { ...hunk, lines: editLines, newStart: startIdx + hunk.newStart };
};

/**
 * Builds a string representation of a code scope (e.g. function) with applied changes
 * @param currentFile - Current state of the file as a string
 * @param scope - AST node representing the scope
 * @param hunk - Diff hunk containing changes
 * @returns Array of strings representing the modified scope
 */
const buildingScopeString = (
  currentFile: string,
  scope: Node,
  hunk: diff.Hunk
) => {
  const res: string[] = [];
  const trimmedHunk = trimHunk(hunk);
  const functionStartLine = scope.loc.start.line;
  const functionEndLine = scope.loc.end.line;
  const updatedFileLines = currentFile.split("\n");
  const functionContext = updatedFileLines.slice(
    functionStartLine - 1,
    functionEndLine
  );
  const injectionIdx =
    hunk.newStart -
    functionStartLine +
    hunk.lines.findIndex(
      (line) => line.startsWith("+") || line.startsWith("-")
    );
  const dropCount = trimmedHunk.lines.filter(
    (line) => !line.startsWith("-")
  ).length;

  const hunkHeader = `@@ -${hunk.oldStart},${hunk.oldLines} +${hunk.newStart},${hunk.newLines} @@`;
  functionContext.splice(injectionIdx, dropCount, ...trimmedHunk.lines);

  res.push(functionContext.join("\n"));
  res.unshift(hunkHeader);
  return res;
};

/**
 * Combines multiple overlapping diff hunks into a single hunk
 * @param file - Current file contents as string
 * @param overlappingHunks - Array of hunks that overlap
 * @returns Combined hunk containing all changes
 * @throws Error if overlappingHunks is empty
 */
const combineHunks = (
  file: string,
  overlappingHunks: diff.Hunk[]
): diff.Hunk => {
  if (!overlappingHunks || overlappingHunks.length === 0) {
    throw "Overlapping hunks are empty, this should never happen.";
  }
  const sortedHunks = overlappingHunks.sort((a, b) => a.newStart - b.newStart);
  const fileLines = file.split("\n");
  let lastHunkEnd = sortedHunks[0].newStart + sortedHunks[0].newLines;

  const combinedHunk: diff.Hunk = {
    oldStart: sortedHunks[0].oldStart,
    oldLines: sortedHunks[0].oldLines,
    newStart: sortedHunks[0].newStart,
    newLines: sortedHunks[0].newLines,
    lines: [...sortedHunks[0].lines],
    linedelimiters: [...sortedHunks[0].linedelimiters],
  };

  for (let i = 1; i < sortedHunks.length; i++) {
    const hunk = sortedHunks[i];

    if (hunk.newStart > lastHunkEnd) {
      combinedHunk.lines.push(
        ...fileLines.slice(lastHunkEnd - 1, hunk.newStart - 1)
      );
      combinedHunk.newLines += hunk.newStart - lastHunkEnd;
    }

    combinedHunk.oldLines += hunk.oldLines;
    combinedHunk.newLines += hunk.newLines;
    combinedHunk.lines.push(...hunk.lines);
    combinedHunk.linedelimiters.push(...hunk.linedelimiters);

    lastHunkEnd = hunk.newStart + hunk.newLines;
  }
  return combinedHunk;
};

/**
 * Generates diff context for each hunk in a file using function-level scope awareness
 * @param file - The PR file to process
 * @param parser - Language-specific parser to use for scope analysis
 * @returns Array of context strings for each hunk
 */
const diffContextPerHunk = (file: PRFile, parser: AbstractParser) => {
  const updatedFile = diff.applyPatch(file.old_contents, file.patch);
  const patches = diff.parsePatch(file.patch);
  if (!updatedFile || typeof updatedFile !== "string") {
    console.log("APPLYING PATCH ERROR - FALLINGBACK");
    throw "THIS SHOULD NOT HAPPEN!";
  }

  const hunks: diff.Hunk[] = [];
  const order: number[] = [];
  const scopeRangeHunkMap = new Map<string, diff.Hunk[]>();
  const scopeRangeNodeMap = new Map<string, Node>();
  const expandStrategy: diff.Hunk[] = [];

  patches.forEach((p) => {
    p.hunks.forEach((hunk) => {
      hunks.push(hunk);
    });
  });

  hunks.forEach((hunk, idx) => {
    try {
      const trimmedHunk = trimHunk(hunk);
      const insertions = hunk.lines.filter((line) =>
        line.startsWith("+")
      ).length;
      const lineStart = trimmedHunk.newStart;
      const lineEnd = lineStart + insertions;
      const largestEnclosingFunction = parser.findEnclosingContext(
        updatedFile,
        lineStart,
        lineEnd
      ).enclosingContext;

      if (largestEnclosingFunction) {
        const enclosingRangeKey = `${largestEnclosingFunction.loc.start.line} -> ${largestEnclosingFunction.loc.end.line}`;
        let existingHunks = scopeRangeHunkMap.get(enclosingRangeKey) || [];
        existingHunks.push(hunk);
        scopeRangeHunkMap.set(enclosingRangeKey, existingHunks);
        scopeRangeNodeMap.set(enclosingRangeKey, largestEnclosingFunction);
      } else {
        throw "No enclosing function.";
      }
      order.push(idx);
    } catch (exc) {
      console.log(file.filename);
      console.log("NORMAL STRATEGY");
      console.log(exc);
      expandStrategy.push(hunk);
      order.push(idx);
    }
  });

  const scopeStategy: [string, diff.Hunk][] = [];
  for (const [range, hunks] of scopeRangeHunkMap.entries()) {
    const combinedHunk = combineHunks(updatedFile, hunks);
    scopeStategy.push([range, combinedHunk]);
  }

  const contexts: string[] = [];
  scopeStategy.forEach(([rangeKey, hunk]) => {
    const context = buildingScopeString(
      updatedFile,
      scopeRangeNodeMap.get(rangeKey),
      hunk
    ).join("\n");
    contexts.push(context);
  });
  expandStrategy.forEach((hunk) => {
    const context = expandHunk(file.old_contents, hunk);
    contexts.push(context);
  });
  return contexts;
};

/**
 * Formats a PR file's changes using function-level context awareness
 * Falls back to expanded patch strategy if parsing fails
 * @param file - The PR file to format
 * @param parser - Language-specific parser to use
 * @returns Formatted string containing filename and contextual diff
 */
const functionContextPatchStrategy = (
  file: PRFile,
  parser: AbstractParser
): string => {
  let res = null;
  try {
    const contextChunks = diffContextPerHunk(file, parser);
    res = `## ${file.filename}\n\n${contextChunks.join("\n\n")}`;
  } catch (exc) {
    console.log(exc);
    res = expandedPatchStrategy(file);
  }
  return res;
};

/**
 * Main strategy for formatting PR file changes with intelligent context
 * Uses language-specific parsing when available, falls back to simple expansion
 * @param file - The PR file to format
 * @returns Formatted string containing filename and contextual diff
 */
export const smarterContextPatchStrategy = (file: PRFile) => {
  const parser: AbstractParser = getParserForExtension(file.filename);
  if (parser != null) {
    return functionContextPatchStrategy(file, parser);
  } else {
    return expandedPatchStrategy(file);
  }
};
