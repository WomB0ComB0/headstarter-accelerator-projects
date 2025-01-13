import type { Reader, ReaderSync } from "./types.js";
/**
 * Read {@linkcode Reader} `r` until EOF (`null`) and resolve to the content as
 * {@linkcode Uint8Array}.
 *
 * @example
 * ```ts
 * import { readAll } from "https://deno.land/std@$STD_VERSION/io/read_all.ts";
 *
 * // Example from stdin
 * const stdinContent = await readAll(Deno.stdin);
 *
 * // Example from file
 * using file = await Deno.open("my_file.txt", {read: true});
 * const myFileContent = await readAll(file);
 * ```
 */
export declare function readAll(reader: Reader): Promise<Uint8Array>;
/**
 * Synchronously reads {@linkcode ReaderSync} `r` until EOF (`null`) and returns
 * the content as {@linkcode Uint8Array}.
 *
 * @example
 * ```ts
 * import { readAllSync } from "https://deno.land/std@$STD_VERSION/io/read_all.ts";
 *
 * // Example from stdin
 * const stdinContent = readAllSync(Deno.stdin);
 *
 * // Example from file
 * using file = Deno.openSync("my_file.txt", {read: true});
 * const myFileContent = readAllSync(file);
 * ```
 */
export declare function readAllSync(reader: ReaderSync): Uint8Array;
