import type { Writer, WriterSync } from "./types.js";
/**
 * Write all the content of the array buffer (`arr`) to the writer (`w`).
 *
 * @example
 * ```ts
 * import { writeAll } from "https://deno.land/std@$STD_VERSION/io/write_all.ts";

 * // Example writing to stdout
 * let contentBytes = new TextEncoder().encode("Hello World");
 * await writeAll(Deno.stdout, contentBytes);
 *
 * // Example writing to file
 * contentBytes = new TextEncoder().encode("Hello World");
 * using file = await Deno.open('test.file', {write: true});
 * await writeAll(file, contentBytes);
 * ```
 */
export declare function writeAll(writer: Writer, data: Uint8Array): Promise<void>;
/**
 * Synchronously write all the content of the array buffer (`arr`) to the
 * writer (`w`).
 *
 * @example
 * ```ts
 * import { writeAllSync } from "https://deno.land/std@$STD_VERSION/io/write_all.ts";
 *
 * // Example writing to stdout
 * let contentBytes = new TextEncoder().encode("Hello World");
 * writeAllSync(Deno.stdout, contentBytes);
 *
 * // Example writing to file
 * contentBytes = new TextEncoder().encode("Hello World");
 * using file = Deno.openSync('test.file', {write: true});
 * writeAllSync(file, contentBytes);
 * ```
 */
export declare function writeAllSync(writer: WriterSync, data: Uint8Array): void;
