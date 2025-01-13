/// <reference types="node" />
import * as dntShim from "../../../../_dnt.shims.js";
import type { Writer } from "../io/types.js";
/**
 * Create a {@linkcode Writer} from a {@linkcode WritableStreamDefaultWriter}.
 *
 * @example
 * ```ts
 * import { copy } from "https://deno.land/std@$STD_VERSION/io/copy.ts";
 * import { writerFromStreamWriter } from "https://deno.land/std@$STD_VERSION/streams/writer_from_stream_writer.ts";
 *
 * using file = await Deno.open("./deno.land.html", { read: true });
 *
 * const writableStream = new WritableStream({
 *   write(chunk): void {
 *     console.log(chunk);
 *   },
 * });
 * const writer = writerFromStreamWriter(writableStream.getWriter());
 * await copy(file, writer);
 * ```
 *
 * @deprecated (will be removed after 1.0.0) Use {@linkcode WritableStreamDefaultWriter} directly.
 */
export declare function writerFromStreamWriter(streamWriter: dntShim.WritableStreamDefaultWriter<Uint8Array>): Writer;
