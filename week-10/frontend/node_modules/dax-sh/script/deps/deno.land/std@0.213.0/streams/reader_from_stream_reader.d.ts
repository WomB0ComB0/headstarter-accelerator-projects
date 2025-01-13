/// <reference types="node" />
import * as dntShim from "../../../../_dnt.shims.js";
import type { Reader } from "../io/types.js";
/**
 * Create a {@linkcode Reader} from a {@linkcode ReadableStreamDefaultReader}.
 *
 * @example
 * ```ts
 * import { copy } from "https://deno.land/std@$STD_VERSION/io/copy.ts";
 * import { readerFromStreamReader } from "https://deno.land/std@$STD_VERSION/streams/reader_from_stream_reader.ts";
 *
 * const res = await fetch("https://deno.land");
 * using file = await Deno.open("./deno.land.html", { create: true, write: true });
 *
 * const reader = readerFromStreamReader(res.body!.getReader());
 * await copy(reader, file);
 * ```
 *
 * @deprecated (will be removed after 1.0.0) Use {@linkcode ReadableStreamDefaultReader} directly.
 */
export declare function readerFromStreamReader(streamReader: dntShim.ReadableStreamDefaultReader<Uint8Array>): Reader;
