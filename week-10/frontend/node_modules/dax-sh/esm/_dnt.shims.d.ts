/// <reference types="node" />
import { Deno } from "@deno/shim-deno";
export { Deno } from "@deno/shim-deno";
import { ReadableStream, WritableStream, TextDecoderStream, TransformStream } from "node:stream/web";
export { ReadableStream, WritableStream, TextDecoderStream, TransformStream, type ReadableStreamDefaultReader, type WritableStreamDefaultWriter, type StreamPipeOptions as PipeOptions, type QueuingStrategy } from "node:stream/web";
export { type BodyInit, type RequestCache, type RequestMode, type RequestRedirect, type ReferrerPolicy } from "undici-types";
export declare const dntGlobalThis: Omit<typeof globalThis, "Deno" | "ReadableStream" | "WritableStream" | "TextDecoderStream" | "TransformStream"> & {
    Deno: typeof Deno;
    ReadableStream: {
        new (underlyingSource: import("stream/web").UnderlyingByteSource, strategy?: import("stream/web").QueuingStrategy<Uint8Array> | undefined): ReadableStream<Uint8Array>;
        new <R = any>(underlyingSource?: import("stream/web").UnderlyingSource<R> | undefined, strategy?: import("stream/web").QueuingStrategy<R> | undefined): ReadableStream<R>;
        prototype: ReadableStream<any>;
    };
    WritableStream: {
        new <W = any>(underlyingSink?: import("stream/web").UnderlyingSink<W> | undefined, strategy?: import("stream/web").QueuingStrategy<W> | undefined): WritableStream<W>;
        prototype: WritableStream<any>;
    };
    TextDecoderStream: {
        new (encoding?: string | undefined, options?: import("stream/web").TextDecoderOptions | undefined): TextDecoderStream;
        prototype: TextDecoderStream;
    };
    TransformStream: {
        new <I = any, O = any>(transformer?: import("stream/web").Transformer<I, O> | undefined, writableStrategy?: import("stream/web").QueuingStrategy<I> | undefined, readableStrategy?: import("stream/web").QueuingStrategy<O> | undefined): TransformStream<I, O>;
        prototype: TransformStream<any, any>;
    };
};
