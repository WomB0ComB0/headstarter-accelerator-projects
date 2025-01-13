/// <reference types="node" />
/// <reference types="node" />
import * as dntShim from "../_dnt.shims.js";
import { type FsFileWrapper, Path } from "./path.js";
import { Buffer } from "./deps.js";
import type { RequestBuilder } from "./request.js";
import type { CommandBuilder, KillSignal } from "./command.js";
/** `Deno.Reader` stream. */
export interface Reader {
    read(p: Uint8Array): Promise<number | null>;
}
/** `Deno.ReaderSync` stream. */
export interface ReaderSync {
    readSync(p: Uint8Array): number | null;
}
/** `Deno.WriterSync` stream. */
export interface WriterSync {
    writeSync(p: Uint8Array): number;
}
/** `Deno.Writer` stream. */
export interface Writer {
    write(p: Uint8Array): Promise<number>;
}
export type PipeReader = Reader | ReaderSync;
export type PipeWriter = Writer | WriterSync;
/** `Deno.Closer` */
export interface Closer {
    close(): void;
}
/** Behaviour to use for stdin.
 * @value "inherit" - Sends the stdin of the process to the shell (default).
 * @value "null" - Does not pipe or redirect the pipe.
 */
export type ShellPipeReaderKind = "inherit" | "null" | Reader | dntShim.ReadableStream<Uint8Array> | Uint8Array | CommandBuilder | FsFileWrapper | Path | RequestBuilder;
/**
 * The behaviour to use for a shell pipe.
 * @value "inherit" - Sends the output directly to the current process' corresponding pipe (default).
 * @value "null" - Does not pipe or redirect the pipe.
 * @value "piped" - Captures the pipe without outputting.
 * @value "inheritPiped" - Captures the pipe with outputting.
 */
export type ShellPipeWriterKind = "inherit" | "null" | "piped" | "inheritPiped" | WriterSync | dntShim.WritableStream<Uint8Array> | FsFileWrapper | Path;
export declare class NullPipeReader implements Reader {
    read(_p: Uint8Array): Promise<number | null>;
}
export declare class NullPipeWriter implements WriterSync {
    writeSync(p: Uint8Array): number;
}
export declare class ShellPipeWriter {
    #private;
    constructor(kind: ShellPipeWriterKind, inner: PipeWriter);
    get kind(): ShellPipeWriterKind;
    get inner(): PipeWriter;
    write(p: Uint8Array): number | Promise<number>;
    writeAll(data: Uint8Array): void | Promise<void>;
    writeText(text: string): void | Promise<void>;
    writeLine(text: string): void | Promise<void>;
}
export declare class CapturingBufferWriter implements Writer {
    #private;
    constructor(innerWriter: Writer, buffer: Buffer);
    getBuffer(): Buffer;
    write(p: Uint8Array): Promise<number>;
}
export declare class CapturingBufferWriterSync implements WriterSync {
    #private;
    constructor(innerWriter: WriterSync, buffer: Buffer);
    getBuffer(): Buffer;
    writeSync(p: Uint8Array): number;
}
export declare class InheritStaticTextBypassWriter implements WriterSync {
    #private;
    constructor(innerWriter: WriterSync);
    writeSync(p: Uint8Array): number;
    flush(): void;
}
export interface PipedBufferListener extends WriterSync, Closer {
    setError(err: Error): void;
}
export declare class PipedBuffer implements WriterSync {
    #private;
    constructor();
    getBuffer(): Buffer | undefined;
    setError(err: Error): void;
    close(): void;
    writeSync(p: Uint8Array): number;
    setListener(listener: PipedBufferListener): void;
}
export declare class PipeSequencePipe implements Reader, WriterSync {
    #private;
    close(): void;
    writeSync(p: Uint8Array): number;
    read(p: Uint8Array): Promise<number | null>;
}
export declare function pipeReaderToWritable(reader: Reader, writable: dntShim.WritableStream<Uint8Array>, signal: AbortSignal): Promise<void>;
export declare function pipeReadableToWriterSync(readable: dntShim.ReadableStream<Uint8Array>, writer: ShellPipeWriter, signal: AbortSignal | KillSignal): Promise<void>;
