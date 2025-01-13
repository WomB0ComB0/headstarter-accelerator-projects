/// <reference types="node" />
/// <reference types="node" />
import * as dntShim from "../_dnt.shims.js";
/**
 * `ExpandGlobOptions` from https://deno.land/std/fs/expand_glob.ts
 * @internal
 */
type DenoStdExpandGlobOptions = import("./deps.js").ExpandGlobOptions;
export type ExpandGlobOptions = DenoStdExpandGlobOptions;
/**
 * `WalkOptions` from https://deno.land/std/fs/walk.ts
 * @internal
 */
type DenoStdWalkOptions = import("./deps.js").WalkOptions;
export type WalkOptions = DenoStdWalkOptions;
/** @internal */
export declare function createPath(path: string | URL | ImportMeta | Path): Path;
export interface WalkEntry extends dntShim.Deno.DirEntry {
    path: Path;
}
export interface PathSymlinkOptions {
    /** Creates the symlink as absolute or relative. */
    kind: "absolute" | "relative";
}
export interface SymlinkOptions extends Partial<dntShim.Deno.SymlinkOptions>, Partial<PathSymlinkOptions> {
}
/**
 * Holds a reference to a path providing helper methods.
 *
 * Create one via `$`: `const srcDir = $.path("src");`
 */
export declare class Path {
    #private;
    /** This is a special symbol that allows different versions of
     * Dax's `Path` API to match on `instanceof` checks. Ideally
     * people shouldn't be mixing versions, but if it happens then
     * this will maybe reduce some bugs (or cause some... tbd).
     * @internal
     */
    private static instanceofSymbol;
    constructor(path: string | URL | ImportMeta | Path);
    /** @internal */
    static [Symbol.hasInstance](instance: any): boolean;
    /** Gets the string representation of this path. */
    toString(): string;
    /** Resolves the path and gets the file URL. */
    toFileUrl(): URL;
    /** If this path reference is the same as another one. */
    equals(otherPath: Path): boolean;
    /** Joins the provided path segments onto this path. */
    join(...pathSegments: string[]): Path;
    /** Resolves this path to an absolute path along with the provided path segments. */
    resolve(...pathSegments: string[]): Path;
    /**
     * Normalizes the `path`, resolving `'..'` and `'.'` segments.
     * Note that resolving these segments does not necessarily mean that all will be eliminated.
     * A `'..'` at the top-level will be preserved, and an empty path is canonically `'.'`.
     */
    normalize(): Path;
    /** Follows symlinks and gets if this path is a directory. */
    isDirSync(): boolean;
    /** Follows symlinks and gets if this path is a file. */
    isFileSync(): boolean;
    /** Gets if this path is a symlink. */
    isSymlinkSync(): boolean;
    /** Gets if this path is an absolute path. */
    isAbsolute(): boolean;
    /** Gets if this path is relative. */
    isRelative(): boolean;
    /** Resolves the `Deno.FileInfo` of this path following symlinks. */
    stat(): Promise<dntShim.Deno.FileInfo | undefined>;
    /** Synchronously resolves the `Deno.FileInfo` of this
     * path following symlinks. */
    statSync(): dntShim.Deno.FileInfo | undefined;
    /** Resolves the `Deno.FileInfo` of this path without
     * following symlinks. */
    lstat(): Promise<dntShim.Deno.FileInfo | undefined>;
    /** Synchronously resolves the `Deno.FileInfo` of this path
     * without following symlinks. */
    lstatSync(): dntShim.Deno.FileInfo | undefined;
    /**
     * Gets the directory path. In most cases, it is recommended
     * to use `.parent()` instead since it will give you a `Path`.
     */
    dirname(): string;
    /** Gets the file or directory name of the path. */
    basename(): string;
    /** Resolves the path getting all its ancestor directories in order. */
    ancestors(): Generator<Path>;
    components(): Generator<string>;
    startsWith(path: Path | URL | string): boolean;
    endsWith(path: Path | URL | string): boolean;
    /** Gets the parent directory or returns undefined if the parent is the root directory. */
    parent(): Path | undefined;
    /** Gets the parent or throws if the current directory was the root. */
    parentOrThrow(): Path;
    /**
     * Returns the extension of the path with leading period or undefined
     * if there is no extension.
     */
    extname(): string | undefined;
    /** Gets a new path reference with the provided extension. */
    withExtname(ext: string): Path;
    /** Gets a new path reference with the provided file or directory name. */
    withBasename(basename: string): Path;
    /** Gets the relative path from this path to the specified path. */
    relative(to: string | URL | Path): string;
    /** Gets if the path exists. Beware of TOCTOU issues. */
    exists(): Promise<boolean>;
    /** Synchronously gets if the path exists. Beware of TOCTOU issues. */
    existsSync(): boolean;
    /** Resolves to the absolute normalized path, with symbolic links resolved. */
    realPath(): Promise<Path>;
    /** Synchronously resolves to the absolute normalized path, with symbolic links resolved. */
    realPathSync(): Path;
    /** Expands the glob using the current path as the root. */
    expandGlob(glob: string | URL, options?: Omit<ExpandGlobOptions, "root">): AsyncGenerator<WalkEntry, void, unknown>;
    /** Synchronously expands the glob using the current path as the root. */
    expandGlobSync(glob: string | URL, options?: Omit<ExpandGlobOptions, "root">): Generator<WalkEntry, void, unknown>;
    /** Walks the file tree rooted at the current path, yielding each file or
     * directory in the tree filtered according to the given options. */
    walk(options?: WalkOptions): AsyncIterableIterator<WalkEntry>;
    /** Synchronously walks the file tree rooted at the current path, yielding each
     * file or directory in the tree filtered according to the given options. */
    walkSync(options?: WalkOptions): Iterable<WalkEntry>;
    /** Creates a directory at this path.
     * @remarks By default, this is recursive.
     */
    mkdir(options?: dntShim.Deno.MkdirOptions): Promise<this>;
    /** Synchronously creates a directory at this path.
     * @remarks By default, this is recursive.
     */
    mkdirSync(options?: dntShim.Deno.MkdirOptions): this;
    /**
     * Creates a symlink to the provided target path.
     */
    createSymlinkTo(targetPath: URL | Path, opts: Partial<dntShim.Deno.SymlinkOptions> & PathSymlinkOptions): Promise<void>;
    /**
     * Creates a symlink at the provided path with the provided target text.
     */
    createSymlinkTo(target: string, opts?: SymlinkOptions): Promise<void>;
    /**
     * Synchronously creates a symlink to the provided target path.
     */
    createSymlinkToSync(targetPath: URL | Path, opts: Partial<dntShim.Deno.SymlinkOptions> & PathSymlinkOptions): void;
    /**
     * Synchronously creates a symlink at the provided path with the provided target text.
     */
    createSymlinkToSync(target: string, opts?: SymlinkOptions): void;
    /** Reads the entries in the directory. */
    readDir(): AsyncIterable<WalkEntry>;
    /** Synchronously reads the entries in the directory. */
    readDirSync(): Iterable<WalkEntry>;
    /** Reads only the directory file paths, not including symlinks. */
    readDirFilePaths(): AsyncIterable<Path>;
    /** Synchronously reads only the directory file paths, not including symlinks. */
    readDirFilePathsSync(): Iterable<Path>;
    /** Reads the bytes from the file. */
    readBytes(options?: dntShim.Deno.ReadFileOptions): Promise<Uint8Array>;
    /** Synchronously reads the bytes from the file. */
    readBytesSync(): Uint8Array;
    /** Calls `.readBytes()`, but returns undefined if the path doesn't exist. */
    readMaybeBytes(options?: dntShim.Deno.ReadFileOptions): Promise<Uint8Array | undefined>;
    /** Calls `.readBytesSync()`, but returns undefined if the path doesn't exist. */
    readMaybeBytesSync(): Uint8Array | undefined;
    /** Reads the text from the file. */
    readText(options?: dntShim.Deno.ReadFileOptions): Promise<string>;
    /** Synchronously reads the text from the file. */
    readTextSync(): string;
    /** Calls `.readText()`, but returns undefined when the path doesn't exist.
     * @remarks This still errors for other kinds of errors reading a file.
     */
    readMaybeText(options?: dntShim.Deno.ReadFileOptions): Promise<string | undefined>;
    /** Calls `.readTextSync()`, but returns undefined when the path doesn't exist.
     * @remarks This still errors for other kinds of errors reading a file.
     */
    readMaybeTextSync(): string | undefined;
    /** Reads and parses the file as JSON, throwing if it doesn't exist or is not valid JSON. */
    readJson<T>(options?: dntShim.Deno.ReadFileOptions): Promise<T>;
    /** Synchronously reads and parses the file as JSON, throwing if it doesn't
     * exist or is not valid JSON. */
    readJsonSync<T>(): T;
    /**
     * Calls `.readJson()`, but returns undefined if the file doesn't exist.
     * @remarks This method will still throw if the file cannot be parsed as JSON.
     */
    readMaybeJson<T>(options?: dntShim.Deno.ReadFileOptions): Promise<T | undefined>;
    /**
     * Calls `.readJsonSync()`, but returns undefined if the file doesn't exist.
     * @remarks This method will still throw if the file cannot be parsed as JSON.
     */
    readMaybeJsonSync<T>(): T | undefined;
    /** Writes out the provided bytes to the file. */
    write(data: Uint8Array, options?: dntShim.Deno.WriteFileOptions): Promise<this>;
    /** Synchronously writes out the provided bytes to the file. */
    writeSync(data: Uint8Array, options?: dntShim.Deno.WriteFileOptions): this;
    /** Writes out the provided text to the file. */
    writeText(text: string, options?: dntShim.Deno.WriteFileOptions): Promise<this>;
    /** Synchronously writes out the provided text to the file. */
    writeTextSync(text: string, options?: dntShim.Deno.WriteFileOptions): this;
    /** Writes out the provided object as compact JSON. */
    writeJson(obj: unknown, options?: dntShim.Deno.WriteFileOptions): Promise<this>;
    /** Synchronously writes out the provided object as compact JSON. */
    writeJsonSync(obj: unknown, options?: dntShim.Deno.WriteFileOptions): this;
    /** Writes out the provided object as formatted JSON. */
    writeJsonPretty(obj: unknown, options?: dntShim.Deno.WriteFileOptions): Promise<this>;
    /** Synchronously writes out the provided object as formatted JSON. */
    writeJsonPrettySync(obj: unknown, options?: dntShim.Deno.WriteFileOptions): this;
    /** Appends the provided bytes to the file. */
    append(data: Uint8Array, options?: Omit<dntShim.Deno.WriteFileOptions, "append">): Promise<this>;
    /** Synchronously appends the provided bytes to the file. */
    appendSync(data: Uint8Array, options?: Omit<dntShim.Deno.WriteFileOptions, "append">): this;
    /** Appends the provided text to the file. */
    appendText(text: string, options?: Omit<dntShim.Deno.WriteFileOptions, "append">): Promise<this>;
    /** Synchronously appends the provided text to the file. */
    appendTextSync(text: string, options?: Omit<dntShim.Deno.WriteFileOptions, "append">): this;
    /** Changes the permissions of the file or directory. */
    chmod(mode: number): Promise<this>;
    /** Synchronously changes the permissions of the file or directory. */
    chmodSync(mode: number): this;
    /** Changes the ownership permissions of the file. */
    chown(uid: number | null, gid: number | null): Promise<this>;
    /** Synchronously changes the ownership permissions of the file. */
    chownSync(uid: number | null, gid: number | null): this;
    /** Creates a new file or opens the existing one. */
    create(): Promise<FsFileWrapper>;
    /** Synchronously creates a new file or opens the existing one. */
    createSync(): FsFileWrapper;
    /** Creates a file throwing if a file previously existed. */
    createNew(): Promise<FsFileWrapper>;
    /** Synchronously creates a file throwing if a file previously existed. */
    createNewSync(): FsFileWrapper;
    /** Opens a file. */
    open(options?: dntShim.Deno.OpenOptions): Promise<FsFileWrapper>;
    /** Opens a file synchronously. */
    openSync(options?: dntShim.Deno.OpenOptions): FsFileWrapper;
    /** Removes the file or directory from the file system. */
    remove(options?: dntShim.Deno.RemoveOptions): Promise<this>;
    /** Removes the file or directory from the file system synchronously. */
    removeSync(options?: dntShim.Deno.RemoveOptions): this;
    /**
     * Ensures that a directory is empty.
     * Deletes directory contents if the directory is not empty.
     * If the directory does not exist, it is created.
     * The directory itself is not deleted.
     */
    emptyDir(): Promise<this>;
    /** Synchronous version of `emptyDir()` */
    emptyDirSync(): this;
    /** Ensures that the directory exists.
     * If the directory structure does not exist, it is created. Like mkdir -p.
     */
    ensureDir(): Promise<this>;
    /** Synchronously ensures that the directory exists.
     * If the directory structure does not exist, it is created. Like mkdir -p.
     */
    ensureDirSync(): this;
    /**
     * Ensures that the file exists.
     * If the file that is requested to be created is in directories that do
     * not exist these directories are created. If the file already exists,
     * it is NOTMODIFIED.
     */
    ensureFile(): Promise<this>;
    /**
     * Synchronously ensures that the file exists.
     * If the file that is requested to be created is in directories that do
     * not exist these directories are created. If the file already exists,
     * it is NOTMODIFIED.
     */
    ensureFileSync(): this;
    /**
     * Copies the file to the specified destination path.
     * @returns The destination file path.
     */
    copyFile(destinationPath: string | URL | Path): Promise<Path>;
    /**
     * Copies the file to the destination path synchronously.
     * @returns The destination file path.
     */
    copyFileSync(destinationPath: string | URL | Path): Path;
    /**
     * Copies the file to the specified directory.
     * @returns The destination file path.
     */
    copyFileToDir(destinationDirPath: string | URL | Path): Promise<Path>;
    /**
     * Copies the file to the specified directory synchronously.
     * @returns The destination file path.
     */
    copyFileToDirSync(destinationDirPath: string | URL | Path): Path;
    /**
     * Moves the file or directory returning a promise that resolves to
     * the renamed path.
     */
    rename(newPath: string | URL | Path): Promise<Path>;
    /**
     * Moves the file or directory returning the renamed path synchronously.
     */
    renameSync(newPath: string | URL | Path): Path;
    /**
     * Moves the file or directory to the specified directory.
     * @returns The destination file path.
     */
    renameToDir(destinationDirPath: string | URL | Path): Promise<Path>;
    /**
     * Moves the file or directory to the specified directory synchronously.
     * @returns The destination file path.
     */
    renameToDirSync(destinationDirPath: string | URL | Path): Path;
    /** Opens the file and pipes it to the writable stream. */
    pipeTo(dest: dntShim.WritableStream<Uint8Array>, options?: dntShim.PipeOptions): Promise<this>;
}
export declare class FsFileWrapper extends dntShim.Deno.FsFile {
    /** Writes the provided text to this file. */
    writeText(text: string): Promise<this>;
    /** Synchronously writes the provided text to this file. */
    writeTextSync(text: string): this;
    /** Writes the provided bytes to the file. */
    writeBytes(bytes: Uint8Array): Promise<this>;
    /** Synchronously writes the provided bytes to the file. */
    writeBytesSync(bytes: Uint8Array): this;
}
export {};
