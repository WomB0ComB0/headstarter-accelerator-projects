import * as dntShim from "../../../../_dnt.shims.js";
export interface Environment {
    /** Gets an environment variable. */
    env(key: string): string | undefined;
    /** Resolves the `Deno.FileInfo` for the specified
     * path following symlinks.
     */
    stat(filePath: string): Promise<Pick<dntShim.Deno.FileInfo, "isFile">>;
    /** Synchronously resolves the `Deno.FileInfo` for
     * the specified path following symlinks.
     */
    statSync(filePath: string): Pick<dntShim.Deno.FileInfo, "isFile">;
    /** Gets the current operating system. */
    os: typeof dntShim.Deno.build.os;
}
export declare class RealEnvironment implements Environment {
    env(key: string): string | undefined;
    stat(path: string): Promise<dntShim.Deno.FileInfo>;
    statSync(path: string): dntShim.Deno.FileInfo;
    get os(): "aix" | "android" | "darwin" | "freebsd" | "linux" | "netbsd" | "windows" | "solaris" | "illumos";
}
/** Finds the path to the specified command asynchronously. */
export declare function which(command: string, environment?: Omit<Environment, "statSync">): Promise<string | undefined>;
/** Finds the path to the specified command synchronously. */
export declare function whichSync(command: string, environment?: Omit<Environment, "stat">): string | undefined;
