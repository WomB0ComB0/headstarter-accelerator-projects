/// <reference types="node" />
/**
 * Ensures that the link exists, and points to a valid file.
 * If the directory structure does not exist, it is created.
 *
 * @param target the source file path
 * @param linkName the destination link path
 */
export declare function ensureSymlink(target: string | URL, linkName: string | URL): Promise<void>;
/**
 * Ensures that the link exists, and points to a valid file.
 * If the directory structure does not exist, it is created.
 *
 * @param target the source file path
 * @param linkName the destination link path
 */
export declare function ensureSymlinkSync(target: string | URL, linkName: string | URL): void;
