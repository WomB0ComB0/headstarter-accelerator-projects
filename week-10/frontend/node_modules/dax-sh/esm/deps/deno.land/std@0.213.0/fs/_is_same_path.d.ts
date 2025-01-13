/// <reference types="node" />
/**
 * Test whether `src` and `dest` resolve to the same location
 * @param src src file path
 * @param dest dest file path
 */
export declare function isSamePath(src: string | URL, dest: string | URL): boolean | void;
