/**
 * @template {string | undefined} T
 * @param {T} path
 * @param {string} root
 * @returns {T}
 */
export function relativeAppPath<T extends string | undefined>(path: T, root: string): T;
export namespace resolve {
    export { relativeAppPath as relative };
    export { absoluteAppPath as absolute };
}
/**
 * @template {string | undefined} T
 * @param {T} path
 * @param {string} root
 * @returns {T}
 */
declare function absoluteAppPath<T extends string | undefined>(path: T, root: string): T;
export {};
//# sourceMappingURL=resolve.d.ts.map