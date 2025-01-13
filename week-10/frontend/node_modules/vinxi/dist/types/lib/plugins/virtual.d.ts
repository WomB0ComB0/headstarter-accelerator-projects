/**
 *
 * @param {{ [key: string] : (ctx: { config: import("../vite-dev.d.ts").ViteConfig }) => (string | Promise<string>) }} modules
 * @param {string} name
 * @param {any} cache
 * @returns {import('../vite-dev.d.ts').Plugin}
 */
export function virtual(modules: {
    [key: string]: (ctx: {
        config: import("../vite-dev.d.ts").ViteConfig;
    }) => (string | Promise<string>);
}, name?: string, cache?: any): import('../vite-dev.d.ts').Plugin;
//# sourceMappingURL=virtual.d.ts.map