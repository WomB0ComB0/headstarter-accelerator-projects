/**
 *
 * @param {string} tag
 * @param {import('../vite-dev.d.ts').CustomizableConfig | ((router: import('../router-mode.d.ts').Router, app: import('../app-types.d.ts').App, env:import('../vite-dev.d.ts').ConfigEnv) => import('../vite-dev.d.ts').CustomizableConfig) } conf
 * @returns {import('../vite-dev.d.ts').Plugin}
 */
export function config(tag: string, conf: import("../vite-dev.d.ts").CustomizableConfig | ((router: import('../router-mode.d.ts').Router, app: import('../app-types.d.ts').App, env: import('../vite-dev.d.ts').ConfigEnv) => import('../vite-dev.d.ts').CustomizableConfig)): import('../vite-dev.d.ts').Plugin;
/**
 *
 * @param {string} path
 * @param {string} file
 * @returns {import('../vite-dev.d.ts').Plugin}
 */
export function input(path: string, file: string): import('../vite-dev.d.ts').Plugin;
//# sourceMappingURL=config.d.ts.map