/** @typedef {{ force?: boolean; devtools?: boolean; port?: number; ws?: { port?: number }; https?: import('@vinxi/listhen').HTTPSOptions | boolean; }} DevConfigInput */
/** @typedef {{ force: boolean; port: number; devtools: boolean; ws: { port: number }; https?: import('@vinxi/listhen').Certificate; }} DevConfig */
/**
 *
 * @returns {import('./vite-dev.d.ts').Plugin}
 */
export function devEntries(): import('./vite-dev.d.ts').Plugin;
/**
 *
 * @param {import('vite').InlineConfig & { router: import("./router-mode.js").Router<any>; app: import("./app.js").App }} config
 * @returns
 */
export function createViteDevServer(config: import('vite').InlineConfig & {
    router: import("./router-mode.js").Router<any>;
    app: import("./app.js").App;
}): Promise<import("vite").ViteDevServer>;
/**
 *
 * @param {import('./app.js').App} app
 * @param {import('./router-mode.d.ts').Router<{ plugins?: any }>} router
 * @param {DevConfig} serveConfig
 * @returns {Promise<import("vite").ViteDevServer>}
 */
export function createViteHandler(router: import("./router-mode.js").Router<{
    plugins?: any;
}>, app: import('./app.js').App, serveConfig: DevConfig): Promise<import("vite").ViteDevServer>;
/**
 *
 * @param {import('./app.js').App} app
 * @param {DevConfigInput} param1
 * @returns
 */
export function createDevServer(app: import('./app.js').App, { force, port, devtools, ws: { port: wsPort }, }: DevConfigInput): Promise<{
    listen: () => Promise<import("../node_modules/@vinxi/listhen/dist/shared/listhen.994822d6.js").a>;
    close: () => Promise<void>;
    h3App: import("h3").App;
    localCall: (context: import("unenv/runtime/fetch/call").CallContext) => Promise<{
        body: BodyInit | null;
        headers: Record<string, string | number | string[] | undefined>;
        status: number;
        statusText: string;
    }>;
    localFetch: (input: string | Request, init: import("unenv/runtime/fetch/index").FetchOptions) => Promise<Response>;
    hooks: import("hookable").Hookable<Record<string, any>, string>;
    captureError: (error: any, context?: {}) => void;
}>;
export * from "./router-dev-plugins.js";
export type DevConfigInput = {
    force?: boolean;
    devtools?: boolean;
    port?: number;
    ws?: {
        port?: number;
    };
    https?: import('@vinxi/listhen').HTTPSOptions | boolean;
};
export type DevConfig = {
    force: boolean;
    port: number;
    devtools: boolean;
    ws: {
        port: number;
    };
    https?: import('@vinxi/listhen').Certificate;
};
//# sourceMappingURL=dev-server.d.ts.map