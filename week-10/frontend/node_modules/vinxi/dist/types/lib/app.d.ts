/** @typedef {{
    devtools?: boolean;
    routers?: import("./router-modes.js").RouterSchemaInput[];
    name?:
    string;
    server?: Omit<import('nitropack').NitroConfig, 'handlers' | 'scanDirs' | 'appConfigFiles' | 'imports' | 'virtual' | 'dev'  | 'buildDir'> & { https?: import('@vinxi/listhen').HTTPSOptions | boolean };
    root?: string
    mode?: string
}} AppOptions */
/** @typedef {{
    config: {
        name: string;
        devtools: boolean;
        server: Omit<import('nitropack').NitroConfig, 'handlers' | 'scanDirs' | 'appConfigFiles' | 'imports' | 'virtual' | 'dev' | 'buildDir'> & { https?: import('@vinxi/listhen').HTTPSOptions | boolean };
        routers: import("./router-mode.js").Router[];
        root: string;
        mode?: string;
    };
    addRouter: (router: any) => App;
    addRouterPlugins: (apply: (router: import("./router-mode.js").Router) => boolean, plugins: () => any[]) => void;
    getRouter: (name: string) => import("./router-mode.js").Router;
    resolveSync: (mod: string) => string;
    import: (mod: string) => Promise<any>;
    stack: (stack: (app: App) => void | Promise<void>) => Promise<App>;
    dev(): Promise<void>;
    build(): Promise<void>;
    hooks: import("hookable").Hookable;
}} App */
/**
 *
 * @param {AppOptions} param0
 * @returns {App}
 */
export function createApp({ routers, name, server, root, mode, }?: AppOptions): App;
export type AppOptions = {
    devtools?: boolean;
    routers?: import("./router-modes.js").RouterSchemaInput[];
    name?: string;
    server?: Omit<import('nitropack').NitroConfig, 'handlers' | 'scanDirs' | 'appConfigFiles' | 'imports' | 'virtual' | 'dev' | 'buildDir'> & {
        https?: import('@vinxi/listhen').HTTPSOptions | boolean;
    };
    root?: string;
    mode?: string;
};
export type App = {
    config: {
        name: string;
        devtools: boolean;
        server: Omit<import('nitropack').NitroConfig, 'handlers' | 'scanDirs' | 'appConfigFiles' | 'imports' | 'virtual' | 'dev' | 'buildDir'> & {
            https?: import('@vinxi/listhen').HTTPSOptions | boolean;
        };
        routers: import("./router-mode.js").Router[];
        root: string;
        mode?: string;
    };
    addRouter: (router: any) => App;
    addRouterPlugins: (apply: (router: import("./router-mode.js").Router) => boolean, plugins: () => any[]) => void;
    getRouter: (name: string) => import("./router-mode.js").Router;
    resolveSync: (mod: string) => string;
    import: (mod: string) => Promise<any>;
    stack: (stack: (app: App) => void | Promise<void>) => Promise<App>;
    dev(): Promise<void>;
    build(): Promise<void>;
    hooks: import("hookable").Hookable;
};
//# sourceMappingURL=app.d.ts.map