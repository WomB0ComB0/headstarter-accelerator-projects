/** @typedef {{ dir: string; extensions: string[] }} FileSystemRouterConfig */
/** @typedef {{ path: string } & any} Route */
/**
 *
 * @param {string} src
 * @param {FileSystemRouterConfig} config
 * @returns
 */
export function cleanPath(src: string, config: FileSystemRouterConfig): string;
/**
 *
 * @param {string} src
 * @returns
 */
export function analyzeModule(src: string): readonly [imports: readonly import("es-module-lexer").ImportSpecifier[], exports: readonly import("es-module-lexer").ExportSpecifier[], facade: boolean];
export { pathToRegexp };
export function glob(path: string): string[];
export class BaseFileSystemRouter extends EventTarget {
    /**
     *
     * @param {FileSystemRouterConfig} config
     * @param {import("./router-mode").Router<any>} router
     * @param {import("./app").AppOptions} app
     */
    constructor(config: FileSystemRouterConfig, router: import("./router-mode").Router<any>, app: import("./app").AppOptions);
    /** @type {any[]} */
    routes: any[];
    /** @type {import("./router-mode").Router<any>} */
    routerConfig: import("./router-mode").Router<any>;
    /** @type {import("./app").AppOptions} */
    appConfig: import("./app").AppOptions;
    /** @type {FileSystemRouterConfig} */
    config: FileSystemRouterConfig;
    glob(): string;
    /**
     * @returns {Promise<any[]>}
     */
    buildRoutes(): Promise<any[]>;
    /**
     *
     * @param {*} src
     * @returns {boolean}
     */
    isRoute(src: any): boolean;
    /**
     *
     * @param {*} src
     * @returns {string}
     */
    toPath(src: any): string;
    /**
     *
     * @param {*} src
     * @returns {Route | null}
     */
    toRoute(src: any): Route | null;
    /**
     * To be attached by vite plugin to the vite dev server
     */
    update: undefined;
    /**
     *
     * @param {Route} route
     */
    _addRoute(route: Route): void;
    /**
     *
     * @param {string} src
     */
    addRoute(src: string): Promise<void>;
    /**
     *
     * @param {string} route
     */
    reload(route: string): void;
    /**
     *
     * @param {string} src
     */
    updateRoute(src: string): Promise<void>;
    /**
     *
     * @param {string} src
     * @returns
     */
    removeRoute(src: string): void;
    /** @type {Promise<any[]> | undefined} */
    buildRoutesPromise: Promise<any[]> | undefined;
    getRoutes(): Promise<any[]>;
}
export type FileSystemRouterConfig = {
    dir: string;
    extensions: string[];
};
export type Route = {
    path: string;
} & any;
import { pathToRegexp } from "path-to-regexp";
//# sourceMappingURL=fs-router.d.ts.map