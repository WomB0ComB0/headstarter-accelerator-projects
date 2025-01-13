/** @typedef {z.infer<typeof clientRouterSchema> & { outDir: string; base: string; order: number; root: string; internals: Internals }} ClientRouterSchema */
/** @typedef {z.infer<typeof customRouterSchema> & { outDir: string; base: string; order: number; root: string; internals: Internals }} CustomRouterSchema */
/** @typedef {z.infer<typeof staticRouterSchema> & { outDir: string; base: string; order: number; internals: Internals }} StaticRouterSchema */
/** @typedef {z.infer<typeof httpRouterSchema> & { outDir: string; base: string; order: number; root: string; internals: Internals }} HTTPRouterSchema */
/** @typedef {z.infer<typeof httpRouterSchema>} HandlerRouterInput */
/** @typedef {z.infer<typeof spaRouterSchema> & { outDir: string; base: string; order: number; root: string; internals: Internals }} SPARouterSchema */
/** @typedef {(HTTPRouterSchema | ClientRouterSchema | SPARouterSchema | StaticRouterSchema | CustomRouterSchema )} RouterSchema  */
/** @typedef {(z.infer<typeof clientRouterSchema> | z.infer<typeof staticRouterSchema> | z.infer<typeof spaRouterSchema> |  z.infer<typeof httpRouterSchema> | z.infer<typeof customRouterSchema>)} RouterSchemaInput  */
/**
 * @template X
 * @param {X} schema
 * @param {import("./router-mode.js").RouterMode<z.z.infer<X>>} mode
 * @returns {import("./router-mode.js").RouterMode<z.z.infer<X>>}
 */
export function createRouterMode<X>(schema: X, mode: import("./router-mode.js").RouterMode<z.TypeOf<X>>): import("./router-mode.js").RouterMode<z.TypeOf<X>>;
/**
 *
 * @param {RouterSchemaInput} router
 * @param {import("./app.js").AppOptions} appConfig
 * @param {number} order
 * @returns {RouterSchema}
 */
export function resolveRouterConfig(router: RouterSchemaInput, appConfig: import("./app.js").AppOptions, order: number): RouterSchema;
export { z };
/**
 * @typedef {{ routes?: CompiledRouter; devServer?: import('vite').ViteDevServer; appWorker?: import('./app-worker-client.js').AppWorkerClient; type: import("./router-mode.js").RouterMode }} Internals
 * @typedef {import('./fs-router.js').BaseFileSystemRouter} CompiledRouter
 * @typedef {(router: RouterSchemaInput, app: import("./app.js").AppOptions) => CompiledRouter} RouterStyleFn
 * */
export const staticRouterSchema: z.ZodObject<{
    name: z.ZodString;
    base: z.ZodOptional<z.ZodDefault<z.ZodString>>;
    type: z.ZodDefault<z.ZodLiteral<"static">>;
    dir: z.ZodString;
    root: z.ZodOptional<z.ZodString>;
}, "strip", z.ZodTypeAny, {
    type: "static";
    name: string;
    dir: string;
    base?: string | undefined;
    root?: string | undefined;
}, {
    name: string;
    dir: string;
    base?: string | undefined;
    type?: "static" | undefined;
    root?: string | undefined;
}>;
export const clientRouterSchema: z.ZodObject<{
    name: z.ZodString;
    base: z.ZodOptional<z.ZodDefault<z.ZodString>>;
    root: z.ZodOptional<z.ZodString>;
    type: z.ZodDefault<z.ZodLiteral<"client">>;
    handler: z.ZodString;
    /** @type {z.ZodOptionalType<z.ZodType<RouterStyleFn, z.ZodTypeDef, RouterStyleFn>>} */
    routes: z.ZodOptionalType<z.ZodType<RouterStyleFn, z.ZodTypeDef, RouterStyleFn>>;
    extensions: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
    outDir: z.ZodOptional<z.ZodString>;
    target: z.ZodOptional<z.ZodDefault<z.ZodEnum<["browser"]>>>;
    plugins: z.ZodOptional<z.ZodType<any, z.ZodTypeDef, any>>;
}, "strip", z.ZodTypeAny, {
    type: "client";
    name: string;
    handler: string;
    base?: string | undefined;
    root?: string | undefined;
    routes?: RouterStyleFn | undefined;
    extensions?: string[] | undefined;
    outDir?: string | undefined;
    target?: "browser" | undefined;
    plugins?: any;
}, {
    name: string;
    handler: string;
    base?: string | undefined;
    root?: string | undefined;
    type?: "client" | undefined;
    routes?: RouterStyleFn | undefined;
    extensions?: string[] | undefined;
    outDir?: string | undefined;
    target?: "browser" | undefined;
    plugins?: any;
}>;
export const httpRouterSchema: z.ZodObject<{
    name: z.ZodString;
    base: z.ZodOptional<z.ZodDefault<z.ZodString>>;
    root: z.ZodOptional<z.ZodString>;
    type: z.ZodDefault<z.ZodLiteral<"http">>;
    build: z.ZodOptional<z.ZodBoolean>;
    worker: z.ZodOptional<z.ZodBoolean>;
    handler: z.ZodString;
    middleware: z.ZodOptional<z.ZodString>;
    /** @type {z.ZodOptionalType<z.ZodType<RouterStyleFn, z.ZodTypeDef, RouterStyleFn>>} */
    routes: z.ZodOptionalType<z.ZodType<RouterStyleFn, z.ZodTypeDef, RouterStyleFn>>;
    outDir: z.ZodOptional<z.ZodString>;
    target: z.ZodOptional<z.ZodDefault<z.ZodEnum<["server"]>>>;
    plugins: z.ZodOptional<z.ZodType<any, z.ZodTypeDef, any>>;
}, "strip", z.ZodTypeAny, {
    type: "http";
    name: string;
    handler: string;
    base?: string | undefined;
    root?: string | undefined;
    build?: boolean | undefined;
    worker?: boolean | undefined;
    middleware?: string | undefined;
    routes?: RouterStyleFn | undefined;
    outDir?: string | undefined;
    target?: "server" | undefined;
    plugins?: any;
}, {
    name: string;
    handler: string;
    base?: string | undefined;
    root?: string | undefined;
    type?: "http" | undefined;
    build?: boolean | undefined;
    worker?: boolean | undefined;
    middleware?: string | undefined;
    routes?: RouterStyleFn | undefined;
    outDir?: string | undefined;
    target?: "server" | undefined;
    plugins?: any;
}>;
export const spaRouterSchema: z.ZodObject<{
    name: z.ZodString;
    base: z.ZodOptional<z.ZodDefault<z.ZodString>>;
    root: z.ZodOptional<z.ZodString>;
    type: z.ZodDefault<z.ZodLiteral<"spa">>;
    /** @type {z.ZodOptionalType<z.ZodType<RouterStyleFn, z.ZodTypeDef, RouterStyleFn>>} */
    routes: z.ZodOptionalType<z.ZodType<RouterStyleFn, z.ZodTypeDef, RouterStyleFn>>;
    handler: z.ZodString;
    outDir: z.ZodOptional<z.ZodString>;
    target: z.ZodOptional<z.ZodDefault<z.ZodEnum<["browser"]>>>;
    plugins: z.ZodOptional<z.ZodType<any, z.ZodTypeDef, any>>;
}, "strip", z.ZodTypeAny, {
    type: "spa";
    name: string;
    handler: string;
    base?: string | undefined;
    root?: string | undefined;
    routes?: RouterStyleFn | undefined;
    outDir?: string | undefined;
    target?: "browser" | undefined;
    plugins?: any;
}, {
    name: string;
    handler: string;
    base?: string | undefined;
    root?: string | undefined;
    type?: "spa" | undefined;
    routes?: RouterStyleFn | undefined;
    outDir?: string | undefined;
    target?: "browser" | undefined;
    plugins?: any;
}>;
export namespace routerSchema {
    export { staticRouterSchema as static };
    export { clientRouterSchema as client };
    export { spaRouterSchema as spa };
    export { httpRouterSchema as http };
    export { customRouterSchema as custom };
}
export type ClientRouterSchema = z.infer<typeof clientRouterSchema> & {
    outDir: string;
    base: string;
    order: number;
    root: string;
    internals: Internals;
};
export type CustomRouterSchema = z.infer<typeof customRouterSchema> & {
    outDir: string;
    base: string;
    order: number;
    root: string;
    internals: Internals;
};
export type StaticRouterSchema = z.infer<typeof staticRouterSchema> & {
    outDir: string;
    base: string;
    order: number;
    internals: Internals;
};
export type HTTPRouterSchema = z.infer<typeof httpRouterSchema> & {
    outDir: string;
    base: string;
    order: number;
    root: string;
    internals: Internals;
};
export type HandlerRouterInput = z.infer<typeof httpRouterSchema>;
export type SPARouterSchema = z.infer<typeof spaRouterSchema> & {
    outDir: string;
    base: string;
    order: number;
    root: string;
    internals: Internals;
};
export type RouterSchema = (HTTPRouterSchema | ClientRouterSchema | SPARouterSchema | StaticRouterSchema | CustomRouterSchema);
export type RouterSchemaInput = (z.infer<typeof clientRouterSchema> | z.infer<typeof staticRouterSchema> | z.infer<typeof spaRouterSchema> | z.infer<typeof httpRouterSchema> | z.infer<typeof customRouterSchema>);
export type Internals = {
    routes?: CompiledRouter;
    devServer?: import('vite').ViteDevServer;
    appWorker?: import('./app-worker-client.js').AppWorkerClient;
    type: import("./router-mode.js").RouterMode;
};
export type CompiledRouter = import('./fs-router.js').BaseFileSystemRouter;
export type RouterStyleFn = (router: RouterSchemaInput, app: import("./app.js").AppOptions) => CompiledRouter;
import * as z from "zod";
declare const customRouterSchema: z.ZodObject<{
    name: z.ZodString;
    base: z.ZodOptional<z.ZodDefault<z.ZodString>>;
    root: z.ZodOptional<z.ZodString>;
    type: z.ZodObject<{
        resolveConfig: z.ZodFunction<z.ZodTuple<[z.ZodAny, z.ZodAny], z.ZodUnknown>, z.ZodAny>;
    }, "strip", z.ZodTypeAny, {
        resolveConfig: (args_0: any, args_1: any, ...args_2: unknown[]) => any;
    }, {
        resolveConfig: (args_0: any, args_1: any, ...args_2: unknown[]) => any;
    }>;
    /** @type {z.ZodOptionalType<z.ZodType<RouterStyleFn, z.ZodTypeDef, RouterStyleFn>>} */
    routes: z.ZodOptionalType<z.ZodType<RouterStyleFn, z.ZodTypeDef, RouterStyleFn>>;
    handler: z.ZodString;
    outDir: z.ZodOptional<z.ZodString>;
    target: z.ZodLiteral<"server">;
    plugins: z.ZodOptional<z.ZodType<any, z.ZodTypeDef, any>>;
}, "strip", z.ZodTypeAny, {
    type: {
        resolveConfig: (args_0: any, args_1: any, ...args_2: unknown[]) => any;
    };
    name: string;
    target: "server";
    handler: string;
    base?: string | undefined;
    root?: string | undefined;
    routes?: RouterStyleFn | undefined;
    outDir?: string | undefined;
    plugins?: any;
}, {
    type: {
        resolveConfig: (args_0: any, args_1: any, ...args_2: unknown[]) => any;
    };
    name: string;
    target: "server";
    handler: string;
    base?: string | undefined;
    root?: string | undefined;
    routes?: RouterStyleFn | undefined;
    outDir?: string | undefined;
    plugins?: any;
}>;
//# sourceMappingURL=router-modes.d.ts.map