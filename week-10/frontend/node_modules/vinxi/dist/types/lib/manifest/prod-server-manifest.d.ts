/**
 *
 * @param {ProdApp} app
 * @returns
 */
export function createProdManifest(app: ProdApp): {};
export type ProdApp = import("../app.js").App & {
    config: {
        buildManifest: {
            [key: string]: any;
        };
    };
};
//# sourceMappingURL=prod-server-manifest.d.ts.map