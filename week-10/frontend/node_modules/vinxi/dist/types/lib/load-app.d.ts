/**
 *
 * @param {string | undefined} configFile
 * @param {{ mode?: string }} args
 * @returns {Promise<import("./app.js").App | undefined>}
 */
export function loadApp(configFile?: string | undefined, args?: {
    mode?: string;
}): Promise<import("./app.js").App | undefined>;
//# sourceMappingURL=load-app.d.ts.map