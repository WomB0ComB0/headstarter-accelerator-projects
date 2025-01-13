/**
 *
 * @param {import("nitropack").Nitro} nitro
 * @returns
 */
export function createDevServer(nitro: import("nitropack").Nitro): Promise<{
    listen: (port: number, opts: Partial<import("@vinxi/listhen").ListenOptions>) => Promise<import("../node_modules/@vinxi/listhen/dist/shared/listhen.994822d6.js").a>;
    h3App: import("h3").App;
    localCall: (context: import("unenv/runtime/fetch/call").CallContext) => Promise<{
        body: BodyInit | null;
        headers: Record<string, string | number | string[] | undefined>;
        status: number;
        statusText: string;
    }>;
    localFetch: (input: string | Request, init: import("unenv/runtime/fetch/index").FetchOptions) => Promise<Response>;
    close: () => Promise<void>;
    hooks: import("hookable").Hookable<Record<string, any>, string>;
    captureError: (error: any, context?: {}) => void;
}>;
//# sourceMappingURL=nitro-dev.d.ts.map