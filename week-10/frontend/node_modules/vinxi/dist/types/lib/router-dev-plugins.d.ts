export namespace ROUTER_MODE_DEV_PLUGINS {
    function spa(router: import("./router-modes.js").SPARouterSchema): (import("./vite-dev.js").Plugin | import("./vite-dev.js").Plugin[] | null)[];
    function http(router: import("./router-modes.js").HTTPRouterSchema): (import("./vite-dev.js").Plugin | import("./vite-dev.js").Plugin[] | null)[];
    function client(router: import("./router-modes.js").ClientRouterSchema): (import("./vite-dev.js").Plugin | import("./vite-dev.js").Plugin[] | null)[];
}
//# sourceMappingURL=router-dev-plugins.d.ts.map