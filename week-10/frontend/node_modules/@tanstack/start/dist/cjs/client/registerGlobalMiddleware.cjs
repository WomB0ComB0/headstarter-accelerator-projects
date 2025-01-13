"use strict";
Object.defineProperty(exports, Symbol.toStringTag, { value: "Module" });
const globalMiddleware = [];
function registerGlobalMiddleware(options) {
  globalMiddleware.push(...options.middleware);
}
exports.globalMiddleware = globalMiddleware;
exports.registerGlobalMiddleware = registerGlobalMiddleware;
//# sourceMappingURL=registerGlobalMiddleware.cjs.map
