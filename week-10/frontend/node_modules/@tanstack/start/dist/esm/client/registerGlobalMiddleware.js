const globalMiddleware = [];
function registerGlobalMiddleware(options) {
  globalMiddleware.push(...options.middleware);
}
export {
  globalMiddleware,
  registerGlobalMiddleware
};
//# sourceMappingURL=registerGlobalMiddleware.js.map
