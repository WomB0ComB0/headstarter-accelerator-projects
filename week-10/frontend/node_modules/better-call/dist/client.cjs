"use strict";
var __defProp = Object.defineProperty;
var __getOwnPropDesc = Object.getOwnPropertyDescriptor;
var __getOwnPropNames = Object.getOwnPropertyNames;
var __hasOwnProp = Object.prototype.hasOwnProperty;
var __export = (target, all) => {
  for (var name in all)
    __defProp(target, name, { get: all[name], enumerable: true });
};
var __copyProps = (to, from, except, desc) => {
  if (from && typeof from === "object" || typeof from === "function") {
    for (let key of __getOwnPropNames(from))
      if (!__hasOwnProp.call(to, key) && key !== except)
        __defProp(to, key, { get: () => from[key], enumerable: !(desc = __getOwnPropDesc(from, key)) || desc.enumerable });
  }
  return to;
};
var __toCommonJS = (mod) => __copyProps(__defProp({}, "__esModule", { value: true }), mod);

// src/client.ts
var client_exports = {};
__export(client_exports, {
  createClient: () => createClient
});
module.exports = __toCommonJS(client_exports);
var import_fetch = require("@better-fetch/fetch");
var createClient = (options) => {
  const fetch = (0, import_fetch.createFetch)(options);
  return async (path, ...options2) => {
    return await fetch(path, {
      ...options2[0]
    });
  };
};
// Annotate the CommonJS export names for ESM import in node:
0 && (module.exports = {
  createClient
});
//# sourceMappingURL=client.cjs.map