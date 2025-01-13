// src/client.ts
import { createFetch } from "@better-fetch/fetch";
var createClient = (options) => {
  const fetch = createFetch(options);
  return async (path, ...options2) => {
    return await fetch(path, {
      ...options2[0]
    });
  };
};
export {
  createClient
};
//# sourceMappingURL=client.js.map