"use strict";
Object.defineProperty(exports, Symbol.toStringTag, { value: "Module" });
const React = require("react");
const invariant = require("tiny-invariant");
function renderRsc(input) {
  if (React.isValidElement(input)) {
    return input;
  }
  if (typeof input === "object" && !input.state) {
    input.state = {
      status: "pending",
      promise: Promise.resolve().then(() => {
        invariant(false, "renderRSC() is coming soon!");
      }).then((element) => {
        input.state.value = element;
        input.state.status = "success";
      }).catch((err) => {
        input.state.status = "error";
        input.state.error = err;
      })
    };
  }
  if (input.state.status === "pending") {
    throw input.state.promise;
  }
  return input.state.value;
}
exports.renderRsc = renderRsc;
//# sourceMappingURL=renderRSC.cjs.map
