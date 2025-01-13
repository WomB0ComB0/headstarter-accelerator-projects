'use strict';

const uncrypto = require('uncrypto');

const rsa = {
  generateKeyPair: async (modulusLength = 2048, hash = "SHA-256") => {
    return await uncrypto.subtle.generateKey(
      {
        name: "RSA-OAEP",
        modulusLength,
        publicExponent: new Uint8Array([1, 0, 1]),
        hash: { name: hash }
      },
      true,
      ["encrypt", "decrypt"]
    );
  },
  exportKey: async (key, format) => {
    return await uncrypto.subtle.exportKey(format, key);
  },
  importKey: async (key, usage = "encrypt", hash = "SHA-256") => {
    return await uncrypto.subtle.importKey(
      "jwk",
      key,
      {
        name: "RSA-OAEP",
        hash: { name: hash }
      },
      true,
      [usage]
    );
  },
  encrypt: async (key, data) => {
    const encodedData = typeof data === "string" ? new TextEncoder().encode(data) : data;
    return await uncrypto.subtle.encrypt({ name: "RSA-OAEP" }, key, encodedData);
  },
  decrypt: async (key, data) => {
    return await uncrypto.subtle.decrypt({ name: "RSA-OAEP" }, key, data);
  },
  sign: async (key, data, saltLength = 32) => {
    const encodedData = typeof data === "string" ? new TextEncoder().encode(data) : data;
    return await uncrypto.subtle.sign(
      {
        name: "RSA-PSS",
        saltLength
      },
      key,
      encodedData
    );
  },
  verify: async (key, {
    signature,
    data,
    saltLength = 32
  }) => {
    if (typeof signature === "string") {
      signature = new TextEncoder().encode(signature);
    }
    const encodedData = typeof data === "string" ? new TextEncoder().encode(data) : data;
    return await uncrypto.subtle.verify(
      {
        name: "RSA-PSS",
        saltLength
      },
      key,
      signature,
      encodedData
    );
  }
};

exports.rsa = rsa;
