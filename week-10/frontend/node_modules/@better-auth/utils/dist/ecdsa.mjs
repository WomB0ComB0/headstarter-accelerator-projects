const ecdsa = {
  generateKeyPair: async (curve = "P-256") => {
    const keyPair = await crypto.subtle.generateKey(
      {
        name: "ECDSA",
        namedCurve: curve
      },
      true,
      ["sign", "verify"]
    );
    const privateKey = await crypto.subtle.exportKey(
      "pkcs8",
      keyPair.privateKey
    );
    const publicKey = await crypto.subtle.exportKey("spki", keyPair.publicKey);
    return { privateKey, publicKey };
  },
  importPrivateKey: async (privateKey, curve, extractable = false) => {
    if (typeof privateKey === "string") {
      privateKey = new TextEncoder().encode(privateKey);
    }
    return await crypto.subtle.importKey(
      "pkcs8",
      privateKey,
      {
        name: "ECDSA",
        namedCurve: curve
      },
      extractable,
      ["sign"]
    );
  },
  importPublicKey: async (publicKey, curve, extractable = false) => {
    if (typeof publicKey === "string") {
      publicKey = new TextEncoder().encode(publicKey);
    }
    return await crypto.subtle.importKey(
      "spki",
      publicKey,
      {
        name: "ECDSA",
        namedCurve: curve
      },
      extractable,
      ["verify"]
    );
  },
  sign: async (privateKey, data, hash = "SHA-256") => {
    if (typeof data === "string") {
      data = new TextEncoder().encode(data);
    }
    const signature = await crypto.subtle.sign(
      {
        name: "ECDSA",
        hash: { name: hash }
      },
      privateKey,
      data
    );
    return signature;
  },
  verify: async (publicKey, {
    signature,
    data,
    hash = "SHA-256"
  }) => {
    if (typeof signature === "string") {
      signature = new TextEncoder().encode(signature);
    }
    if (typeof data === "string") {
      data = new TextEncoder().encode(data);
    }
    return await crypto.subtle.verify(
      {
        name: "ECDSA",
        hash: { name: hash }
      },
      publicKey,
      signature,
      data
    );
  },
  exportKey: async (key, format) => {
    return await crypto.subtle.exportKey(format, key);
  }
};

export { ecdsa };
