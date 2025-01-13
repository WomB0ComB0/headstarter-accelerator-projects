export default findAssetsInViteManifest;
/**
 * Traverses the module graph and collects assets for a given chunk
 *
 * @param {any} manifest Client manifest
 * @param {string} id Chunk id
 * @param {Map<string, string[]>} assetMap Cache of assets
 * @param {string[]} stack Stack of chunk ids to prevent circular dependencies
 * @returns Array of asset URLs
 */
declare function findAssetsInViteManifest(manifest: any, id: string, assetMap?: Map<string, string[]>, stack?: string[]): any[];
//# sourceMappingURL=vite-manifest.d.ts.map