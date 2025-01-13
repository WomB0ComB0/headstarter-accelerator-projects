import { RouteNode } from './types.js';
import { Config } from './config.js';
export declare const CONSTANTS: {
    APIRouteExportVariable: string;
};
export declare function generator(config: Config, root: string): Promise<void>;
/**
 * Removes the last segment from a given path. Segments are considered to be separated by a '/'.
 *
 * @param {string} routePath - The path from which to remove the last segment. Defaults to '/'.
 * @returns {string} The path with the last segment removed.
 * @example
 * removeLastSegmentFromPath('/workspace/_auth/foo') // '/workspace/_auth'
 */
export declare function removeLastSegmentFromPath(routePath?: string): string;
export declare function hasParentRoute(routes: Array<RouteNode>, node: RouteNode, routePathToCheck: string | undefined): RouteNode | null;
/**
 * Gets the final variable name for a route
 */
export declare const getResolvedRouteNodeVariableName: (routeNode: RouteNode) => string;
/**
 * Creates a map from fullPath to routeNode
 */
export declare const createRouteNodesByFullPath: (routeNodes: Array<RouteNode>) => Map<string, RouteNode>;
/**
 * Create a map from 'to' to a routeNode
 */
export declare const createRouteNodesByTo: (routeNodes: Array<RouteNode>) => Map<string, RouteNode>;
/**
 * Create a map from 'id' to a routeNode
 */
export declare const createRouteNodesById: (routeNodes: Array<RouteNode>) => Map<string, RouteNode>;
/**
 * Infers the full path for use by TS
 */
export declare const inferFullPath: (routeNode: RouteNode) => string;
/**
 * Infers the path for use by TS
 */
export declare const inferPath: (routeNode: RouteNode) => string;
/**
 * Infers to path
 */
export declare const inferTo: (routeNode: RouteNode) => string;
/**
 * Dedupes branches and index routes
 */
export declare const dedupeBranchesAndIndexRoutes: (routes: Array<RouteNode>) => Array<RouteNode>;
export type StartAPIRoutePathSegment = {
    value: string;
    type: 'path' | 'param' | 'splat';
};
/**
 * This function takes in a path in the format accepted by TanStack Router
 * and returns an array of path segments that can be used to generate
 * the pathname of the TanStack Start API route.
 *
 * @param src
 * @returns
 */
export declare function startAPIRouteSegmentsFromTSRFilePath(src: string, config: Config): Array<StartAPIRoutePathSegment>;
