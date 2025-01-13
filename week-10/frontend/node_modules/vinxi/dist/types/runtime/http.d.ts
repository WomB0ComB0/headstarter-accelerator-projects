/**
 *
 * @param {{ onRequest?: import("h3")._RequestMiddleware | import("h3")._RequestMiddleware[]; onBeforeResponse?: import("h3")._ResponseMiddleware | import("h3")._ResponseMiddleware[] }} options
 * @returns
 */
export function defineMiddleware(options: {
    onRequest?: import("h3")._RequestMiddleware | import("h3")._RequestMiddleware[];
    onBeforeResponse?: import("h3")._ResponseMiddleware | import("h3")._ResponseMiddleware[];
}): {
    onRequest?: import("h3")._RequestMiddleware<import("h3").EventHandlerRequest> | import("h3")._RequestMiddleware<import("h3").EventHandlerRequest>[] | undefined;
    onBeforeResponse?: import("h3")._ResponseMiddleware<import("h3").EventHandlerRequest, any> | import("h3")._ResponseMiddleware<import("h3").EventHandlerRequest, any>[] | undefined;
};
export function toWebRequest(event: import('h3').H3Event): Request | undefined;
export function isEvent(obj: any): boolean;
export function getEvent(): any;
export function handleHTTPEvent(event: any): Promise<any>;
export const HTTPEventSymbol: unique symbol;
export function readRawBody(...args: any[]): any;
export function readBody(...args: any[]): any;
export function getQuery(...args: any[]): any;
export function isMethod(...args: any[]): any;
export function isPreflightRequest(...args: any[]): any;
export function getValidatedQuery(...args: any[]): any;
export function getRouterParams(...args: any[]): any;
export function getRouterParam(...args: any[]): any;
export function getValidatedRouterParams(...args: any[]): any;
export function assertMethod(...args: any[]): any;
export function getRequestHeaders(...args: any[]): any;
export function getRequestHeader(...args: any[]): any;
export function getRequestURL(...args: any[]): any;
export function getRequestHost(...args: any[]): any;
export function getRequestProtocol(...args: any[]): any;
export function getRequestIP(...args: any[]): any;
export function send(...args: any[]): any;
export function sendNoContent(...args: any[]): any;
export function setResponseStatus(...args: any[]): any;
export function getResponseStatus(...args: any[]): any;
export function getResponseStatusText(...args: any[]): any;
export function getResponseHeaders(...args: any[]): any;
export function getResponseHeader(...args: any[]): any;
export function setResponseHeaders(...args: any[]): any;
export function setResponseHeader(...args: any[]): any;
export function appendResponseHeaders(...args: any[]): any;
export function appendResponseHeader(...args: any[]): any;
export function defaultContentType(...args: any[]): any;
export function sendRedirect(...args: any[]): any;
export function sendStream(...args: any[]): any;
export function writeEarlyHints(...args: any[]): any;
export function sendError(...args: any[]): any;
export function sendProxy(...args: any[]): any;
export function proxyRequest(...args: any[]): any;
export function fetchWithEvent(...args: any[]): any;
export function getProxyRequestHeaders(...args: any[]): any;
export function parseCookies(...args: any[]): any;
export function getCookie(...args: any[]): any;
export function setCookie(...args: any[]): any;
export function deleteCookie(...args: any[]): any;
export function useBase(...args: any[]): any;
export function useSession(...args: any[]): any;
export function getSession(...args: any[]): any;
export function updateSession(...args: any[]): any;
export function sealSession(...args: any[]): any;
export function unsealSession(...args: any[]): any;
export function clearSession(...args: any[]): any;
export function handleCacheHeaders(...args: any[]): any;
export function handleCors(...args: any[]): any;
export function appendCorsHeaders(...args: any[]): any;
export function appendCorsPreflightHeaders(...args: any[]): any;
export function sendWebResponse(...args: any[]): any;
export function appendHeader(...args: any[]): any;
export function appendHeaders(...args: any[]): any;
export function setHeader(...args: any[]): any;
export function setHeaders(...args: any[]): any;
export function getHeader(...args: any[]): any;
export function getHeaders(...args: any[]): any;
export function getRequestFingerprint(...args: any[]): any;
export function getRequestWebStream(...args: any[]): any;
export function readFormData(...args: any[]): any;
export function readMultipartFormData(...args: any[]): any;
export function readValidatedBody(...args: any[]): any;
export function removeResponseHeader(...args: any[]): any;
export function getContext(...args: any[]): any;
export function setContext(...args: any[]): any;
export function clearResponseHeaders(...args: any[]): any;
export function getWebRequest(...args: any[]): any;
export { H3Error, H3Event, MIMES, callNodeListener, createApp, createAppEventHandler, createEvent, createRouter, defineEventHandler, defineLazyEventHandler, defineNodeListener, defineNodeMiddleware, defineRequestMiddleware, defineResponseMiddleware, dynamicEventHandler, defineWebSocket, eventHandler, splitCookiesString, fromNodeMiddleware, fromPlainHandler, fromWebHandler, isError, isEventHandler, isWebResponse, lazyEventHandler, promisifyNodeListener, serveStatic, toEventHandler, toNodeListener, toPlainHandler, toWebHandler, isCorsOriginAllowed, isStream, createError, sanitizeStatusCode, sanitizeStatusMessage, createApp as createServer } from "h3";
//# sourceMappingURL=http.d.ts.map