import { AnyMiddleware } from './createMiddleware.cjs';
export declare const globalMiddleware: Array<AnyMiddleware>;
export declare function registerGlobalMiddleware(options: {
    middleware: Array<AnyMiddleware>;
}): void;
