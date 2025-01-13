import type { configSchema } from '@tanstack/router-generator';
import type { AppOptions as VinxiAppOptions, RouterSchemaInput as VinxiRouterSchemaInput } from 'vinxi';
import type { z } from 'zod';
export declare function tanstackStartVinxiFileRouter(opts: {
    tsrConfig: z.infer<typeof configSchema>;
    apiBase: string;
}): (router: VinxiRouterSchemaInput, app: VinxiAppOptions) => {
    toPath(src: string): string;
    toRoute(src: string): {
        path: string;
        filePath: string;
        $APIRoute: {
            src: string;
            pick: string[];
        } | undefined;
    };
    routes: any[];
    routerConfig: import("vinxi/dist/types/lib/router-mode").Router<any>;
    appConfig: import("vinxi/dist/types/lib/app").AppOptions;
    config: import("vinxi/dist/types/lib/fs-router").FileSystemRouterConfig;
    glob(): string;
    buildRoutes(): Promise<any[]>;
    isRoute(src: any): boolean;
    update: undefined;
    _addRoute(route: import("vinxi/dist/types/lib/fs-router").Route): void;
    addRoute(src: string): Promise<void>;
    reload(route: string): void;
    updateRoute(src: string): Promise<void>;
    removeRoute(src: string): void;
    buildRoutesPromise: Promise<any[]> | undefined;
    getRoutes(): Promise<any[]>;
    addEventListener(type: string, callback: EventListenerOrEventListenerObject | null, options?: AddEventListenerOptions | boolean): void;
    dispatchEvent(event: Event): boolean;
    removeEventListener(type: string, callback: EventListenerOrEventListenerObject | null, options?: EventListenerOptions | boolean): void;
};
