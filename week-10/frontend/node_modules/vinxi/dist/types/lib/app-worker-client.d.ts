export class AppWorkerClient {
    /**
     *
     * @param {URL} url
     */
    constructor(url: URL);
    /** @type {import('node:worker_threads').Worker | null} */
    worker: import('node:worker_threads').Worker | null;
    /** @type {Map<string, (event: any) => void>} */
    responses: Map<string, (event: any) => void>;
    url: URL;
    /**
     *
     * @param {any} workerData
     * @param {() => void} onReload
     */
    init(workerData: any, onReload: () => void): Promise<void>;
    /**
     *
     * @param {import('h3').H3Event} event
     * @returns
     */
    handle(event: import('h3').H3Event): null;
    close(): void;
}
//# sourceMappingURL=app-worker-client.d.ts.map