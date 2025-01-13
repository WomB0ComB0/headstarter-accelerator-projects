/**
 *
 * @param {string} id
 * @returns {import('node:http').ServerResponse}
 */
export function createServerResponse(id: string, { onChunk, onHeader, onFinish, }: {
    onChunk?: ((chunk: any, encoding: any) => void) | undefined;
    onHeader?: ((header: any, value: any) => void) | undefined;
    onFinish?: (() => void) | undefined;
}): import('node:http').ServerResponse;
/**
 *
 * @returns {import('node:http').IncomingMessage}
 */
export function createIncomingMessage(url: any, method: any, headers: any): import('node:http').IncomingMessage;
//# sourceMappingURL=http-stream.d.ts.map