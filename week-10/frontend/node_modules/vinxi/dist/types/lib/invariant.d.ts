/**
 * @param {any} condition
 * @param {string | number} message
 * @returns {asserts condition}
 */
export default function invariant(condition: any, message: string | number): asserts condition;
export class InvariantError extends Error {
    constructor(message?: string | number);
    framesToPop: number;
}
//# sourceMappingURL=invariant.d.ts.map