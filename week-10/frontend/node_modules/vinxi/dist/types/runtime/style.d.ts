/**
 *
 * @param {{ attrs: Record<string, string>; children: string }[]} styles
 * @param {*} data
 */
export function updateStyles(styles: {
    attrs: Record<string, string>;
    children: string;
}[], data: any): void;
/**
 *
 * @param {{ attrs: Record<string, string>; children: string }[]} styles
 */
export function preloadStyles(styles: {
    attrs: Record<string, string>;
    children: string;
}[]): void;
/**
 *
 * @param {{ attrs: Record<string, string>; children: string }[]} styles
 */
export function appendStyles(styles: {
    attrs: Record<string, string>;
    children: string;
}[]): void;
/**
 *
 * @param {{ attrs: Record<string, string>}[]} styles
 */
export function cleanupStyles(styles: {
    attrs: Record<string, string>;
}[]): void;
//# sourceMappingURL=style.d.ts.map