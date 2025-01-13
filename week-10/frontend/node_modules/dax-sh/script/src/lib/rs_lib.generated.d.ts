/**
 * @param {string} command
 * @returns {any}
 */
export function parse(command: string): any;
/**
 * @param {any} items
 * @param {number} cols
 * @param {number} rows
 * @returns {string | undefined}
 */
export function static_text_render_text(items: any, cols: number, rows: number): string | undefined;
/**
 * @param {number} cols
 * @param {number} rows
 * @returns {string | undefined}
 */
export function static_text_clear_text(cols: number, rows: number): string | undefined;
/**
 * @param {any} items
 * @param {number} cols
 * @param {number} rows
 * @returns {string | undefined}
 */
export function static_text_render_once(items: any, cols: number, rows: number): string | undefined;
/**
 * @param {string} text
 * @returns {string}
 */
export function strip_ansi_codes(text: string): string;
/** Instantiates an instance of the Wasm module returning its functions.
 * @remarks It is safe to call this multiple times and once successfully
 * loaded it will always return a reference to the same object.
 */
export function instantiate(): {
    parse: typeof parse;
    static_text_render_text: typeof static_text_render_text;
    static_text_clear_text: typeof static_text_clear_text;
    static_text_render_once: typeof static_text_render_once;
    strip_ansi_codes: typeof strip_ansi_codes;
};
/** Instantiates an instance of the Wasm module along with its exports.
 * @remarks It is safe to call this multiple times and once successfully
 * loaded it will always return a reference to the same object.
 * @returns {{
 *   instance: WebAssembly.Instance;
 *   exports: { parse: typeof parse; static_text_render_text: typeof static_text_render_text; static_text_clear_text: typeof static_text_clear_text; static_text_render_once: typeof static_text_render_once; strip_ansi_codes: typeof strip_ansi_codes }
 * }}
 */
export function instantiateWithInstance(): {
    instance: WebAssembly.Instance;
    exports: {
        parse: typeof parse;
        static_text_render_text: typeof static_text_render_text;
        static_text_clear_text: typeof static_text_clear_text;
        static_text_render_once: typeof static_text_render_once;
        strip_ansi_codes: typeof strip_ansi_codes;
    };
};
/** Gets if the Wasm module has been instantiated. */
export function isInstantiated(): boolean;
