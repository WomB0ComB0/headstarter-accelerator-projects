export declare enum Keys {
    Up = 0,
    Down = 1,
    Left = 2,
    Right = 3,
    Enter = 4,
    Space = 5,
    Backspace = 6
}
export declare function readKeys(): AsyncGenerator<string | Keys, void, unknown>;
export declare function hideCursor(): void;
export declare function showCursor(): void;
export declare let isOutputTty: boolean;
export declare function setNotTtyForTesting(): void;
export declare function resultOrExit<T>(result: T | undefined): T;
export interface SelectionOptions<TReturn> {
    message: string;
    render: () => TextItem[];
    noClear: boolean | undefined;
    onKey: (key: string | Keys) => TReturn | undefined;
}
export declare function createSelection<TReturn>(options: SelectionOptions<TReturn>): Promise<TReturn | undefined>;
export type TextItem = string | HangingTextItem;
export interface HangingTextItem {
    text: string;
    indent: number;
}
export interface ConsoleSize {
    columns: number;
    rows: number;
}
export declare function safeConsoleSize(): ConsoleSize | undefined;
export declare const staticText: {
    set(items: TextItem[], size?: ConsoleSize): void;
    outputItems(items: TextItem[], size?: ConsoleSize): void;
    clear(size?: ConsoleSize): void;
};
