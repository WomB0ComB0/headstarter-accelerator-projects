import { ConsoleSize, TextItem } from "./utils.js";
export declare enum LoggerRefreshItemKind {
    ProgressBars = 0,
    Selection = 1
}
declare function setItems(kind: LoggerRefreshItemKind, items: TextItem[] | undefined, size?: ConsoleSize): void;
declare function logAboveStaticText(inner: () => void, providedSize?: ConsoleSize): void;
declare function logOnce(items: TextItem[], size?: ConsoleSize): void;
declare const logger: {
    setItems: typeof setItems;
    logOnce: typeof logOnce;
    logAboveStaticText: typeof logAboveStaticText;
};
export { logger };
