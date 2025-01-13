import { ConsoleSize, TextItem } from "../utils.js";
export interface RenderIntervalProgressBar {
    render(size: ConsoleSize): TextItem[];
}
export declare function addProgressBar(render: (size: ConsoleSize) => TextItem[]): RenderIntervalProgressBar;
export declare function removeProgressBar(pb: RenderIntervalProgressBar): boolean;
export declare function forceRender(): void;
export declare function isShowingProgressBars(): boolean;
