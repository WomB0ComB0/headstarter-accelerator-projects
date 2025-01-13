import { CommandContext } from "../command_handler.js";
import { ExecuteResult } from "../result.js";
interface CatFlags {
    paths: string[];
}
export declare function catCommand(context: CommandContext): Promise<ExecuteResult>;
export declare function parseCatArgs(args: string[]): CatFlags;
export {};
