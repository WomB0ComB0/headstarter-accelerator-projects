import { CommandContext } from "../command_handler.js";
import { ExecuteResult } from "../result.js";
export declare function rmCommand(context: CommandContext): Promise<ExecuteResult>;
interface RmFlags {
    force: boolean;
    recursive: boolean;
    dir: boolean;
    paths: string[];
}
export declare function parseArgs(args: string[]): RmFlags;
export {};
