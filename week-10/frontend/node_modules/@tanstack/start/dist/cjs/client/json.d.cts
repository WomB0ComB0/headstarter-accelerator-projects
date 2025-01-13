import { JsonResponse } from './createServerFn.cjs';
export declare function json<TData>(payload: TData, init?: ResponseInit): JsonResponse<TData>;
