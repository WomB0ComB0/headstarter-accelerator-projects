type EnvOnlyFn = <TFn extends (...args: Array<any>) => any>(fn: TFn) => TFn;
export declare const serverOnly: EnvOnlyFn;
export declare const clientOnly: EnvOnlyFn;
export {};
