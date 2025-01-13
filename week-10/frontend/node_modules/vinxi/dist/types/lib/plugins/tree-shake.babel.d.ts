export { treeShake as default };
declare function treeShake({ types: t }: {
    types: any;
}): {
    visitor: {
        Program: {
            enter(path: any, state: any): void;
        };
    };
};
//# sourceMappingURL=tree-shake.babel.d.ts.map