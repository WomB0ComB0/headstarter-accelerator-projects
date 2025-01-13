type Alphabet = "a-z" | "A-Z" | "0-9" | "-_";
declare function createRandomStringGenerator<A extends Alphabet>(...characters: A[]): <SubA extends Alphabet>(length: number, alphabet?: SubA | undefined, ...arg_1: SubA[]) => string;

export { createRandomStringGenerator };
