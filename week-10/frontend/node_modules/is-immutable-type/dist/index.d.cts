import ts from "typescript";
import { TypeDeclarationFileSpecifier, TypeDeclarationLibSpecifier, TypeDeclarationPackageSpecifier } from "ts-declaration-location";
/**
 * The immutability values are sorted in ascending order.
 */
declare enum Immutability {
    // eslint-disable-next-line ts/prefer-literal-enum-member
    Unknown,
    // MutableDeep = 1,
    Mutable = 2,
    // MutableShallow = 2,
    // Readonly = 3,
    ReadonlyShallow = 3,
    ReadonlyDeep = 4,
    Immutable = 5,
    // eslint-disable-next-line ts/prefer-literal-enum-member
    Calculating
}
type PatternSpecifier = {
    name?: string | string[];
    ignoreName?: string | string[];
    pattern?: RegExp | RegExp[];
    ignorePattern?: RegExp | RegExp[];
} & ({
    name: string | string[];
} | {
    pattern: RegExp | RegExp[];
});
type FileSpecifier = PatternSpecifier & TypeDeclarationFileSpecifier;
type LibSpecifier = PatternSpecifier & TypeDeclarationLibSpecifier;
type PackageSpecifier = PatternSpecifier & TypeDeclarationPackageSpecifier;
/**
 * How a type can be specified.
 */
type TypeSpecifier = string | RegExp | FileSpecifier | LibSpecifier | PackageSpecifier;
/**
 * Check if a type matches a specifier.
 */
type TypeMatchesPatternSpecifier = (program: ts.Program, type: ts.Type, typeNode: ts.TypeNode | null, include: ReadonlyArray<string | RegExp>, exclude: ReadonlyArray<string | RegExp>) => boolean;
/**
 * A list of immutability overrides.
 */
type ImmutabilityOverrides = ReadonlyArray<{
    type: TypeSpecifier;
    to: Immutability;
    from?: Immutability;
}>;
/**
 * Get the default overrides that are applied.
 */
declare function getDefaultOverrides(): ImmutabilityOverrides;
/**
 * A cache used to keep track of what types have already been calculated.
 */
type ImmutabilityCache = WeakMap<object, Immutability>;
/**
 * Get the immutability of the given type.
 *
 * If you only care about the immutability up to a certain point, a
 * `maxImmutability` can be specified to help improve performance.
 *
 * @param program - The TypeScript Program to use.
 * @param typeOrTypeNode - The type to test the immutability of.
 * @param overrides - The overrides to use when calculating the immutability.
 * @param useCache - Either a custom cache to use, `true` to use the global
 * cache, or `false` to not use any predefined cache.
 * @param maxImmutability - If set then any return value equal to or greater
 * than this value will state the type's minimum immutability rather than it's
 * actual. This allows for early-escapes to be made in the type calculation.
 * @param typeMatchesPatternSpecifier - Allows for overriding how we check if a
 * type matches a pattern. This is used for checking if an override should be
 * applied or not.
 */
declare function getTypeImmutability(program: ts.Program, typeOrTypeNode: ts.Type | ts.TypeNode, overrides?: ImmutabilityOverrides, useCache?: ImmutabilityCache | boolean, maxImmutability?: Immutability, typeMatchesPatternSpecifier?: TypeMatchesPatternSpecifier): Immutability;
/**
 * Get the minimum immutability from the given values.
 *
 * Note: Unknown immutability will be ignore; thus Unknown will be return if
 * and only if all values are Unknown.
 */
declare function min(a: Immutability, b: Immutability): Immutability;
/**
 * Get the maximum immutability from the given values.
 *
 * Note: Unknown immutability will be ignore; thus Unknown will be return if
 * and only if all values are Unknown.
 */
declare function max(a: Immutability, b: Immutability): Immutability;
/**
 * Clamp the immutability between min and max.
 */
declare function clamp(minValue: Immutability, value: Immutability, maxValue: Immutability): number;
/**
 * Is the given immutability immutable?
 */
declare function isImmutable(immutability: Immutability): boolean;
/**
 * Is the given immutability at least ReadonlyDeep?
 */
declare function isReadonlyDeep(immutability: Immutability): boolean;
/**
 * Is the given immutability at least ReadonlyShallow?
 */
declare function isReadonlyShallow(immutability: Immutability): boolean;
/**
 * Is the given immutability Mutable?
 */
declare function isMutable(immutability: Immutability): boolean;
/**
 * Is the given immutability unknown?
 */
declare function isUnknown(value: Immutability): boolean;
/**
 * Is the immutability of the given type immutable.
 *
 * @param program - The TypeScript Program to use.
 * @param typeOrTypeNode - The type to test the immutability of.
 * @param overrides - The overrides to use when calculating the immutability.
 * @param useCache - Either a custom cache to use, `true` to use the global
 * cache, or `false` to not use any predefined cache.
 */
declare function isImmutableType(program: ts.Program, typeOrTypeNode: ts.Type | ts.TypeNode, overrides?: ImmutabilityOverrides, useCache?: ImmutabilityCache | boolean): boolean;
/**
 * Is the immutability of the given type at least readonly deep.
 *
 * @param program - The TypeScript Program to use.
 * @param typeOrTypeNode - The type to test the immutability of.
 * @param overrides - The overrides to use when calculating the immutability.
 * @param useCache - Either a custom cache to use, `true` to use the global
 * cache, or `false` to not use any predefined cache.
 */
declare function isReadonlyDeepType(program: ts.Program, typeOrTypeNode: ts.Type | ts.TypeNode, overrides?: ImmutabilityOverrides, useCache?: ImmutabilityCache | boolean): boolean;
/**
 * Is the immutability of the given type at least readonly shallow.
 *
 * @param program - The TypeScript Program to use.
 * @param typeOrTypeNode - The type to test the immutability of.
 * @param overrides - The overrides to use when calculating the immutability.
 * @param useCache - Either a custom cache to use, `true` to use the global
 * cache, or `false` to not use any predefined cache.
 */
declare function isReadonlyShallowType(program: ts.Program, typeOrTypeNode: ts.Type | ts.TypeNode, overrides?: ImmutabilityOverrides, useCache?: ImmutabilityCache | boolean): boolean;
/**
 * Is the immutability of the given type mutable.
 *
 * @param program - The TypeScript Program to use.
 * @param typeOrTypeNode - The type to test the immutability of.
 * @param overrides - The overrides to use when calculating the immutability.
 * @param useCache - Either a custom cache to use, `true` to use the global
 * cache, or `false` to not use any predefined cache.
 */
declare function isMutableType(program: ts.Program, typeOrTypeNode: ts.Type | ts.TypeNode, overrides?: ImmutabilityOverrides, useCache?: ImmutabilityCache | boolean): boolean;
export { getDefaultOverrides, getTypeImmutability, ImmutabilityCache, ImmutabilityOverrides, clamp, isImmutable, isMutable, isReadonlyDeep, isReadonlyShallow, isUnknown, max, min, Immutability, isImmutableType, isMutableType, isReadonlyDeepType, isReadonlyShallowType, TypeMatchesPatternSpecifier, TypeSpecifier };
