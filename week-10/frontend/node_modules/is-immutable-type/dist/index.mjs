import { getTypeOfPropertyOfType } from '@typescript-eslint/type-utils';
import { isIntrinsicErrorType, isTypeReference, isUnionType, isIntersectionType, isConditionalType, isObjectType, isPropertyReadonlyInType, isSymbolFlagSet, hasType, isIntrinsicType } from 'ts-api-utils';
import ts from 'typescript';
import typeMatchesSpecifier from 'ts-declaration-location';

/**
 * The immutability values are sorted in ascending order.
 */
var Immutability;
(function (Immutability) {
    // eslint-disable-next-line ts/prefer-literal-enum-member
    Immutability[Immutability["Unknown"] = Number.NaN] = "Unknown";
    // MutableDeep = 1,
    Immutability[Immutability["Mutable"] = 2] = "Mutable";
    // MutableShallow = 2,
    // Readonly = 3,
    Immutability[Immutability["ReadonlyShallow"] = 3] = "ReadonlyShallow";
    Immutability[Immutability["ReadonlyDeep"] = 4] = "ReadonlyDeep";
    Immutability[Immutability["Immutable"] = 5] = "Immutable";
    // eslint-disable-next-line ts/prefer-literal-enum-member
    Immutability[Immutability["Calculating"] = Number.POSITIVE_INFINITY] = "Calculating";
})(Immutability || (Immutability = {}));

/**
 * Get the minimum immutability from the given values.
 *
 * Note: Unknown immutability will be ignore; thus Unknown will be return if
 * and only if all values are Unknown.
 */
function min(a, b) {
    if (isUnknown(a)) {
        return b;
    }
    if (isUnknown(b)) {
        return a;
    }
    return Math.min(a, b);
}
/**
 * Get the maximum immutability from the given values.
 *
 * Note: Unknown immutability will be ignore; thus Unknown will be return if
 * and only if all values are Unknown.
 */
function max(a, b) {
    if (isUnknown(a)) {
        return b;
    }
    if (isUnknown(b)) {
        return a;
    }
    return Math.max(a, b);
}
/**
 * Clamp the immutability between min and max.
 */
function clamp(minValue, value, maxValue) {
    return Math.max(minValue, Math.min(maxValue, value));
}
/**
 * Is the given immutability immutable?
 */
function isImmutable(immutability) {
    return immutability >= Immutability.Immutable;
}
/**
 * Is the given immutability at least ReadonlyDeep?
 */
function isReadonlyDeep(immutability) {
    return immutability >= Immutability.ReadonlyDeep;
}
/**
 * Is the given immutability at least ReadonlyShallow?
 */
function isReadonlyShallow(immutability) {
    return immutability >= Immutability.ReadonlyShallow;
}
/**
 * Is the given immutability Mutable?
 */
function isMutable(immutability) {
    return immutability <= Immutability.Mutable;
}
/**
 * Is the given immutability unknown?
 */
function isUnknown(value) {
    return Number.isNaN(value);
}

/**
 * Type guard to check if a Node has a Symbol.
 */
function hasSymbol(node) {
    return Object.hasOwn(node, "symbol");
}
/**
 * Type guard to check if a Type is TypeNode.
 */
function isTypeNode(typeLike) {
    return Object.hasOwn(typeLike, "kind");
}
/**
 * Check if a type node is anonymous;
 */
function isAnonymousTypeNode(typeNode) {
    return typeNode.pos < 0;
}
/**
 * Get the type data from the given type or type node.
 *
 * @throws if the type is an error type.
 */
function getTypeData(type, typeNode) {
    if (isIntrinsicErrorType(type)) {
        throw new Error("ErrorType encountered.");
    }
    return {
        type,
        typeNode: typeNode === undefined ||
            typeNode === null ||
            isAnonymousTypeNode(typeNode)
            ? null
            : typeNode,
    };
}
/**
 * Cache a value by its type
 */
function cacheData(program, cache, typeData, value) {
    const checker = program.getTypeChecker();
    const identity = checker.getRecursionIdentity(typeData.type);
    if (typeData.typeNode !== null) {
        cache.set(typeData.typeNode, value);
    }
    cache.set(identity, value);
}
/**
 * Get a value by its cashed type.
 */
function getCachedData(program, cache, typeData) {
    const checker = program.getTypeChecker();
    const identity = typeData.typeNode ?? checker.getRecursionIdentity(typeData.type);
    return cache.get(identity);
}
/**
 * Does the given type/typeNode match the given specifier.
 */
function typeDataMatchesSpecifier(program, specifier, typeData, typeMatchesPatternSpecifier) {
    if (isIntrinsicErrorType(typeData.type)) {
        return false;
    }
    if (typeof specifier === "string" || specifier instanceof RegExp) {
        return typeNameMatchesSpecifier(program, specifier, typeData, typeMatchesPatternSpecifier);
    }
    return (typeMatchesSpecifier(program, specifier, typeData.type) &&
        typeNameMatchesSpecifier(program, specifier, typeData, typeMatchesPatternSpecifier));
}
/**
 * Test if the given type name matches the given specifier.
 */
function typeNameMatchesSpecifier(program, specifier, typeData, typeMatchesPatternSpecifier) {
    const names = typeof specifier === "string"
        ? [specifier]
        : specifier instanceof RegExp || specifier.name === undefined
            ? []
            : Array.isArray(specifier.name)
                ? specifier.name
                : [specifier.name];
    const patterns = specifier instanceof RegExp
        ? [specifier]
        : typeof specifier === "string" || specifier.pattern === undefined
            ? []
            : Array.isArray(specifier.pattern)
                ? specifier.pattern
                : [specifier.pattern];
    const ignoreNames = typeof specifier === "string" ||
        specifier instanceof RegExp ||
        specifier.ignoreName === undefined
        ? []
        : Array.isArray(specifier.ignoreName)
            ? specifier.ignoreName
            : [specifier.ignoreName];
    const ignorePatterns = typeof specifier === "string" ||
        specifier instanceof RegExp ||
        specifier.ignorePattern === undefined
        ? []
        : Array.isArray(specifier.ignorePattern)
            ? specifier.ignorePattern
            : [specifier.ignorePattern];
    const include = [...names, ...patterns];
    const exclude = [...ignoreNames, ...ignorePatterns];
    return typeMatchesPatternSpecifier(program, typeData.type, typeData.typeNode, include, exclude);
}
/**
 * The default `TypeMatchesPatternSpecifier`.
 */
function defaultTypeMatchesPatternSpecifier(program, type, typeNode, include, exclude = []) {
    if (include.length === 0) {
        return false;
    }
    let m_shouldInclude = false;
    const typeNameAlias = getTypeAliasName(type, typeNode);
    if (typeNameAlias !== null) {
        const testTypeNameAlias = (pattern) => typeof pattern === "string"
            ? pattern === typeNameAlias
            : pattern.test(typeNameAlias);
        if (exclude.some(testTypeNameAlias)) {
            return false;
        }
        m_shouldInclude ||= include.some(testTypeNameAlias);
    }
    const typeValue = getTypeAsString(program, type, typeNode);
    const testTypeValue = (pattern) => typeof pattern === "string"
        ? pattern === typeValue
        : pattern.test(typeValue);
    if (exclude.some(testTypeValue)) {
        return false;
    }
    m_shouldInclude ||= include.some(testTypeValue);
    const typeNameName = extractTypeName(typeValue);
    if (typeNameName !== null) {
        const testTypeNameName = (pattern) => typeof pattern === "string"
            ? pattern === typeNameName
            : pattern.test(typeNameName);
        if (exclude.some(testTypeNameName)) {
            return false;
        }
        m_shouldInclude ||= include.some(testTypeNameName);
    }
    // Special handling for arrays not written in generic syntax.
    if (program.getTypeChecker().isArrayType(type) && typeNode !== null) {
        if ((ts.isTypeOperatorNode(typeNode) &&
            typeNode.operator === ts.SyntaxKind.ReadonlyKeyword) ||
            (ts.isTypeOperatorNode(typeNode.parent) &&
                typeNode.parent.operator === ts.SyntaxKind.ReadonlyKeyword)) {
            const testIsReadonlyArray = (pattern) => typeof pattern === "string" && pattern === "ReadonlyArray";
            if (exclude.some(testIsReadonlyArray)) {
                return false;
            }
            m_shouldInclude ||= include.some(testIsReadonlyArray);
        }
        else {
            const testIsArray = (pattern) => typeof pattern === "string" && pattern === "Array";
            if (exclude.some(testIsArray)) {
                return false;
            }
            m_shouldInclude ||= include.some(testIsArray);
        }
    }
    return m_shouldInclude;
}
/**
 * Get the type alias name from the given type data.
 *
 * Null will be returned if the type is not a type alias.
 */
function getTypeAliasName(type, typeNode) {
    if (typeNode === null) {
        const t = "target" in type ? type.target : type;
        return t.aliasSymbol?.getName() ?? null;
    }
    return ts.isTypeAliasDeclaration(typeNode.parent)
        ? typeNode.parent.name.getText()
        : null;
}
/**
 * Get the type as a string.
 */
function getTypeAsString(program, type, typeNode) {
    return typeNode === null
        ? program
            .getTypeChecker()
            .typeToString(type, undefined, ts.TypeFormatFlags.AddUndefined |
            ts.TypeFormatFlags.NoTruncation |
            ts.TypeFormatFlags.OmitParameterModifiers |
            ts.TypeFormatFlags.UseFullyQualifiedType |
            ts.TypeFormatFlags.WriteArrayAsGenericType |
            ts.TypeFormatFlags.WriteArrowStyleSignature |
            ts.TypeFormatFlags.WriteTypeArgumentsOfSignature)
        : typeNode.getText();
}
/**
 * Get the type name extracted from the the type's string.
 *
 * This only work if the type is a type reference.
 */
function extractTypeName(typeValue) {
    const match = /^([^<]+)<.+>$/u.exec(typeValue);
    return match?.[1] ?? null;
}
/**
 * Get string representations of the given property name.
 */
function propertyNameToString(propertyName) {
    return ts.isIdentifier(propertyName) || ts.isPrivateIdentifier(propertyName)
        ? identifierToString(propertyName)
        : propertyName.getText();
}
/**
 * Get string representations of the given identifier.
 */
function identifierToString(identifier) {
    return identifier.getText();
}
/**
 * Is type a (non-namespace) function?
 */
function isFunction(type) {
    return (type.getCallSignatures().length > 0 && type.getProperties().length === 0);
}
/**
 * Is type a type reference with type arguments?
 */
function isTypeReferenceWithTypeArguments(type) {
    return (isTypeReference(type) &&
        type.typeArguments !== undefined &&
        type.typeArguments.length > 0);
}

/**
 * Get the default overrides that are applied.
 */
function getDefaultOverrides() {
    return [
        {
            type: { from: "lib", name: "Map" },
            to: Immutability.Mutable,
        },
        {
            type: { from: "lib", name: "Set" },
            to: Immutability.Mutable,
        },
        {
            type: { from: "lib", name: "Date" },
            to: Immutability.Mutable,
        },
        {
            type: { from: "lib", name: "URL" },
            to: Immutability.Mutable,
        },
        {
            type: { from: "lib", name: "URLSearchParams" },
            to: Immutability.Mutable,
        },
    ];
}
/**
 * A global cache that can be used between consumers.
 */
const globalCache = new WeakMap();
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
function getTypeImmutability(program, typeOrTypeNode, overrides = getDefaultOverrides(), useCache = true, maxImmutability = Immutability.Immutable, typeMatchesPatternSpecifier = defaultTypeMatchesPatternSpecifier) {
    const givenTypeNode = isTypeNode(typeOrTypeNode);
    const type = givenTypeNode
        ? program.getTypeChecker().getTypeFromTypeNode(typeOrTypeNode)
        : typeOrTypeNode;
    const typeNode = givenTypeNode ? typeOrTypeNode : undefined;
    const typeData = getTypeData(type, typeNode);
    return getTypeImmutabilityHelper(program, typeData, overrides, useCache, maxImmutability, typeMatchesPatternSpecifier);
}
/**
 * The stage of a taskState.
 */
var TaskStateStage;
(function (TaskStateStage) {
    TaskStateStage[TaskStateStage["Triage"] = 0] = "Triage";
    TaskStateStage[TaskStateStage["ReduceChildren"] = 1] = "ReduceChildren";
    TaskStateStage[TaskStateStage["ObjectProperties"] = 2] = "ObjectProperties";
    TaskStateStage[TaskStateStage["ObjectTypeReference"] = 3] = "ObjectTypeReference";
    TaskStateStage[TaskStateStage["ObjectIndexSignature"] = 4] = "ObjectIndexSignature";
    TaskStateStage[TaskStateStage["CheckDone"] = 5] = "CheckDone";
    TaskStateStage[TaskStateStage["ApplyOverride"] = 6] = "ApplyOverride";
    TaskStateStage[TaskStateStage["Done"] = 7] = "Done";
})(TaskStateStage || (TaskStateStage = {}));
/**
 * Get the immutability of the given type data.
 */
function getTypeImmutabilityHelper(program, td, overrides, useCache, maxImmutability, typeMatchesPatternSpecifier) {
    const cache = useCache === true
        ? globalCache
        : useCache === false
            ? new WeakMap()
            : useCache;
    const parameters = {
        program,
        overrides,
        cache,
        immutabilityLimits: {
            min: Immutability.Mutable,
            max: maxImmutability,
        },
        typeMatchesPatternSpecifier,
    };
    const m_stack = [createNewTaskState(td)];
    let m_PreviousImmutability = Immutability.Unknown;
    let m_state;
    do {
        m_state = m_stack.pop();
        switch (m_state.stage) {
            case 0 /* TaskStateStage.Triage */: {
                taskTriage(parameters, m_stack, m_state);
                break;
            }
            case 1 /* TaskStateStage.ReduceChildren */: {
                taskReduceChildren(m_state);
                break;
            }
            case 3 /* TaskStateStage.ObjectTypeReference */: {
                taskObjectTypeReference(m_stack, m_state);
                break;
            }
            case 4 /* TaskStateStage.ObjectIndexSignature */: {
                taskObjectIndexSignature(parameters, m_stack, m_state);
                break;
            }
            case 6 /* TaskStateStage.ApplyOverride */: {
                taskApplyOverride(m_state);
                m_state = m_state.taskState;
                break;
            }
            case 5 /* TaskStateStage.CheckDone */: {
                taskCheckDone(m_state, m_PreviousImmutability);
                m_state = m_state.taskState;
                break;
            }
        }
        if (m_state.immutability !== Immutability.Calculating) {
            cacheData(program, cache, m_state.typeData, m_state.immutability);
            m_PreviousImmutability = m_state.immutability;
        }
    } while (m_stack.length > 0);
    if (m_state.immutability === Immutability.Calculating) {
        // @ts-expect-error Unreachable Code
        return Immutability.Unknown;
    }
    return m_state.immutability;
}
/**
 * Create the state for a new task.
 */
function createNewTaskState(typeData) {
    return {
        typeData,
        stage: 0 /* TaskStateStage.Triage */,
        immutability: Immutability.Calculating,
    };
}
/**
 * Create the state for a new task that reduces the children task states.
 */
function createChildrenReducerTaskState(parent, children, childrenReducer) {
    return {
        typeData: parent.typeData,
        children,
        childrenReducer,
        stage: 1 /* TaskStateStage.ReduceChildren */,
        immutability: Immutability.Calculating,
    };
}
/**
 * Create the state for a new task that checks if the previous task has found
 * the type's immutability. If it hasn't the given action is called.
 */
function createCheckDoneTaskState(taskState, notDoneAction) {
    return {
        taskState,
        notDoneAction,
        stage: 5 /* TaskStateStage.CheckDone */,
    };
}
/**
 * Create the state for a new task that applies an override if the from
 * immutability check matches.
 */
function createApplyOverrideTaskState(taskState, override) {
    return {
        taskState,
        override,
        stage: 6 /* TaskStateStage.ApplyOverride */,
    };
}
/**
 * Get the override for the type if it has one.
 */
function getOverride(parameters, typeData) {
    return parameters.overrides.find((potentialOverride) => typeDataMatchesSpecifier(parameters.program, potentialOverride.type, typeData, parameters.typeMatchesPatternSpecifier));
}
/**
 * Process the state and create any new next task that need to be used to process it.
 */
function taskTriage(parameters, m_stack, m_state) {
    const cached = getCachedData(parameters.program, parameters.cache, m_state.typeData);
    if (cached !== undefined) {
        m_state.immutability = cached;
        return;
    }
    const override = getOverride(parameters, m_state.typeData);
    if (override?.to !== undefined) {
        // Early escape if we don't need to check the override from.
        if (override.from === undefined) {
            m_state.immutability = override.to;
            cacheData(parameters.program, parameters.cache, m_state.typeData, m_state.immutability);
            return;
        }
        m_stack.push(createApplyOverrideTaskState(m_state, override));
    }
    cacheData(parameters.program, parameters.cache, m_state.typeData, m_state.immutability);
    if (isUnionType(m_state.typeData.type)) {
        handleTypeUnion(m_stack, m_state);
        return;
    }
    if (isIntersectionType(m_state.typeData.type)) {
        handleTypeIntersection(parameters, m_stack, m_state);
        return;
    }
    if (isConditionalType(m_state.typeData.type)) {
        handleTypeConditional(parameters, m_stack, m_state);
        return;
    }
    if (isFunction(m_state.typeData.type)) {
        handleTypeFunction(m_state);
        return;
    }
    const checker = parameters.program.getTypeChecker();
    if (checker.isTupleType(m_state.typeData.type)) {
        handleTypeTuple(parameters, m_stack, m_state);
        return;
    }
    if (checker.isArrayType(m_state.typeData.type)) {
        handleTypeArray(parameters, m_stack, m_state);
        return;
    }
    if (isObjectType(m_state.typeData.type)) {
        handleTypeObject(parameters, m_stack, m_state);
        return;
    }
    // Must be a primitive.
    handleTypePrimitive(m_state);
}
/**
 * We know we're dealling with a TypeReference, check its type arguments.
 * If we're not done, move on to the ObjectIndexSignature task.
 */
function taskObjectTypeReference(m_stack, m_state) {
    m_stack.push(createCheckDoneTaskState(m_state, () => {
        m_state.stage = 4 /* TaskStateStage.ObjectIndexSignature */;
        m_stack.push(m_state);
    }));
    handleTypeArguments(m_stack, m_state);
}
/**
 * We know we're dealling with an object, check its index signatures.
 */
function taskObjectIndexSignature(parameters, m_stack, m_state) {
    const [types, typeNodes] = isIntersectionType(m_state.typeData.type)
        ? [
            m_state.typeData.type.types,
            m_state.typeData.typeNode?.types,
        ]
        : [
            [m_state.typeData.type],
            m_state.typeData.typeNode === null
                ? undefined
                : [m_state.typeData.typeNode],
        ];
    m_stack.push(createCheckDoneTaskState(m_state, () => {
        m_state.stage = 7 /* TaskStateStage.Done */;
        m_state.immutability = max(m_state.limits.min, m_state.limits.max);
        m_stack.push(m_state);
    }), createCheckDoneTaskState(m_state, () => {
        const children = types.flatMap((type, index) => createIndexSignatureTaskStates(parameters, m_state, ts.IndexKind.Number, getTypeData(type, typeNodes?.[index])));
        if (children.length > 0) {
            m_stack.push(createChildrenReducerTaskState(m_state, children, max), ...children);
        }
    }));
    const children = types.flatMap((type, index) => createIndexSignatureTaskStates(parameters, m_state, ts.IndexKind.String, getTypeData(type, typeNodes?.[index])));
    if (children.length > 0) {
        m_stack.push(createChildrenReducerTaskState(m_state, children, max), ...children);
    }
}
/**
 * Apply an override if its criteria are met.
 */
function taskApplyOverride(m_state) {
    if ((m_state.override.from <= m_state.taskState.immutability &&
        m_state.taskState.immutability <= m_state.override.to) ||
        (m_state.override.from >= m_state.taskState.immutability &&
            m_state.taskState.immutability >= m_state.override.to)) {
        m_state.taskState.immutability = m_state.override.to;
    }
}
/**
 * Check if we're found the type's immutability.
 */
function taskCheckDone(m_state, immutability) {
    if (immutability !== Immutability.Calculating) {
        m_state.taskState.limits.max = min(m_state.taskState.limits.max, immutability);
        if (m_state.taskState.limits.min >= m_state.taskState.limits.max) {
            m_state.taskState.immutability = m_state.taskState.limits.min;
            return;
        }
    }
    m_state.notDoneAction();
}
/**
 * Reduce the children's immutability values to a single value.
 */
function taskReduceChildren(m_state) {
    m_state.immutability = (m_state.children[0]).immutability;
    for (let m_index = 1; m_index < m_state.children.length; m_index++) {
        m_state.immutability = m_state.childrenReducer(m_state.immutability, m_state.children[m_index].immutability);
    }
}
/**
 * Handle a type we know is a union.
 */
function handleTypeUnion(m_stack, m_state) {
    const children = m_state.typeData.type.types.map((type, index) => {
        const typeNode = m_state.typeData.typeNode !== null &&
            ts.isUnionTypeNode(m_state.typeData.typeNode)
            ? m_state.typeData.typeNode.types[index]
            : undefined; // TODO: can we safely get a union type node nested within a different type node?
        return createNewTaskState(getTypeData(type, typeNode));
    });
    m_stack.push(createChildrenReducerTaskState(m_state, children, min), ...children);
}
/**
 * Handle a type we know is an intersection.
 */
function handleTypeIntersection(parameters, m_stack, m_state) {
    handleTypeObject(parameters, m_stack, m_state);
}
/**
 * Handle a type we know is a conditional type.
 */
function handleTypeConditional(parameters, m_stack, m_state) {
    const checker = parameters.program.getTypeChecker();
    const children = [
        m_state.typeData.type.root.node.trueType,
        m_state.typeData.type.root.node.falseType,
    ].map((typeNode) => {
        const type = checker.getTypeFromTypeNode(typeNode);
        return createNewTaskState(getTypeData(type, typeNode));
    });
    m_stack.push(createChildrenReducerTaskState(m_state, children, min), ...children);
}
/**
 * Handle a type we know is a non-namespace function.
 */
function handleTypeFunction(m_state) {
    m_state.immutability = Immutability.Immutable;
}
/**
 * Handle a type we know is a tuple.
 */
function handleTypeTuple(parameters, m_stack, m_state) {
    if (!m_state.typeData.type.target.readonly) {
        m_state.immutability = Immutability.Mutable;
        return;
    }
    handleTypeArray(parameters, m_stack, m_state);
}
/**
 * Handle a type we know is an array (this includes tuples).
 */
function handleTypeArray(parameters, m_stack, m_state) {
    // It will have limits after being processed by `handleTypeObject`.
    const m_stateWithLimits = m_state;
    m_stack.push(createCheckDoneTaskState(m_stateWithLimits, () => {
        m_stateWithLimits.stage = 7 /* TaskStateStage.Done */;
        m_stateWithLimits.immutability = max(m_stateWithLimits.limits.min, m_stateWithLimits.limits.max);
        m_stack.push(m_stateWithLimits);
    }), createCheckDoneTaskState(m_stateWithLimits, () => {
        if (isTypeReferenceWithTypeArguments(m_stateWithLimits.typeData.type)) {
            handleTypeArguments(m_stack, m_stateWithLimits);
        }
    }));
    handleTypeObject(parameters, m_stack, m_state);
}
/**
 * Handle a type that all we know is that it's an object.
 */
function handleTypeObject(parameters, m_stack, m_state) {
    // Add limits.
    const m_stateWithLimits = m_state;
    m_stateWithLimits.stage = 2 /* TaskStateStage.ObjectProperties */;
    m_stateWithLimits.limits = {
        ...parameters.immutabilityLimits,
    };
    m_stack.push(createCheckDoneTaskState(m_stateWithLimits, () => {
        if (isTypeReferenceWithTypeArguments(m_stateWithLimits.typeData.type)) {
            m_stateWithLimits.stage = 3 /* TaskStateStage.ObjectTypeReference */;
            m_stack.push(m_stateWithLimits);
            return;
        }
        m_stateWithLimits.stage = 4 /* TaskStateStage.ObjectIndexSignature */;
        m_stack.push(m_stateWithLimits);
    }));
    const checker = parameters.program.getTypeChecker();
    const properties = m_stateWithLimits.typeData.type.getProperties();
    if (properties.length > 0) {
        for (const property of properties) {
            if (isPropertyReadonlyInType(m_stateWithLimits.typeData.type, property.getEscapedName(), checker) ||
                // Ignore "length" for tuples.
                // TODO: Report this issue to upstream.
                (property.escapedName === "length" &&
                    checker.isTupleType(m_stateWithLimits.typeData.type))) {
                continue;
            }
            const name = ts.getNameOfDeclaration(property.valueDeclaration);
            if (name !== undefined && ts.isPrivateIdentifier(name)) {
                continue;
            }
            const declarations = property.getDeclarations() ?? [];
            if (declarations.length > 0) {
                if (declarations.some((declaration) => hasSymbol(declaration) &&
                    isSymbolFlagSet(declaration.symbol, ts.SymbolFlags.Method))) {
                    m_stateWithLimits.limits.max = min(m_stateWithLimits.limits.max, Immutability.ReadonlyDeep);
                    continue;
                }
                if (declarations.every((declaration) => ts.isPropertySignature(declaration) &&
                    declaration.type !== undefined &&
                    ts.isFunctionTypeNode(declaration.type))) {
                    m_stateWithLimits.limits.max = min(m_stateWithLimits.limits.max, Immutability.ReadonlyDeep);
                    continue;
                }
            }
            m_stateWithLimits.immutability = Immutability.Mutable;
            return;
        }
    }
    const propertyNodes = new Map(m_stateWithLimits.typeData.typeNode !== null &&
        hasType(m_stateWithLimits.typeData.typeNode) &&
        m_stateWithLimits.typeData.typeNode.type !== undefined &&
        ts.isTypeLiteralNode(m_stateWithLimits.typeData.typeNode.type)
        ? m_stateWithLimits.typeData.typeNode.type.members
            .map((member) => member.name === undefined ||
            !hasType(member) ||
            member.type === undefined
            ? undefined
            : [propertyNameToString(member.name), member.type])
            .filter((v) => v !== undefined)
        : []);
    const children = properties
        .map((property) => {
        const propertyType = getTypeOfPropertyOfType(checker, m_stateWithLimits.typeData.type, property);
        if (propertyType === undefined ||
            (isIntrinsicType(propertyType) &&
                propertyType.intrinsicName === "error")) {
            return null;
        }
        const propertyTypeNode = propertyNodes.get(property.getEscapedName());
        return createNewTaskState(getTypeData(propertyType, propertyTypeNode));
    })
        .filter((taskState) => taskState !== null);
    if (children.length > 0) {
        m_stateWithLimits.limits.min = Immutability.ReadonlyShallow;
        m_stack.push(createChildrenReducerTaskState(m_stateWithLimits, children, min), ...children);
    }
}
/**
 * Handle the type arguments of a type reference.
 */
function handleTypeArguments(m_stack, m_state) {
    const children = m_state.typeData.type.typeArguments.map((type) => createNewTaskState(getTypeData(type, undefined)));
    m_stack.push(createChildrenReducerTaskState(m_state, children, min), ...children);
}
/**
 * Handle a primitive type.
 */
function handleTypePrimitive(m_state) {
    m_state.immutability = Immutability.Immutable;
}
/**
 * Create the task states for analyzing an object's index signatures.
 */
function createIndexSignatureTaskStates(parameters, m_state, kind, typeData) {
    const checker = parameters.program.getTypeChecker();
    const indexInfo = checker.getIndexInfoOfType(typeData.type, kind);
    if (indexInfo === undefined) {
        m_state.immutability = Immutability.Unknown;
        return [];
    }
    if (parameters.immutabilityLimits.max <= Immutability.ReadonlyShallow) {
        m_state.immutability = Immutability.ReadonlyShallow;
        return [];
    }
    if (indexInfo.isReadonly) {
        if (indexInfo.type === typeData.type) {
            m_state.immutability = parameters.immutabilityLimits.max;
            return [];
        }
        const child = createNewTaskState(getTypeData(indexInfo.type, undefined));
        return [
            createChildrenReducerTaskState(m_state, [{ immutability: Immutability.ReadonlyShallow }, child], max),
            child,
        ];
    }
    m_state.immutability = Immutability.Mutable;
    return [];
}

/**
 * Is the immutability of the given type immutable.
 *
 * @param program - The TypeScript Program to use.
 * @param typeOrTypeNode - The type to test the immutability of.
 * @param overrides - The overrides to use when calculating the immutability.
 * @param useCache - Either a custom cache to use, `true` to use the global
 * cache, or `false` to not use any predefined cache.
 */
function isImmutableType(program, typeOrTypeNode, overrides = getDefaultOverrides(), useCache = true) {
    const immutability = getTypeImmutability(program, typeOrTypeNode, overrides, useCache, Immutability.Immutable);
    return isImmutable(immutability);
}
/**
 * Is the immutability of the given type at least readonly deep.
 *
 * @param program - The TypeScript Program to use.
 * @param typeOrTypeNode - The type to test the immutability of.
 * @param overrides - The overrides to use when calculating the immutability.
 * @param useCache - Either a custom cache to use, `true` to use the global
 * cache, or `false` to not use any predefined cache.
 */
function isReadonlyDeepType(program, typeOrTypeNode, overrides = getDefaultOverrides(), useCache = true) {
    const immutability = getTypeImmutability(program, typeOrTypeNode, overrides, useCache, Immutability.ReadonlyDeep);
    return isReadonlyDeep(immutability);
}
/**
 * Is the immutability of the given type at least readonly shallow.
 *
 * @param program - The TypeScript Program to use.
 * @param typeOrTypeNode - The type to test the immutability of.
 * @param overrides - The overrides to use when calculating the immutability.
 * @param useCache - Either a custom cache to use, `true` to use the global
 * cache, or `false` to not use any predefined cache.
 */
function isReadonlyShallowType(program, typeOrTypeNode, overrides = getDefaultOverrides(), useCache = true) {
    const immutability = getTypeImmutability(program, typeOrTypeNode, overrides, useCache, Immutability.ReadonlyShallow);
    return isReadonlyShallow(immutability);
}
/**
 * Is the immutability of the given type mutable.
 *
 * @param program - The TypeScript Program to use.
 * @param typeOrTypeNode - The type to test the immutability of.
 * @param overrides - The overrides to use when calculating the immutability.
 * @param useCache - Either a custom cache to use, `true` to use the global
 * cache, or `false` to not use any predefined cache.
 */
function isMutableType(program, typeOrTypeNode, overrides = getDefaultOverrides(), useCache = true) {
    const immutability = getTypeImmutability(program, typeOrTypeNode, overrides, useCache, Immutability.Mutable);
    return isMutable(immutability);
}

export { Immutability, clamp, getDefaultOverrides, getTypeImmutability, isImmutable, isImmutableType, isMutable, isMutableType, isReadonlyDeep, isReadonlyDeepType, isReadonlyShallow, isReadonlyShallowType, isUnknown, max, min };
