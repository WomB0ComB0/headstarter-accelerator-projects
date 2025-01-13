import { L as LiteralString } from '../helper-Bi8FQwDD.cjs';
import 'zod';

type SubArray<T extends unknown[] | readonly unknown[] | any[]> = T[number][];
type Subset<K extends keyof R, R extends Record<string | LiteralString, readonly string[] | readonly LiteralString[]>> = {
    [P in K]: SubArray<R[P]>;
};
type StatementsPrimitive = {
    readonly [resource: string]: readonly LiteralString[];
};

type Connector = "OR" | "AND";
type AuthorizeResponse = {
    success: false;
    error: string;
} | {
    success: true;
    error?: undefined;
};
declare function createAccessControl<TStatements extends StatementsPrimitive = StatementsPrimitive>(s: TStatements): {
    newRole<K extends keyof TStatements>(statements: Subset<K, TStatements>): {
        statements: Subset<K, TStatements>;
        authorize<K_1 extends K>(request: Subset<K_1, Subset<K, TStatements>>, connector?: Connector): AuthorizeResponse;
    };
};
declare function role<TStatements extends StatementsPrimitive>(statements: TStatements): {
    statements: TStatements;
    authorize<K extends keyof TStatements>(request: Subset<K, TStatements>, connector?: Connector): AuthorizeResponse;
};
type AccessControl<TStatements extends StatementsPrimitive = StatementsPrimitive> = ReturnType<typeof createAccessControl<TStatements>>;
type Role<TStatements extends StatementsPrimitive = Record<string, any>> = {
    authorize: (request: any, connector?: Connector) => AuthorizeResponse;
    statements: TStatements;
};

declare const defaultStatements: {
    readonly organization: readonly ["update", "delete"];
    readonly member: readonly ["create", "update", "delete"];
    readonly invitation: readonly ["create", "cancel"];
};
declare const defaultAc: {
    newRole<K extends "organization" | "member" | "invitation">(statements: Subset<K, {
        readonly organization: readonly ["update", "delete"];
        readonly member: readonly ["create", "update", "delete"];
        readonly invitation: readonly ["create", "cancel"];
    }>): {
        statements: Subset<K, {
            readonly organization: readonly ["update", "delete"];
            readonly member: readonly ["create", "update", "delete"];
            readonly invitation: readonly ["create", "cancel"];
        }>;
        authorize<K_1 extends K>(request: Subset<K_1, Subset<K, {
            readonly organization: readonly ["update", "delete"];
            readonly member: readonly ["create", "update", "delete"];
            readonly invitation: readonly ["create", "cancel"];
        }>>, connector?: "AND" | "OR"): {
            success: false;
            error: string;
        } | {
            success: true;
            error?: undefined;
        };
    };
};
declare const adminAc: {
    statements: Subset<"organization" | "member" | "invitation", {
        readonly organization: readonly ["update", "delete"];
        readonly member: readonly ["create", "update", "delete"];
        readonly invitation: readonly ["create", "cancel"];
    }>;
    authorize<K extends "organization" | "member" | "invitation">(request: Subset<K, Subset<"organization" | "member" | "invitation", {
        readonly organization: readonly ["update", "delete"];
        readonly member: readonly ["create", "update", "delete"];
        readonly invitation: readonly ["create", "cancel"];
    }>>, connector?: "AND" | "OR"): {
        success: false;
        error: string;
    } | {
        success: true;
        error?: undefined;
    };
};
declare const ownerAc: {
    statements: Subset<"organization" | "member" | "invitation", {
        readonly organization: readonly ["update", "delete"];
        readonly member: readonly ["create", "update", "delete"];
        readonly invitation: readonly ["create", "cancel"];
    }>;
    authorize<K extends "organization" | "member" | "invitation">(request: Subset<K, Subset<"organization" | "member" | "invitation", {
        readonly organization: readonly ["update", "delete"];
        readonly member: readonly ["create", "update", "delete"];
        readonly invitation: readonly ["create", "cancel"];
    }>>, connector?: "AND" | "OR"): {
        success: false;
        error: string;
    } | {
        success: true;
        error?: undefined;
    };
};
declare const memberAc: {
    statements: Subset<"organization" | "member" | "invitation", {
        readonly organization: readonly ["update", "delete"];
        readonly member: readonly ["create", "update", "delete"];
        readonly invitation: readonly ["create", "cancel"];
    }>;
    authorize<K extends "organization" | "member" | "invitation">(request: Subset<K, Subset<"organization" | "member" | "invitation", {
        readonly organization: readonly ["update", "delete"];
        readonly member: readonly ["create", "update", "delete"];
        readonly invitation: readonly ["create", "cancel"];
    }>>, connector?: "AND" | "OR"): {
        success: false;
        error: string;
    } | {
        success: true;
        error?: undefined;
    };
};
declare const defaultRoles: {
    admin: {
        statements: Subset<"organization" | "member" | "invitation", {
            readonly organization: readonly ["update", "delete"];
            readonly member: readonly ["create", "update", "delete"];
            readonly invitation: readonly ["create", "cancel"];
        }>;
        authorize<K extends "organization" | "member" | "invitation">(request: Subset<K, Subset<"organization" | "member" | "invitation", {
            readonly organization: readonly ["update", "delete"];
            readonly member: readonly ["create", "update", "delete"];
            readonly invitation: readonly ["create", "cancel"];
        }>>, connector?: "AND" | "OR"): {
            success: false;
            error: string;
        } | {
            success: true;
            error?: undefined;
        };
    };
    owner: {
        statements: Subset<"organization" | "member" | "invitation", {
            readonly organization: readonly ["update", "delete"];
            readonly member: readonly ["create", "update", "delete"];
            readonly invitation: readonly ["create", "cancel"];
        }>;
        authorize<K extends "organization" | "member" | "invitation">(request: Subset<K, Subset<"organization" | "member" | "invitation", {
            readonly organization: readonly ["update", "delete"];
            readonly member: readonly ["create", "update", "delete"];
            readonly invitation: readonly ["create", "cancel"];
        }>>, connector?: "AND" | "OR"): {
            success: false;
            error: string;
        } | {
            success: true;
            error?: undefined;
        };
    };
    member: {
        statements: Subset<"organization" | "member" | "invitation", {
            readonly organization: readonly ["update", "delete"];
            readonly member: readonly ["create", "update", "delete"];
            readonly invitation: readonly ["create", "cancel"];
        }>;
        authorize<K extends "organization" | "member" | "invitation">(request: Subset<K, Subset<"organization" | "member" | "invitation", {
            readonly organization: readonly ["update", "delete"];
            readonly member: readonly ["create", "update", "delete"];
            readonly invitation: readonly ["create", "cancel"];
        }>>, connector?: "AND" | "OR"): {
            success: false;
            error: string;
        } | {
            success: true;
            error?: undefined;
        };
    };
};

export { type AccessControl, type Role, type StatementsPrimitive, type SubArray, type Subset, adminAc, createAccessControl, defaultAc, defaultRoles, defaultStatements, memberAc, ownerAc, role };
