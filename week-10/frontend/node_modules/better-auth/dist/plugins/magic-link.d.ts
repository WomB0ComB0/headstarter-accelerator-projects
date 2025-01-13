import * as better_call from 'better-call';
import { z } from 'zod';

interface MagicLinkOptions {
    /**
     * Time in seconds until the magic link expires.
     * @default (60 * 5) // 5 minutes
     */
    expiresIn?: number;
    /**
     * Send magic link implementation.
     */
    sendMagicLink: (data: {
        email: string;
        url: string;
        token: string;
    }, request?: Request) => Promise<void> | void;
    /**
     * Disable sign up if user is not found.
     *
     * @default false
     */
    disableSignUp?: boolean;
    /**
     * Rate limit configuration.
     *
     * @default {
     *  window: 60,
     *  max: 5,
     * }
     */
    rateLimit?: {
        window: number;
        max: number;
    };
}
declare const magicLink: (options: MagicLinkOptions) => {
    id: "magic-link";
    endpoints: {
        signInMagicLink: {
            <C extends [better_call.Context<"/sign-in/magic-link", {
                method: "POST";
                requireHeaders: true;
                body: z.ZodObject<{
                    email: z.ZodString;
                    callbackURL: z.ZodOptional<z.ZodString>;
                }, "strip", z.ZodTypeAny, {
                    email: string;
                    callbackURL?: string | undefined;
                }, {
                    email: string;
                    callbackURL?: string | undefined;
                }>;
                metadata: {
                    openapi: {
                        description: string;
                        responses: {
                            200: {
                                description: string;
                                content: {
                                    "application/json": {
                                        schema: {
                                            type: "object";
                                            properties: {
                                                status: {
                                                    type: string;
                                                };
                                            };
                                        };
                                    };
                                };
                            };
                        };
                    };
                };
            }>]>(...ctx: C): Promise<C extends [{
                asResponse: true;
            }] ? Response : {
                status: boolean;
            }>;
            path: "/sign-in/magic-link";
            options: {
                method: "POST";
                requireHeaders: true;
                body: z.ZodObject<{
                    email: z.ZodString;
                    callbackURL: z.ZodOptional<z.ZodString>;
                }, "strip", z.ZodTypeAny, {
                    email: string;
                    callbackURL?: string | undefined;
                }, {
                    email: string;
                    callbackURL?: string | undefined;
                }>;
                metadata: {
                    openapi: {
                        description: string;
                        responses: {
                            200: {
                                description: string;
                                content: {
                                    "application/json": {
                                        schema: {
                                            type: "object";
                                            properties: {
                                                status: {
                                                    type: string;
                                                };
                                            };
                                        };
                                    };
                                };
                            };
                        };
                    };
                };
            };
            method: better_call.Method | better_call.Method[];
            headers: Headers;
        };
        magicLinkVerify: {
            <C extends [better_call.Context<"/magic-link/verify", {
                method: "GET";
                query: z.ZodObject<{
                    token: z.ZodString;
                    callbackURL: z.ZodOptional<z.ZodString>;
                }, "strip", z.ZodTypeAny, {
                    token: string;
                    callbackURL?: string | undefined;
                }, {
                    token: string;
                    callbackURL?: string | undefined;
                }>;
                use: better_call.Endpoint<better_call.Handler<string, better_call.EndpointOptions, void>, better_call.EndpointOptions>[];
                requireHeaders: true;
                metadata: {
                    openapi: {
                        description: string;
                        responses: {
                            200: {
                                description: string;
                                content: {
                                    "application/json": {
                                        schema: {
                                            type: "object";
                                            properties: {
                                                session: {
                                                    $ref: string;
                                                };
                                                user: {
                                                    $ref: string;
                                                };
                                            };
                                        };
                                    };
                                };
                            };
                        };
                    };
                };
            }>]>(...ctx: C): Promise<C extends [{
                asResponse: true;
            }] ? Response : {
                token: string;
                user: {
                    id: string;
                    email: string;
                    emailVerified: boolean;
                    name: string;
                    image: string | null | undefined;
                    createdAt: Date;
                    updatedAt: Date;
                };
            }>;
            path: "/magic-link/verify";
            options: {
                method: "GET";
                query: z.ZodObject<{
                    token: z.ZodString;
                    callbackURL: z.ZodOptional<z.ZodString>;
                }, "strip", z.ZodTypeAny, {
                    token: string;
                    callbackURL?: string | undefined;
                }, {
                    token: string;
                    callbackURL?: string | undefined;
                }>;
                use: better_call.Endpoint<better_call.Handler<string, better_call.EndpointOptions, void>, better_call.EndpointOptions>[];
                requireHeaders: true;
                metadata: {
                    openapi: {
                        description: string;
                        responses: {
                            200: {
                                description: string;
                                content: {
                                    "application/json": {
                                        schema: {
                                            type: "object";
                                            properties: {
                                                session: {
                                                    $ref: string;
                                                };
                                                user: {
                                                    $ref: string;
                                                };
                                            };
                                        };
                                    };
                                };
                            };
                        };
                    };
                };
            };
            method: better_call.Method | better_call.Method[];
            headers: Headers;
        };
    };
    rateLimit: {
        pathMatcher(path: string): boolean;
        window: number;
        max: number;
    }[];
};

export { magicLink };
