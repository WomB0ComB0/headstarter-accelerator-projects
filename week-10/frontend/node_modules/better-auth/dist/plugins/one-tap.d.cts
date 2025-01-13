import * as better_call from 'better-call';
import { z } from 'zod';

interface OneTapOptions {
    /**
     * Disable the signup flow
     *
     * @default false
     */
    disableSignup?: boolean;
}
declare const oneTap: (options?: OneTapOptions) => {
    id: "one-tap";
    endpoints: {
        oneTapCallback: {
            <C extends [better_call.Context<"/one-tap/callback", {
                method: "POST";
                body: z.ZodObject<{
                    idToken: z.ZodString;
                }, "strip", z.ZodTypeAny, {
                    idToken: string;
                }, {
                    idToken: string;
                }>;
                metadata: {
                    openapi: {
                        summary: string;
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
                            400: {
                                description: string;
                            };
                        };
                    };
                };
            }>]>(...ctx: C): Promise<C extends [{
                asResponse: true;
            }] ? Response : {
                error: string;
            } | {
                token: string;
                user: {
                    id: any;
                    email: any;
                    emailVerified: any;
                    name: any;
                    image: any;
                    createdAt: any;
                    updatedAt: any;
                };
            }>;
            path: "/one-tap/callback";
            options: {
                method: "POST";
                body: z.ZodObject<{
                    idToken: z.ZodString;
                }, "strip", z.ZodTypeAny, {
                    idToken: string;
                }, {
                    idToken: string;
                }>;
                metadata: {
                    openapi: {
                        summary: string;
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
                            400: {
                                description: string;
                            };
                        };
                    };
                };
            };
            method: better_call.Method | better_call.Method[];
            headers: Headers;
        };
    };
};

export { oneTap };
