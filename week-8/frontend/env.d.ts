declare global {
  namespace NodeJS {
    interface ProcessEnv {
      NODE_ENV: 'development' | 'production' | 'test';
      NEXT_PUBLIC_UPSTASH_REDIS_REST_URL: string;
      NEXT_PUBLIC_UPSTASH_REDIS_REST_TOKEN: string;
      NEXT_PUBLIC_API_KEY: string;
      NEXT_PUBLIC_SENTRY_DSN: string;
      NEXT_PUBLIC_SENTRY_AUTH_TOKEN: string;
      NEXT_PUBLIC_BLOB_READ_WRITE_TOKEN: string;
      NEXT_PUBLIC_SENTRY_SUPPRESS_TURBOPACK_WARNING: number;
    }
  }
}

export { }