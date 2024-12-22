declare global {
  namespace NodeJS {
    interface ProcessEnv {
      NODE_ENV: 'development' | 'production' | 'test';
      UPSTASH_REDIS_REST_URL: string;
      UPSTASH_REDIS_REST_TOKEN: string;
      API_KEY: string;
      SENTRY_DSN: string;
      SENTRY_AUTH_TOKEN: string;
      BLOB_READ_WRITE_TOKEN: string;
      SENTRY_SUPPRESS_TURBOPACK_WARNING: number;
    }
  }
}

export { }