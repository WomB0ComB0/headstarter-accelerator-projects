declare global {
  namespace NodeJS {
    interface ProcessEnv {
      NEXT_PUBLIC_GOOGLE_AI_API_KEY: string;
    }
  }
}
export { }