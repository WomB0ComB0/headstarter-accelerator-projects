import { createAPIFileRoute } from "@tanstack/start/api";
import { auth } from "~/lib/middleware/server/auth";

export const APIRoute = createAPIFileRoute("/api/auth/$")({
  GET: ({ request }) => {
    return auth.handler(request);
  },
  POST: ({ request }) => {
    return auth.handler(request);
  },
});
