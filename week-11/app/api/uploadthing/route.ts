import { createNextRouteHandler } from "uploadthing/next";
import { ourFileRouter } from "~/app/utils/uploadthing";

export const { GET, POST } = createNextRouteHandler({
  router: ourFileRouter,
}); 