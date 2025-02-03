import { authMiddleware } from "@clerk/nextjs/server";
import { NextResponse } from "next/server";

export default authMiddleware({
  publicRoutes: ["/", "/api/uploadthing"],
  ignoredRoutes: ["/api/uploadthing"],
  beforeAuth: (req) => {
    // Your beforeAuth logic here
  },
});

export const config = {
  matcher: ["/((?!.+\\.[\\w]+$|_next).*)", "/", "/(api|trpc)(.*)"],
};
