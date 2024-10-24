import { NextRequest, NextResponse } from 'next/server';
import { clerkMiddleware, createRouteMatcher } from "@clerk/nextjs/server";
import { useAuth } from '@clerk/nextjs';

const isProtectedRoute = createRouteMatcher(["/api(.*)"]);

export default async function middleware(req: NextRequest) {
  const auth = useAuth();
  clerkMiddleware((auth, req) => {
    if (isProtectedRoute(req)) auth.protect();
  });

  return NextResponse.next();
}

export const config = {
  matcher: [
    '/((?!_next|[^?]*\\.(?:html?|css|js(?!on)|jpe?g|webp|png|gif|svg|ttf|woff2?|ico|csv|docx?|xlsx?|zip|webmanifest|json)).*)',
    '/(api|trpc)(.*)',
  ],
}
