import { Ratelimit } from '@upstash/ratelimit';
import { Redis } from '@upstash/redis';
import { NextResponse } from 'next/server';
import type { NextRequest } from 'next/server';
import { getConfig } from './config';

const { redis: redisConfig } = getConfig('integration');

const redis = new Redis({
  url: redisConfig.url,
  token: redisConfig.token,
});

const ratelimit = new Ratelimit({
  redis,
  limiter: Ratelimit.slidingWindow(10, '60s'),
  analytics: true,
  prefix: '@upstash/ratelimit',
});

export async function middleware(request: NextRequest) {
  try {
    const ip =
      request.headers.get('x-forwarded-for') ??
      request.headers.get('cf-connecting-ip') ??
      request.headers.get('x-real-ip') ??
      '127.0.0.1';

    if (request.nextUrl.pathname.startsWith('/api')) {
      const { success, limit, reset, remaining } = await ratelimit.limit(ip);

      const response = success
        ? NextResponse.next()
        : NextResponse.json({ error: 'Too many requests' }, { status: 429 });

      response.headers.set('X-RateLimit-Limit', limit.toString());
      response.headers.set('X-RateLimit-Remaining', remaining.toString());
      response.headers.set('X-RateLimit-Reset', reset.toString());

      return response;
    }

    return NextResponse.next();
  } catch (error) {
    console.error('Rate limiting error:', error);
    return NextResponse.json({ error: 'Internal server error' }, { status: 500 });
  }
}

export const config = {
  matcher: [
    /*
     * Match all request paths except static files and images
     */
    '/((?!_next/static|_next/image|favicon.ico).*)',
  ],
};
