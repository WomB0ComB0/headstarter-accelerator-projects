import pwa from '@ducanh2912/next-pwa';
import MillionLint from '@million/lint';
import {withSentryConfig} from "@sentry/nextjs";
import type { NextConfig } from "next";
import path from 'node:path';

const withPwa = pwa({
  dest: 'public',
});

const nextConfig: NextConfig = {
  reactStrictMode: true,
  pageExtensions: ['tsx', 'mdx', 'ts', 'js'],
  logging: {
    fetches: {
      fullUrl: true,
    }
  },
  images: {
    remotePatterns: [
      {
        protocol: 'https',
        hostname: '**.vercel-storage.com',
      },
    ],
  },
  experimental: {
    optimizeCss: true,
    serverActions: {
      allowedOrigins: ['https://*.vercel-storage.com', 'http://localhost:3000', process.env.NEXT_PUBLIC_APP_URL || ''],
      bodySizeLimit: '10mb',
    },
    turbo: process.env.NODE_ENV === 'development' ? undefined : {
      resolveAlias: {
        '@': path.resolve(__dirname, 'src'),
      },
      rules: {
        '**/*.{ts,tsx}': ['eslint'],
        '**/*.{js,jsx}': ['eslint'],
        '**/*.{json}': ['prettier --write'],
      }
    }
  },
  productionBrowserSourceMaps: true,
  headers: async () => [
    {
      source: '/:path*',
      headers: [
        { key: 'X-DNS-Prefetch-Control', value: 'on' },
        { key: 'X-XSS-Protection', value: '1; mode=block' },
        { key: 'X-Frame-Options', value: 'SAMEORIGIN' },
        { key: 'X-Content-Type-Options', value: 'nosniff' },
      ],
    },
    {
      source: '/_next/static/media/:path*',
      headers: [
        {
          key: 'Cache-Control',
          value: 'public, max-age=31536000, immutable',
        },
      ],
    },
    {
      source: '/api/:path*',
      headers: [
        { key: 'Access-Control-Allow-Credentials', value: 'true' },
        {
          key: 'Access-Control-Allow-Origin',
          value: process.env.NODE_ENV === 'production' ? 'https://www.kappathetapi.org' : '*',
        },
        { key: 'Access-Control-Allow-Methods', value: 'GET,DELETE,PATCH,POST,PUT,OPTIONS' },
        {
          key: 'Access-Control-Allow-Headers',
          value:
            'X-CSRF-Token, X-Requested-With, Accept, Accept-Version, Content-Length, Content-MD5, Content-Type, Date, X-Api-Version',
        },
      ],
      },
    {
      source: '/(.*).png',
      headers: [{ key: 'Content-Type', value: 'image/png' }],
    },
  ],
};

const millionConfig = MillionLint.next({
  rsc: true,
  filter: {
    include: '**/components/**/*.{mtsx,mjsx,tsx,jsx}',
    exclude: ['**/api/**/*.{ts,tsx}'],
  },
});

const sentryConfig = withSentryConfig(nextConfig, {
  org: 'womb0comb0',
  project: 'pentagram',
  authToken: process.env.NEXT_PUBLIC_SENTRY_AUTH_TOKEN,
  silent: true,
  sourcemaps: {
    assets: './**/*.{js,map}',
    ignore: ['node_modules/**/*'],
  },
  hideSourceMaps: true,
  widenClientFileUpload: true,
  autoInstrumentServerFunctions: true,
  autoInstrumentMiddleware: true,
  autoInstrumentAppDirectory: true,
  tunnelRoute: '/monitoring',
  disableLogger: true,
  automaticVercelMonitors: true,
  reactComponentAnnotation: {
    enabled: true,
  },
  bundleSizeOptimizations: {
    excludeDebugStatements: true,
    excludeReplayShadowDom: true,
    excludeReplayIframe: true,
    excludeReplayWorker: true,
  },
});

export default millionConfig(withPwa(sentryConfig));
