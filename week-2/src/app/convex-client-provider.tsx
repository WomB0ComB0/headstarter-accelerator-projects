'use client';

import React from 'react'
import { ConvexReactClient } from 'convex/react'
import { ConvexProviderWithClerk } from 'convex/react-clerk'
import { useAuth } from '@clerk/nextjs'
import { Authenticated, Unauthenticated } from 'convex/react'
import { SignOutButton, RedirectToSignIn } from '@clerk/nextjs'
const convexClient = new ConvexReactClient(process.env.NEXT_PUBLIC_CONVEX_URL! as string);

export const ConvexClientProvider = ({ children }: {
  children: React.ReactNode;
}) => {
  return (
    <ConvexProviderWithClerk client={convexClient} useAuth={useAuth}>
      <Authenticated>
        {children}
      </Authenticated>
      <Unauthenticated>
        <RedirectToSignIn />
      </Unauthenticated>
    </ConvexProviderWithClerk>
  );
};
