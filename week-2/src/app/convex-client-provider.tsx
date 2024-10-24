'use client';

import React from 'react'
import { ConvexReactClient } from 'convex/react'
import { ConvexProviderWithClerk } from 'convex/react-clerk'
import { ClerkLoaded, ClerkLoading, ClerkProvider, useAuth } from '@clerk/nextjs'
import { MultisessionAppSupport } from "@clerk/clerk-react/internal";

const convexClient = new ConvexReactClient(process.env.NEXT_PUBLIC_CONVEX_URL! as string);

export const ConvexClientProvider = ({ children }: {
  children: React.ReactNode;
}) => {
  return (
    <ClerkProvider
      publishableKey={process.env.NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY}
      signInUrl="/sign-in"
      signUpUrl="/sign-up"
    >
      <MultisessionAppSupport>
        <ConvexProviderWithClerk client={convexClient} useAuth={useAuth}>
          <ClerkLoading>
            <Loader />
          </ClerkLoading>
          <ClerkLoaded>
            {children}
          </ClerkLoaded>
        </ConvexProviderWithClerk>
      </MultisessionAppSupport>
    </ClerkProvider>
  );
};

const Loader = () => {
  return <div>Loading...</div>
}
