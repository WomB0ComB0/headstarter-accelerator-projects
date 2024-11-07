'use client';

import React from 'react'
import { ConvexReactClient } from 'convex/react'
import { ConvexProviderWithClerk } from 'convex/react-clerk'
import { ClerkProvider, RedirectToSignIn, useAuth } from '@clerk/nextjs'
import { MultisessionAppSupport } from "@clerk/clerk-react/internal";
import { Authenticated, Unauthenticated, AuthLoading } from 'convex/react'; 

const convexClient = new ConvexReactClient(process.env.NEXT_PUBLIC_CONVEX_URL! as string);

export const ConvexClientProvider = ({ children }: {
  children: React.ReactNode;
}) => {
  return (
    <ClerkProvider
      publishableKey={process.env.NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY}
      signInUrl="/sign-in"
      signUpUrl="/sign-up"
      // dynamic
    >
      <MultisessionAppSupport>
        <ConvexProviderWithClerk client={convexClient} useAuth={useAuth}>
          <AuthLoading>
            <Loader />
          </AuthLoading>
          <Authenticated>
            {children}
          </Authenticated>
          <Unauthenticated>
            <RedirectToSignIn />
          </Unauthenticated>
        </ConvexProviderWithClerk>
      </MultisessionAppSupport>
    </ClerkProvider>
  );
};

const Loader = () => {
  return <div>Loading...</div>
}
