"use client";

import Link from "next/link";
import { useUser, SignInButton, UserButton } from "@clerk/nextjs";
import { Home, Compass, Upload } from "lucide-react";

export function Navigation() {
  const { isSignedIn, user } = useUser();

  return (
    <div className="fixed left-0 top-0 h-full w-[240px] bg-black border-r border-gray-800 p-4">
      <div className="flex flex-col h-full">
        <div className="mb-8">
          <h1 className="text-xl font-bold text-white">TikTok Clone</h1>
        </div>

        <nav className="flex-1">
          <Link href="/" className="flex items-center p-4 hover:bg-gray-900 rounded-lg mb-2 text-white">
            <Home className="w-5 h-5 mr-2" />
            For You
          </Link>
          <Link href="/following" className="flex items-center p-4 hover:bg-gray-900 rounded-lg mb-2 text-white">
            <Compass className="w-5 h-5 mr-2" />
            Following
          </Link>
          {isSignedIn && (
            <Link href="/upload" className="flex items-center p-4 hover:bg-gray-900 rounded-lg mb-2 text-white">
              <Upload className="w-5 h-5 mr-2" />
              Upload
            </Link>
          )}
        </nav>

        <div className="mt-auto p-4">
          {isSignedIn ? (
            <div className="flex items-center gap-2">
              <UserButton afterSignOutUrl="/" />
              <span className="text-white">{user.username}</span>
            </div>
          ) : (
            <SignInButton mode="modal">
              <button className="w-full bg-[#F02C56] text-white rounded-lg py-2.5 font-semibold hover:bg-[#E61F4A] transition">
                Sign In
              </button>
            </SignInButton>
          )}
        </div>
      </div>
    </div>
  );
} 