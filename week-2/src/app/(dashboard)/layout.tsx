'use client'

import React from 'react'
import { Authenticated, Unauthenticated } from 'convex/react'
import { SignOutButton, RedirectToSignIn } from '@clerk/nextjs'
import { User2Icon, PlusIcon } from 'lucide-react'
import { Sidebar, SidebarContent, SidebarFooter, SidebarGroup, SidebarGroupAction, SidebarGroupContent, SidebarGroupLabel, SidebarMenu, SidebarMenuButton, SidebarMenuItem, SidebarProvider } from '@/components/ui/sidebar'
import Link from 'next/link'
import { useQuery } from 'convex/react'
import { api } from '../../../convex/_generated/api'
import { Avatar, AvatarFallback, AvatarImage } from '@/components/ui/avatar'
import { DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuTrigger } from '@/components/ui/dropdown-menu'
import { useRouter } from 'next/navigation'

export default function DashboardLayout({ children }: { children: React.ReactNode }) {
  const router = useRouter()

  return (
    <>
      <SidebarProvider>
        <div className="flex">
          <DashboardSidebar />
          <main className="flex-1">
            {children}
          </main>
        </div>
      </SidebarProvider>
    </>
  )
}

function DashboardSidebar() {
  const user = useQuery(api.functions.user.get)

  if (user === undefined) return <div>Loading...</div>
  if (user === null) return <div>Error loading user data</div>

  return (
    <Sidebar>
      <SidebarContent>
        <SidebarGroup>
          <SidebarGroupContent>
            <SidebarMenu>
              <SidebarMenuItem>
                <SidebarMenuButton asChild>
                  <Link href="/friends">
                    <User2Icon className="w-4 h-4 mr-2" />
                    Friends
                  </Link>
                </SidebarMenuButton>
              </SidebarMenuItem>
            </SidebarMenu>
          </SidebarGroupContent>
        </SidebarGroup>
        <SidebarGroup>
          <SidebarGroupLabel>
            Direct Messages
          </SidebarGroupLabel>
          <SidebarGroupAction>
            <PlusIcon className="w-4 h-4" />
            <span className="sr-only">
              New Direct Message
            </span>
          </SidebarGroupAction>
        </SidebarGroup>
      </SidebarContent>
      <SidebarFooter>
        <SidebarGroup>
          <SidebarGroupContent>
            <SidebarMenu>
              <SidebarMenuItem>
                <DropdownMenu>
                  <DropdownMenuTrigger asChild>
                    <SidebarMenuButton className="flex items-center gap-2">
                      <Avatar className="size-6">
                        <AvatarImage src={user.image} />
                        <AvatarFallback>
                          {user.username[0]}
                        </AvatarFallback>
                      </Avatar>
                      <span className="text-sm font-medium">
                        {user.username}
                      </span>
                    </SidebarMenuButton>
                  </DropdownMenuTrigger>
                  <DropdownMenuContent>
                    <DropdownMenuItem>
                      <SignOutButton>Sign out</SignOutButton>
                    </DropdownMenuItem>
                  </DropdownMenuContent>
                </DropdownMenu>
              </SidebarMenuItem>
            </SidebarMenu>
          </SidebarGroupContent>
        </SidebarGroup>
      </SidebarFooter>
    </Sidebar>
  )
}
