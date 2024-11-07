"use client";

import { Button } from "@/components/ui/button";
import {
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { api } from "@/api";
import { Dialog } from "@radix-ui/react-dialog";
import { useMutation } from "convex/react";
import { useState } from "react";
import { toast } from "sonner";
import { catchError } from "@/lib/utils";
export function AddFriend() {
  const [open, setOpen] = useState(false);
  const createFriendRequest = useMutation(api.functions.friend.createFriendRequest);

  const handleSubmit = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    const username = e.currentTarget.username.value;

    const [error] = await catchError(
      createFriendRequest({ username })
    );

    if (error) {
      toast.error("Failed to send friend request", {
        description: error instanceof Error ? error.message : "An unknown error occurred",
      });
      return;
    }

    toast.success("Friend request sent");
    setOpen(false);
  };

  return (
    <Dialog open={open} onOpenChange={setOpen}>
      <DialogTrigger asChild>
        <Button size="sm">Add Friend</Button>
      </DialogTrigger>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>Add Friend</DialogTitle>
          <DialogDescription>
            You can add a friend by their username.
          </DialogDescription>
        </DialogHeader>
        <form className="contents" onSubmit={handleSubmit}>
          <div className="flex flex-col gap-1">
            <Label htmlFor="username">Username</Label>
            <Input id="username" type="text" required />
          </div>
          <DialogFooter>
            <Button>Send Friend Request</Button>
          </DialogFooter>
        </form>
      </DialogContent>
    </Dialog>
  );
}
