'use client';

import { api } from "../../convex/_generated/api";
import { useMutation, useQuery, Authenticated, Unauthenticated } from "convex/react";
import { SignInButton, SignOutButton } from "@clerk/nextjs";

interface Message {
  sender: string;
  content: string;
}

export default function Home() {
  const messages = useQuery(api.functions.message.list);
  const createMessage = useMutation(api.functions.message.create);

  const handleSubmit = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    const form = e.currentTarget;
    const messageInput = form.elements.namedItem('message') as HTMLInputElement;

    if (messageInput && messageInput.value) {
      await createMessage({
        sender: 'user',
        content: messageInput.value,
      });
      form.reset();
    }
  };

  return (
    <>
      <>
        {messages?.map((message, index) => (
          <div key={index}>
            <p>{message.sender}</p>
            <p>{message.content}</p>
          </div>
        ))}
        <form onSubmit={handleSubmit}>
          <input type="text" name="message" />
          <button type="submit">Send</button>
        </form>
      </>
    </>
  );
}
