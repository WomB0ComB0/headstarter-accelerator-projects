'use client'

import { Message } from "@/app/types"
import { cn } from "@/lib/utils"
import { Avatar, AvatarFallback, AvatarImage } from "@/components"
import { format } from "date-fns"

interface ChatMessageProps {
  message: Message
}

export function ChatMessage({ message }: ChatMessageProps) {
  return (
    <div
      className={cn(
        "flex gap-3 mb-4",
        message.role === "user" ? "justify-end" : "justify-start"
      )}
    >
      {message.role === "assistant" && (
        <Avatar className="w-8 h-8">
          <AvatarImage src="/bot-avatar.png" alt="AI Assistant" />
          <AvatarFallback>AI</AvatarFallback>
        </Avatar>
      )}
      <div
        className={cn(
          "px-4 py-2 rounded-lg max-w-[80%]",
          message.role === "assistant"
            ? "bg-secondary text-secondary-foreground"
            : "bg-primary text-primary-foreground"
        )}
      >
        <p className="mb-1">{message.content}</p>
        <time className="text-xs opacity-50">
          {format(message.timestamp, "HH:mm")}
        </time>
      </div>
      {message.role === "user" && (
        <Avatar className="w-8 h-8">
          <AvatarImage src="/user-avatar.png" alt="User" />
          <AvatarFallback>U</AvatarFallback>
        </Avatar>
      )}
    </div>
  )
}

