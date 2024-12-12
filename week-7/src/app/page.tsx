"use client"

import { useState } from "react"
import { useAtom } from 'jotai'
import { Message } from "./types"
import { ChatInput, Button, Input, ChatMessage } from "@/components"
import { Trash2 } from 'lucide-react'
import { useToast } from "@/hooks/use-toast"
import { urlAtom } from '@/atoms'

export default function Chat() {
  const [messages, setMessages] = useState<Message[]>([
    {
      role: "assistant",
      content: "Hello! How can I help you today?",
      timestamp: new Date(),
    },
  ])
  const [isLoading, setIsLoading] = useState(false)
  const [url, setUrl] = useAtom(urlAtom)
  const { toast } = useToast()

  const handleSend = async (message: string) => {
    const userMessage: Message = {
      role: "user",
      content: message,
      timestamp: new Date(),
    }
    setMessages((prev) => [...prev, userMessage])
    setIsLoading(true)

    try {
      const response = await fetch("/api/chat", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          messages: messages.map((msg) => ({
            role: msg.role,
            content: msg.content,
          })),
          url: url,
        }),
      })

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`)
      }

      const data = await response.json()

      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content: data.message,
          timestamp: new Date(),
        },
      ])
    } catch (error) {
      console.error("Error:", error)
      toast({
        title: "Error",
        description: "Failed to send message. Please try again.",
        variant: "destructive",
      })
    } finally {
      setIsLoading(false)
    }
  }

  const handleClearConversation = () => {
    setMessages([
      {
        role: "assistant",
        content: "Hello! How can I help you today?",
        timestamp: new Date(),
      },
    ])
  }

  return (
    <div className="flex flex-col h-screen bg-background">
      <header className="sticky top-0 z-10 bg-background border-b">
        <div className="container flex justify-between items-center py-4">
          <h1 className="text-2xl font-bold">AI Chat</h1>
          <Button
            variant="outline"
            size="icon"
            onClick={handleClearConversation}
            title="Clear conversation"
          >
            <Trash2 className="h-4 w-4" />
            <span className="sr-only">Clear conversation</span>
          </Button>
        </div>
      </header>

      <main className="flex-1 overflow-y-auto py-4">
        <div className="container space-y-4">
          <div className="flex items-center space-x-2">
            <Input
              type="url"
              placeholder="Enter URL"
              value={url}
              onChange={(e: React.ChangeEvent<HTMLInputElement>) => setUrl(e.target.value)}
              className="flex-grow"
            />
          </div>
          {messages.map((msg, index) => (
            <ChatMessage key={index} message={msg} />
          ))}
          {isLoading && (
            <div className="flex justify-start">
              <div className="bg-secondary text-secondary-foreground px-4 py-2 rounded-lg">
                <div className="flex items-center gap-1">
                  <div className="w-2 h-2 bg-primary rounded-full animate-bounce [animation-delay:-0.3s]"></div>
                  <div className="w-2 h-2 bg-primary rounded-full animate-bounce [animation-delay:-0.15s]"></div>
                  <div className="w-2 h-2 bg-primary rounded-full animate-bounce"></div>
                </div>
              </div>
            </div>
          )}
        </div>
      </main>

      <footer className="sticky bottom-0 z-10 bg-background border-t">
        <div className="container py-4">
          <ChatInput onSend={handleSend} isLoading={isLoading} />
        </div>
      </footer>
    </div>
  )
}

