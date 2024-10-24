'use client'

import { useState, useRef, useEffect } from 'react'
import { api } from '../../convex/_generated/api'
import { useMutation, useQuery } from 'convex/react'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Card, CardContent, CardFooter, CardHeader, CardTitle } from '@/components/ui/card'
import { ScrollArea } from '@/components/ui/scroll-area'
import { Send } from 'lucide-react'

interface Message {
  sender: string
  content: string
}

export default function ChatApp() {
  const messages = useQuery(api.functions.message.list)
  const createMessage = useMutation(api.functions.message.create)
  const [input, setInput] = useState('')
  const scrollAreaRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (scrollAreaRef.current) {
      scrollAreaRef.current.scrollTop = scrollAreaRef.current.scrollHeight
    }
  }, [messages])

  const handleSubmit = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault()
    if (input.trim()) {
      await createMessage({
        sender: 'User',
        content: input.trim(),
      })
      setInput('')
    }
  }

  return (
    <div className="flex items-center justify-center min-h-screen bg-gray-100 p-4">
      <Card className="w-full max-w-md">
        <CardHeader>
          <CardTitle className="text-2xl font-bold text-center">Chat App</CardTitle>
        </CardHeader>
        <CardContent>
          <ScrollArea className="h-[400px] pr-4" ref={scrollAreaRef}>
            {messages?.map((message, index) => (
              <div
                key={index}
                className={`mb-4 ${message.sender === 'User' ? 'text-right' : 'text-left'
                  }`}
              >
                <div
                  className={`inline-block p-2 rounded-lg ${message.sender === 'User'
                      ? 'bg-primary text-primary-foreground'
                      : 'bg-secondary text-secondary-foreground'
                    }`}
                >
                  <p className="font-semibold">{message.sender}</p>
                  <p>{message.content}</p>
                </div>
              </div>
            ))}
          </ScrollArea>
        </CardContent>
        <CardFooter>
          <form onSubmit={handleSubmit} className="flex w-full space-x-2">
            <Input
              type="text"
              placeholder="Type your message..."
              value={input}
              onChange={(e) => setInput(e.target.value)}
              className="flex-grow"
            />
            <Button type="submit" size="icon">
              <Send className="h-4 w-4" />
              <span className="sr-only">Send message</span>
            </Button>
          </form>
        </CardFooter>
      </Card>
    </div>
  )
}