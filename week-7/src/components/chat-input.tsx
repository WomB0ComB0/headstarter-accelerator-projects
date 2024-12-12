'use client'

import { Textarea, Button } from "@/components"
import { SendHorizontal } from 'lucide-react'
import { useAtom } from 'jotai/react'
import { messageAtom } from "@/atoms"

interface ChatInputProps {
  onSend: (message: string) => void
  isLoading: boolean
}

export function ChatInput({ onSend, isLoading }: ChatInputProps) {
  const [message, setMessage] = useAtom(messageAtom)

  const handleSend = () => {
    if (message.trim()) {
      onSend(message)
      setMessage("")
    }
  }

  return (
    <div className="flex gap-2 items-end">
      <Textarea
        value={message}
        onChange={(e) => setMessage(e.target.value)}
        onKeyPress={(e) => e.key === "Enter" && !e.shiftKey && handleSend()}
        placeholder="Type your message..."
        className="flex-1 min-h-[80px] resize-none"
      />
      <Button onClick={handleSend} disabled={isLoading || !message.trim()}>
        <SendHorizontal className="mr-2 h-4 w-4" />
        {isLoading ? "Sending..." : "Send"}
      </Button>
    </div>
  )
}

