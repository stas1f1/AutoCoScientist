'use client'

import { ScrollArea } from '@/components/ui/scroll-area'
import { MessageList } from './MessageList'
import { InputArea } from './InputArea'
import { WelcomeScreen } from './WelcomeScreen'
import { useSessionStore } from '@/stores/useSessionStore'
import { useAgentWebSocket } from '@/hooks/useAgentWebSocket'
import { useAutoScroll } from '@/hooks/useAutoScroll'
import { ChevronDown } from 'lucide-react'

export function ChatContainer() {
  const { currentSessionId, messages, isStreaming } = useSessionStore()

  // Smart auto-scroll with user scroll detection
  // Use instant scroll during streaming to avoid jitter from competing smooth animations
  const { viewportRef, hasNewMessages, scrollToBottom, handleScroll } = useAutoScroll({
    threshold: 100,
    deps: [messages],
    isStreaming,
  })

  // Connect WebSocket when session is active
  useAgentWebSocket(currentSessionId)

  if (!currentSessionId) {
    return <WelcomeScreen />
  }

  return (
    <div className="flex flex-col h-full">
      {/* Messages Area */}
      <div className="flex-1 overflow-hidden relative">
        <ScrollArea
          className="h-full"
          viewportRef={viewportRef}
          onScroll={handleScroll}
        >
          <div className="max-w-4xl mx-auto py-6 px-4">
            {messages.length === 0 ? (
              <div className="flex flex-col items-center justify-center py-20 text-center">
                <div className="h-16 w-16 rounded-2xl bg-gradient-to-br from-accent/20 to-accent/5 border border-accent/20 flex items-center justify-center mb-4">
                  <span className="text-2xl">🤖</span>
                </div>
                <h2 className="text-lg font-medium text-text-primary mb-2">
                  Ready to assist
                </h2>
                <p className="text-text-secondary max-w-md">
                  Send a message to start your data science task. I can help with
                  data exploration, feature engineering, model training, and more.
                </p>
              </div>
            ) : (
              <MessageList messages={messages} />
            )}
          </div>
        </ScrollArea>

        {/* New Messages Button */}
        {hasNewMessages && (
          <button
            onClick={scrollToBottom}
            className="absolute bottom-4 left-1/2 -translate-x-1/2 flex items-center gap-2 px-4 py-2 bg-accent text-white rounded-full shadow-lg hover:bg-accent/90 transition-all animate-in fade-in slide-in-from-bottom-2 duration-200"
          >
            <ChevronDown className="h-4 w-4" />
            <span className="text-sm font-medium">New messages</span>
          </button>
        )}
      </div>

      {/* Input Area */}
      <div className="border-t border-border bg-background-secondary/50 backdrop-blur-sm">
        <div className="max-w-4xl mx-auto p-4">
          <InputArea />
        </div>
      </div>
    </div>
  )
}
