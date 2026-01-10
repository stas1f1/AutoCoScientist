'use client'

import { useEffect, useRef, useCallback } from 'react'
import { apiClient } from '@/lib/api/client'
import { useSessionStore, Message } from '@/stores/useSessionStore'

interface WebSocketMessage {
  type: 'token' | 'tool' | 'status' | 'environment'
  data: string
  message_id?: string
  timestamp: string
  truncated?: boolean
}

export function useAgentWebSocket(sessionId: string | null) {
  const wsRef = useRef<WebSocket | null>(null)
  const reconnectTimeoutRef = useRef<NodeJS.Timeout>()
  const reconnectAttemptsRef = useRef(0)
  const maxReconnectAttempts = 5

  // Get stable action references from store (these don't change)
  const addMessage = useSessionStore((state) => state.addMessage)
  const appendToLastMessage = useSessionStore((state) => state.appendToLastMessage)
  const updateLastMessage = useSessionStore((state) => state.updateLastMessage)
  const setStreaming = useSessionStore((state) => state.setStreaming)
  const setStatus = useSessionStore((state) => state.setStatus)
  const clearMessages = useSessionStore((state) => state.clearMessages)

  const connect = useCallback(() => {
    if (!sessionId) return

    // Close existing connection
    if (wsRef.current) {
      wsRef.current.close()
    }

    const wsUrl = apiClient.getWebSocketUrl(sessionId)
    const ws = new WebSocket(wsUrl)

    ws.onopen = () => {
      console.log('[WS] Connected to session:', sessionId)
      // Clear messages before receiving replayed history from backend
      // Backend replays all messages on reconnect, so we clear to avoid duplicates
      clearMessages()
      setStatus('idle')
      reconnectAttemptsRef.current = 0
    }

    ws.onmessage = (event) => {
      try {
        const msg: WebSocketMessage = JSON.parse(event.data)

        switch (msg.type) {
          case 'token': {
            // Get fresh state to check last message (avoid stale closure)
            const currentMessages = useSessionStore.getState().messages
            const lastMsg = currentMessages[currentMessages.length - 1]

            // Determine the message ID to use
            let incomingId: string
            if (msg.message_id) {
              // Use the LLM-provided message ID
              incomingId = msg.message_id
            } else if (lastMsg?.role === 'assistant' && lastMsg.isStreaming) {
              // No message_id provided, but there's an active streaming message - reuse its ID
              incomingId = lastMsg.id
            } else {
              // First chunk of a new message without message_id - generate new ID
              incomingId = `msg-${Date.now()}`
            }

            // Check if this belongs to a different LLM call
            if (
              !lastMsg ||
              lastMsg.role !== 'assistant' ||
              !lastMsg.isStreaming ||
              lastMsg.id !== incomingId
            ) {
              // Finalize previous message if streaming
              if (lastMsg?.isStreaming) {
                updateLastMessage({ isStreaming: false })
              }
              // Start new message with the LLM's message_id
              const newMessage: Message = {
                id: incomingId,
                role: 'assistant',
                content: msg.data,
                timestamp: new Date(msg.timestamp),
                isStreaming: true,
              }
              addMessage(newMessage)
              setStreaming(true)
              setStatus('streaming')
            } else {
              // Append to existing streaming message
              appendToLastMessage(msg.data)
            }
            break
          }

          case 'tool': {
            // Create separate environment message for tool output
            const toolMessage: Message = {
              id: `tool-${Date.now()}`,
              role: 'environment',
              content: msg.data,
              timestamp: new Date(msg.timestamp),
              isStreaming: false,
              isTruncated: msg.truncated,
            }
            addMessage(toolMessage)
            break
          }

          case 'environment': {
            // Environment output (e.g., bash command output, code execution results)
            const envMessage: Message = {
              id: `env-${Date.now()}`,
              role: 'environment',
              content: msg.data,
              timestamp: new Date(msg.timestamp),
              isStreaming: false,
              isTruncated: msg.truncated,
            }
            addMessage(envMessage)
            break
          }

          case 'status':
            if (msg.data === 'completed' || msg.data === 'done') {
              updateLastMessage({ isStreaming: false })
              setStreaming(false)
              setStatus('idle')
            } else if (msg.data === 'cancelled') {
              updateLastMessage({ isStreaming: false })
              setStreaming(false)
              setStatus('idle')
            } else if (msg.data === 'cancelling') {
              setStatus('cancelling')
            } else if (msg.data.startsWith('Error:') || msg.data.startsWith('error')) {
              updateLastMessage({ isStreaming: false })
              setStreaming(false)
              setStatus('error', msg.data)
            } else if (msg.data === 'started' || msg.data === 'running') {
              setStatus('streaming')
            }
            break
        }
      } catch (e) {
        console.error('[WS] Failed to parse message:', e)
      }
    }

    ws.onclose = (event) => {
      console.log('[WS] Connection closed:', event.code, event.reason)

      // Attempt reconnect if not intentionally closed
      if (event.code !== 1000 && reconnectAttemptsRef.current < maxReconnectAttempts) {
        reconnectAttemptsRef.current++
        const delay = Math.min(1000 * Math.pow(2, reconnectAttemptsRef.current), 10000)
        console.log(`[WS] Reconnecting in ${delay}ms (attempt ${reconnectAttemptsRef.current})`)
        reconnectTimeoutRef.current = setTimeout(connect, delay)
      }
    }

    ws.onerror = (error) => {
      console.error('[WS] Error:', error)
      setStatus('error', 'WebSocket connection error')
    }

    wsRef.current = ws
  }, [sessionId, addMessage, appendToLastMessage, updateLastMessage, setStreaming, setStatus, clearMessages])

  useEffect(() => {
    connect()

    return () => {
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current)
      }
      if (wsRef.current) {
        wsRef.current.close(1000, 'Component unmounting')
      }
    }
  }, [connect])

  const disconnect = useCallback(() => {
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current)
    }
    if (wsRef.current) {
      wsRef.current.close(1000, 'User disconnect')
    }
  }, [])

  return { disconnect, reconnect: connect }
}
