'use client'

import { useRef, useState, useCallback, useEffect } from 'react'

interface UseAutoScrollOptions {
  /** Distance from bottom (in px) to consider "at bottom" */
  threshold?: number
  /** Dependency array that triggers scroll check (e.g., messages) */
  deps?: unknown[]
  /** Whether content is actively streaming (uses instant scroll to avoid jitter) */
  isStreaming?: boolean
}

interface UseAutoScrollReturn {
  /** Ref to attach to the scrollable viewport element */
  viewportRef: React.RefObject<HTMLDivElement>
  /** Whether the user is currently at or near the bottom */
  isAtBottom: boolean
  /** Whether new content arrived while user was scrolled up */
  hasNewMessages: boolean
  /** Scroll to bottom and reset hasNewMessages */
  scrollToBottom: () => void
  /** Handler to attach to onScroll event */
  handleScroll: React.UIEventHandler<HTMLDivElement>
}

export function useAutoScroll(options: UseAutoScrollOptions = {}): UseAutoScrollReturn {
  const { threshold = 50, deps = [], isStreaming = false } = options

  const viewportRef = useRef<HTMLDivElement>(null)
  const [isAtBottom, setIsAtBottom] = useState(true)
  const [hasNewMessages, setHasNewMessages] = useState(false)

  // Track if we should auto-scroll (user hasn't manually scrolled up)
  const shouldAutoScrollRef = useRef(true)
  // Track previous content height to detect new content
  const prevScrollHeightRef = useRef(0)

  const checkIfAtBottom = useCallback(() => {
    const viewport = viewportRef.current
    if (!viewport) return true

    const { scrollTop, scrollHeight, clientHeight } = viewport
    const distanceFromBottom = scrollHeight - scrollTop - clientHeight
    return distanceFromBottom <= threshold
  }, [threshold])

  const scrollToBottom = useCallback(() => {
    const viewport = viewportRef.current
    if (!viewport) return

    viewport.scrollTo({
      top: viewport.scrollHeight,
      behavior: 'smooth',
    })

    // Re-enable auto-scroll and clear new messages indicator
    shouldAutoScrollRef.current = true
    setHasNewMessages(false)
    setIsAtBottom(true)
  }, [])

  const handleScroll = useCallback<React.UIEventHandler<HTMLDivElement>>(() => {
    const atBottom = checkIfAtBottom()
    setIsAtBottom(atBottom)

    if (atBottom) {
      // User scrolled back to bottom - re-enable auto-scroll
      shouldAutoScrollRef.current = true
      setHasNewMessages(false)
    } else {
      // User scrolled up - disable auto-scroll
      shouldAutoScrollRef.current = false
    }
  }, [checkIfAtBottom])

  // Effect to handle new content (messages)
  useEffect(() => {
    const viewport = viewportRef.current
    if (!viewport) return

    const currentScrollHeight = viewport.scrollHeight
    const hasNewContent = currentScrollHeight > prevScrollHeightRef.current

    if (hasNewContent) {
      if (shouldAutoScrollRef.current) {
        // Auto-scroll to bottom
        // Use instant scroll during streaming to avoid jitter from competing animations
        viewport.scrollTo({
          top: currentScrollHeight,
          behavior: isStreaming ? 'instant' : 'smooth',
        })
      } else {
        // Show "new messages" indicator
        setHasNewMessages(true)
      }
    }

    prevScrollHeightRef.current = currentScrollHeight
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps)

  // Initialize scroll position tracking on mount
  useEffect(() => {
    const viewport = viewportRef.current
    if (viewport) {
      prevScrollHeightRef.current = viewport.scrollHeight
      setIsAtBottom(checkIfAtBottom())
    }
  }, [checkIfAtBottom])

  return {
    viewportRef,
    isAtBottom,
    hasNewMessages,
    scrollToBottom,
    handleScroll,
  }
}
