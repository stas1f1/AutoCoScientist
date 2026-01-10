'use client'

import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { apiClient, Session } from '@/lib/api/client'

export function useSessions() {
  return useQuery({
    queryKey: ['sessions'],
    queryFn: async () => {
      try {
        const sessions = await apiClient.listSessions()
        // Ensure we always return an array
        return Array.isArray(sessions) ? sessions : []
      } catch (error) {
        console.error('[useSessions] Failed to fetch sessions:', error)
        return []
      }
    },
    staleTime: 30000,
    retry: 2,
    retryDelay: 1000,
  })
}

export function useSession(id: string | null) {
  return useQuery({
    queryKey: ['session', id],
    queryFn: () => (id ? apiClient.getSession(id) : null),
    enabled: !!id,
  })
}

export function useSessionHistory(id: string | null) {
  return useQuery({
    queryKey: ['session-history', id],
    queryFn: () => (id ? apiClient.getSessionHistory(id) : []),
    enabled: !!id,
  })
}

export function useCreateSession() {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: () => apiClient.createSession(),
    onSuccess: (newSession) => {
      queryClient.invalidateQueries({ queryKey: ['sessions'] })
      return newSession
    },
  })
}

export function useSendMessage() {
  return useMutation({
    mutationFn: ({ sessionId, message }: { sessionId: string; message: string }) =>
      apiClient.sendMessage(sessionId, message),
  })
}

export function useExecuteTask() {
  return useMutation({
    mutationFn: ({
      sessionId,
      task,
      options,
    }: {
      sessionId: string
      task: string
      options?: { provider?: string; model?: string; api_key?: string }
    }) => apiClient.executeTask(sessionId, task, options),
  })
}
