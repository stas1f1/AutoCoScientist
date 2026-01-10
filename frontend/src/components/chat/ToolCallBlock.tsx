'use client'

import { useState } from 'react'
import {
  Code,
  Terminal,
  FileCode,
  CheckSquare,
  Send,
  ChevronDown,
  ChevronRight,
  Copy,
  Check,
} from 'lucide-react'
import { cn } from '@/lib/utils/cn'
import { CodeBlock } from './CodeBlock'

interface ToolCallBlockProps {
  tool: string
  content: string
  attributes?: Record<string, string>
}

const toolIcons: Record<string, typeof Code> = {
  jupyter: Code,
  ipython: Code,
  shell: Terminal,
  codeblocks: FileCode,
  fileblocks: FileCode,
  todo: CheckSquare,
  submit: Send,
}

const toolColors: Record<string, string> = {
  jupyter: 'text-orange-400 border-orange-400/30 bg-orange-400/5',
  ipython: 'text-orange-400 border-orange-400/30 bg-orange-400/5',
  shell: 'text-emerald-400 border-emerald-400/30 bg-emerald-400/5',
  codeblocks: 'text-sky-400 border-sky-400/30 bg-sky-400/5',
  fileblocks: 'text-purple-400 border-purple-400/30 bg-purple-400/5',
  todo: 'text-yellow-400 border-yellow-400/30 bg-yellow-400/5',
  submit: 'text-accent border-accent/30 bg-accent/5',
}

export function ToolCallBlock({ tool, content, attributes }: ToolCallBlockProps) {
  const [expanded, setExpanded] = useState(true)
  const [copied, setCopied] = useState(false)

  const Icon = toolIcons[tool.toLowerCase()] || Code
  const colorClass = toolColors[tool.toLowerCase()] || 'text-text-secondary border-border bg-surface'

  // Determine language for syntax highlighting
  const language = tool.toLowerCase() === 'shell' ? 'bash' : 'python'

  // Check if content looks like code output vs input
  const isOutput = content.startsWith('>>>') || content.includes('[ERROR]')

  const handleCopy = async () => {
    await navigator.clipboard.writeText(content)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  return (
    <div className={cn('tool-block rounded-lg border overflow-hidden', colorClass)}>
      {/* Header */}
      <div
        className={cn(
          'flex items-center justify-between px-3 py-2 cursor-pointer select-none',
          'hover:bg-black/10 transition-colors'
        )}
        onClick={() => setExpanded(!expanded)}
      >
        <div className="flex items-center gap-2">
          {expanded ? (
            <ChevronDown className="h-4 w-4" />
          ) : (
            <ChevronRight className="h-4 w-4" />
          )}
          <Icon className="h-4 w-4" />
          <span className="font-mono text-xs font-medium uppercase tracking-wider">
            {tool}
          </span>
          {attributes?.lang && (
            <span className="text-2xs px-1.5 py-0.5 rounded bg-black/20">
              {attributes.lang}
            </span>
          )}
        </div>
        <button
          onClick={(e) => {
            e.stopPropagation()
            handleCopy()
          }}
          className="p-1 rounded hover:bg-black/20 transition-colors"
        >
          {copied ? (
            <Check className="h-3.5 w-3.5" />
          ) : (
            <Copy className="h-3.5 w-3.5" />
          )}
        </button>
      </div>

      {/* Content */}
      {expanded && (
        <div className="border-t border-current/10">
          {isOutput ? (
            <pre className="p-3 overflow-x-auto text-sm font-mono bg-black/20">
              <code className="text-text-primary whitespace-pre-wrap break-all">
                {content}
              </code>
            </pre>
          ) : (
            <CodeBlock language={language} code={content} />
          )}
        </div>
      )}
    </div>
  )
}
