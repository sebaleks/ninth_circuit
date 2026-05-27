'use client'

import { useEffect, useRef } from 'react'
import CitationCard from './citation-card'

function renderAnswerWithLinks(answer) {
  // Replace [N] markers with styled spans (kept inline; citations panel below has the cards)
  const parts = answer.split(/(\[\d+\])/g)
  return parts.map((part, i) => {
    const match = part.match(/^\[(\d+)\]$/)
    if (match) {
      return (
        <a
          key={i}
          href={`#cite-${match[1]}`}
          className="inline-block px-1 mx-0.5 text-[10px] font-mono text-accent border border-accent rounded-sm leading-tight"
        >
          {match[1]}
        </a>
      )
    }
    return <span key={i}>{part}</span>
  })
}

function MessageBubble({ msg }) {
  if (msg.role === 'user') {
    return (
      <div className="flex justify-end">
        <div className="max-w-[85%] bg-accent/10 border border-accent/30 text-text px-3 py-2 text-sm">
          {msg.content}
        </div>
      </div>
    )
  }
  if (msg.role === 'assistant') {
    return (
      <div className="space-y-2">
        <div className="flex items-center gap-2">
          <span className="text-[10px] font-mono tracking-wider text-muted uppercase">
            Assistant
          </span>
          <span className="text-[9px] font-mono tracking-wider px-1.5 py-0.5 bg-no-bg text-no-text uppercase">
            experimental
          </span>
          {msg.latency_ms != null && (
            <span className="text-[10px] font-mono text-muted">{msg.latency_ms} ms</span>
          )}
        </div>
        <div className="text-sm text-text leading-relaxed whitespace-pre-wrap">
          {renderAnswerWithLinks(msg.content)}
        </div>
        {msg.citations && msg.citations.length > 0 && (
          <div className="space-y-2 pt-2">
            <span className="text-[10px] font-mono tracking-wider text-muted uppercase">
              Citations
            </span>
            {msg.citations.map((c, i) => (
              <div key={c.chunk_id} id={`cite-${i + 1}`}>
                <CitationCard citation={c} index={i + 1} />
              </div>
            ))}
          </div>
        )}
      </div>
    )
  }
  if (msg.role === 'error') {
    return (
      <div className="border border-no-bg bg-no-bg/30 text-no-text text-xs p-3">
        {msg.content}
      </div>
    )
  }
  return null
}

export default function ChatMessages({ messages, loading }) {
  const endRef = useRef(null)

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, loading])

  if (messages.length === 0 && !loading) {
    return (
      <div className="flex-1 flex items-center justify-center text-muted text-xs font-mono tracking-wider px-6 text-center">
        ASK ABOUT THE CASES IN THIS CORPUS.
        <br />
        EXAMPLES: &quot;gang persecution&quot;, &quot;one-year filing bar&quot;,
        <br />
        &quot;credibility findings&quot;.
      </div>
    )
  }

  return (
    <div className="flex-1 overflow-y-auto p-4 space-y-6">
      {messages.map((m, i) => (
        <MessageBubble key={i} msg={m} />
      ))}
      {loading && (
        <div className="flex items-center gap-2 text-muted text-xs font-mono tracking-wider animate-pulse">
          <span>THINKING</span>
          <span className="inline-block w-1 h-1 bg-muted rounded-full" />
          <span className="inline-block w-1 h-1 bg-muted rounded-full" />
          <span className="inline-block w-1 h-1 bg-muted rounded-full" />
        </div>
      )}
      <div ref={endRef} />
    </div>
  )
}
