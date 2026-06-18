'use client'

import { useCallback, useEffect, useRef, useState } from 'react'
import {
  search as searchApi,
  enableVerify,
  disableVerify,
  openVerifyStream,
} from '@/lib/rag-client'
import ChatInput from './chat-input'
import ChatMessages from './chat-messages'
import VerifyControls from './verify-controls'

const STORAGE_KEY = 'asylum-chat-open'

export default function ChatPanel() {
  const [open, setOpen] = useState(false)
  const [messages, setMessages] = useState([])
  const [loading, setLoading] = useState(false)

  // Result Cross Verification (opt-in CE verifier).
  // verifyModel: 'off' | 'loading' | 'ready'  (server-side model state)
  // streaming: true while an SSE scoring pass is in flight
  const [verifyModel, setVerifyModel] = useState('off')
  const [streaming, setStreaming] = useState(false)
  const [verifyNote, setVerifyNote] = useState(null)
  const esRef = useRef(null)

  // Close any open stream (→ proxy disconnects → backend halts scoring).
  const closeStream = useCallback(() => {
    if (esRef.current) {
      esRef.current.close()
      esRef.current = null
    }
    setStreaming(false)
  }, [])

  // Unload the model server-side when the page goes away, and tear down any
  // live stream on unmount. Uses sendBeacon on pagehide so the POST survives.
  useEffect(() => {
    const onPageHide = () => {
      closeStream()
      if (verifyModel === 'ready' || verifyModel === 'loading') {
        disableVerify({ beacon: true })
      }
    }
    window.addEventListener('pagehide', onPageHide)
    window.addEventListener('beforeunload', onPageHide)
    return () => {
      window.removeEventListener('pagehide', onPageHide)
      window.removeEventListener('beforeunload', onPageHide)
      closeStream()
    }
  }, [closeStream, verifyModel])

  const handleEnable = async () => {
    setVerifyNote(null)
    setVerifyModel('loading')
    try {
      const res = await enableVerify()
      setVerifyModel(res?.loaded ? 'ready' : 'off')
      if (!res?.loaded) setVerifyNote('Verifier failed to load — try again.')
    } catch (err) {
      setVerifyModel('off')
      setVerifyNote(`Verifier failed to load: ${err.message}`)
    }
  }

  const handleDisable = async () => {
    closeStream()
    setVerifyModel('off')
    setVerifyNote(null)
    try {
      await disableVerify()
    } catch {
      // best-effort unload; UI is already reverted
    }
  }

  // Apply a streamed `result` event to the target assistant message: update the
  // matching card by case_link, or append a backfilled (rank > k) result.
  const applyResultEvent = useCallback((msgIndex, ev) => {
    const confidence = {
      label: ev.label,
      color: ev.color,
      treatment: ev.treatment,
      tooltip: `CE ${Number(ev.ce_score).toFixed(2)} · dense ${Number(ev.dense_score).toFixed(2)}`,
    }
    setMessages((prev) =>
      prev.map((m, i) => {
        if (i !== msgIndex || m.role !== 'assistant') return m
        const cites = m.citations || []
        const at = cites.findIndex((c) => c.case_link === ev.result_id)
        if (at >= 0) {
          const next = cites.slice()
          next[at] = { ...next[at], confidence }
          return { ...m, citations: next }
        }
        // Backfilled result not currently shown — add it as a new card.
        const added = {
          case_link: ev.case_link ?? ev.result_id,
          chunk_id: `verify-${ev.result_id}-${ev.rank}`,
          snippet: ev.snippet ?? '',
          page: ev.page,
          dense_score: ev.dense_score,
          case_disposition: ev.case_disposition,
          case_pub_status: ev.case_pub_status,
          confidence,
        }
        return { ...m, citations: [...cites, added] }
      }),
    )
  }, [])

  const handleVerify = (msgIndex, query) => {
    if (verifyModel !== 'ready' || streaming) return
    setVerifyNote(null)
    closeStream()
    const es = openVerifyStream(query, 5)
    esRef.current = es
    setStreaming(true)

    es.onmessage = (e) => {
      let ev
      try {
        ev = JSON.parse(e.data)
      } catch {
        return
      }
      if (ev.event === 'result') {
        applyResultEvent(msgIndex, ev)
      } else if (ev.event === 'not_applicable') {
        setVerifyNote(
          'Results already high-confidence / lookup — verification not applicable.',
        )
      } else if (ev.event === 'error') {
        if (ev.reason === 'not_loaded') {
          setVerifyModel('off')
          setVerifyNote('Verifier was unloaded — click Enable to load it again.')
        } else {
          setVerifyNote(`Verification error: ${ev.reason ?? 'unknown'}`)
        }
        closeStream()
      } else if (ev.event === 'done') {
        closeStream()
      }
    }
    es.onerror = () => {
      // EventSource fires onerror on normal stream close too. Only surface a
      // message if the model went away; otherwise just stop streaming.
      closeStream()
    }
  }

  // Restore open/closed state from localStorage
  useEffect(() => {
    try {
      const saved = window.localStorage.getItem(STORAGE_KEY)
      if (saved === '1') setOpen(true)
    } catch {}
  }, [])

  useEffect(() => {
    try {
      window.localStorage.setItem(STORAGE_KEY, open ? '1' : '0')
    } catch {}
  }, [open])

  const handleSubmit = async (query) => {
    // A fresh search supersedes any in-flight verification.
    closeStream()
    setVerifyNote(null)
    setMessages((m) => [...m, { role: 'user', content: query }])
    setLoading(true)
    try {
      const resp = await searchApi(query, 5)
      setMessages((m) => [
        ...m,
        {
          role: 'assistant',
          query,
          citations: resp.hits || [],
          latency_ms: resp.latency_ms,
          refused: resp.refused,
        },
      ])
    } catch (err) {
      setMessages((m) => [
        ...m,
        {
          role: 'error',
          content: `Request failed: ${err.message}. The backend may be cold-starting — try again in ~30s.`,
        },
      ])
    } finally {
      setLoading(false)
    }
  }

  // The latest assistant message holds the "current query" + its result cards —
  // that's what Cross Verify operates on.
  let lastAssistant = null
  for (let i = messages.length - 1; i >= 0; i--) {
    if (messages[i].role === 'assistant' && !messages[i].refused) {
      lastAssistant = { index: i, query: messages[i].query }
      break
    }
  }

  if (!open) {
    // Collapsed: thin vertical tab at the right edge of the viewport
    return (
      <button
        type="button"
        onClick={() => setOpen(true)}
        className="hidden sm:flex fixed top-1/2 right-0 -translate-y-1/2 z-30 flex-col items-center gap-2 bg-drawer-bg border border-r-0 border-border px-2 py-3 text-text hover:border-accent hover:text-accent transition-colors"
        title="Open case search"
        aria-label="Open case search"
      >
        <span className="text-base">🔎</span>
        <span className="text-[10px] font-mono tracking-wider uppercase [writing-mode:vertical-rl]">
          Search
        </span>
      </button>
    )
  }

  return (
    <aside className="hidden sm:flex flex-col w-[400px] shrink-0 border-l border-border bg-drawer-bg">
      {/* Testing-mode banner — always visible while panel is open */}
      <div className="bg-yes-bg text-yes-text border-b border-border px-3 py-2 text-[11px] font-mono tracking-wider leading-snug">
        🧪 TESTING MODE — retrieval may miss relevant cases; always verify against the cited PDFs.
      </div>

      {/* Header */}
      <div className="flex items-center justify-between px-3 py-2 border-b border-border">
        <div className="flex items-center gap-2">
          <span className="font-mono text-xs tracking-wider uppercase text-text font-semibold">
            Case Search
          </span>
          <span className="text-[9px] font-mono tracking-wider px-1.5 py-0.5 bg-no-bg text-no-text uppercase">
            beta
          </span>
        </div>
        <div className="flex items-center gap-1">
          {messages.length > 0 && (
            <button
              type="button"
              onClick={() => setMessages([])}
              className="px-2 py-1 text-[10px] font-mono tracking-wider uppercase text-muted hover:text-text transition-colors"
              title="Clear conversation"
            >
              Clear
            </button>
          )}
          <button
            type="button"
            onClick={() => setOpen(false)}
            className="text-muted hover:text-text transition-colors text-lg leading-none px-1"
            title="Close chat"
            aria-label="Close chat"
          >
            ×
          </button>
        </div>
      </div>

      <ChatMessages messages={messages} loading={loading} />

      <VerifyControls
        verifyModel={verifyModel}
        streaming={streaming}
        note={verifyNote}
        canVerify={lastAssistant != null}
        onEnable={handleEnable}
        onDisable={handleDisable}
        onVerify={() =>
          lastAssistant != null &&
          handleVerify(lastAssistant.index, lastAssistant.query)
        }
        onStop={closeStream}
      />

      <ChatInput onSubmit={handleSubmit} disabled={loading} />
    </aside>
  )
}
