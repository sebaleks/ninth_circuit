'use client'

// Client-side fetch wrappers for the RAG backend.
// The browser hits Next.js's /api/rag/* proxy routes, which in turn hit
// process.env.RAG_API_URL (server-only) so we don't leak the backend URL
// or expose it to CORS.

export async function chat(question, k = 5) {
  const res = await fetch('/api/rag/chat', {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ question, k }),
  })
  if (!res.ok) throw new Error(`chat failed: ${res.status}`)
  return res.json()
}

export async function search(query, k = 10) {
  const res = await fetch('/api/rag/search', {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ query, k }),
  })
  if (!res.ok) throw new Error(`search failed: ${res.status}`)
  return res.json()
}

export async function health() {
  const res = await fetch('/api/rag/health')
  if (!res.ok) throw new Error(`health failed: ${res.status}`)
  return res.json()
}

// --- Result Cross Verification (opt-in CE verifier) ---

// Loads the cross-encoder model on the backend. Slow (a few seconds) — the
// caller should show a loading state. Returns { loaded: true }.
export async function enableVerify() {
  const res = await fetch('/api/rag/verify/enable', { method: 'POST' })
  if (!res.ok) throw new Error(`enableVerify failed: ${res.status}`)
  return res.json()
}

// Unloads the CE model + frees memory. When called from a page-unload handler,
// pass { beacon: true } so it uses navigator.sendBeacon — a normal fetch is
// cancelled by the browser as the page tears down, but a beacon is guaranteed
// to be sent. Returns true (beacon) or the parsed JSON (normal POST).
export async function disableVerify({ beacon = false } = {}) {
  if (beacon && typeof navigator !== 'undefined' && navigator.sendBeacon) {
    // sendBeacon POSTs with no JSON content-type; the proxy route accepts that.
    return navigator.sendBeacon('/api/rag/verify/disable')
  }
  const res = await fetch('/api/rag/verify/disable', { method: 'POST' })
  if (!res.ok) throw new Error(`disableVerify failed: ${res.status}`)
  return res.json()
}

// Opens the SSE stream for `query` and returns the EventSource. The caller wires
// up onmessage / onerror, and calls .close() to abort — closing the EventSource
// disconnects the proxy, which aborts the upstream backend fetch and halts
// scoring (frees the CPU). EventSource always issues a GET, which suits an SSE
// endpoint and carries the query in the URL.
export function openVerifyStream(query, k = 5) {
  const url =
    `/api/rag/verify/stream?query=${encodeURIComponent(query)}&k=${encodeURIComponent(k)}`
  return new EventSource(url)
}
