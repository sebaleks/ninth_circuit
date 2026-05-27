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
