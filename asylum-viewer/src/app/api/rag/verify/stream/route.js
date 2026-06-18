// Server-side proxy: GET /api/rag/verify/stream -> $RAG_API_URL/verify/stream.
// Proxies a Server-Sent Events (SSE) stream of per-result CE verification.
//
// LOAD-BEARING: the backend stops scoring when its HTTP request disconnects (it
// checks between each result). We pass `{ signal: request.signal }` to the
// upstream fetch — Next's route handler aborts `request.signal` when the browser
// closes the connection (EventSource.close() / Stop / navigate away). That abort
// propagates to the upstream fetch, dropping the backend's GET, which is exactly
// how the backend learns to halt and free the CPU. We then return the upstream
// `body` ReadableStream verbatim so chunks flow through unbuffered.

import { NextResponse } from 'next/server'

export const runtime = 'nodejs'

export async function GET(request) {
  const base = process.env.RAG_API_URL
  if (!base) {
    return NextResponse.json(
      { error: 'RAG_API_URL not configured on the server' },
      { status: 500 },
    )
  }

  const { searchParams } = request.nextUrl
  const query = searchParams.get('query') ?? ''
  const k = searchParams.get('k') ?? '5'
  const upstreamUrl =
    `${base}/verify/stream?query=${encodeURIComponent(query)}&k=${encodeURIComponent(k)}`

  let upstream
  try {
    upstream = await fetch(upstreamUrl, {
      method: 'GET',
      headers: { accept: 'text/event-stream' },
      // Propagate the client disconnect to the backend so it halts scoring.
      signal: request.signal,
    })
  } catch (err) {
    // Aborts surface here as well; nothing to stream once disconnected.
    if (request.signal.aborted) return new Response(null, { status: 499 })
    return NextResponse.json({ error: String(err) }, { status: 502 })
  }

  // Stream the upstream SSE body straight through. Disabling buffering keeps
  // events flowing in real time.
  return new Response(upstream.body, {
    status: upstream.status,
    headers: {
      'Content-Type': 'text/event-stream',
      'Cache-Control': 'no-cache, no-transform',
      Connection: 'keep-alive',
      'X-Accel-Buffering': 'no',
    },
  })
}
