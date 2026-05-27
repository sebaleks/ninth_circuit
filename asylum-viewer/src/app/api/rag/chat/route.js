// Server-side proxy: forwards POST /api/rag/chat to $RAG_API_URL/chat.
// Keeps the backend URL server-only and avoids the browser making cross-origin
// requests (which would otherwise wake the Render free-tier service from cold
// start with CORS preflights).

import { NextResponse } from 'next/server'

export const runtime = 'nodejs'

export async function POST(request) {
  const base = process.env.RAG_API_URL
  if (!base) {
    return NextResponse.json(
      { error: 'RAG_API_URL not configured on the server' },
      { status: 500 },
    )
  }
  const body = await request.text()
  const upstream = await fetch(`${base}/chat`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body,
    // Render free tier cold-start can take ~30s
    signal: AbortSignal.timeout(60_000),
  })
  const payload = await upstream.text()
  return new NextResponse(payload, {
    status: upstream.status,
    headers: { 'content-type': 'application/json' },
  })
}
