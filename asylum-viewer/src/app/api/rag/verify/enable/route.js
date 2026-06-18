// Server-side proxy: POST /api/rag/verify/enable -> $RAG_API_URL/verify/enable.
// Loads the cross-encoder (CE) verifier model on the backend. This takes a few
// seconds on the weak CPU, so we allow a generous timeout.

import { NextResponse } from 'next/server'

export const runtime = 'nodejs'

export async function POST() {
  const base = process.env.RAG_API_URL
  if (!base) {
    return NextResponse.json(
      { error: 'RAG_API_URL not configured on the server' },
      { status: 500 },
    )
  }
  const upstream = await fetch(`${base}/verify/enable`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    // Loading the CE model on a cold/weak CPU can take a while.
    signal: AbortSignal.timeout(120_000),
  })
  const payload = await upstream.text()
  return new NextResponse(payload, {
    status: upstream.status,
    headers: { 'content-type': 'application/json' },
  })
}
