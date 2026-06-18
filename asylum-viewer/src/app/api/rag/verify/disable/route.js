// Server-side proxy: POST /api/rag/verify/disable -> $RAG_API_URL/verify/disable.
// Unloads the CE verifier model and frees memory on the backend.
//
// This must also work for navigator.sendBeacon (page-unload), which POSTs with
// no JSON content-type and no body — so we never read the request body and just
// forward an empty POST upstream.

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
  try {
    const upstream = await fetch(`${base}/verify/disable`, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      signal: AbortSignal.timeout(60_000),
    })
    const payload = await upstream.text()
    return new NextResponse(payload, {
      status: upstream.status,
      headers: { 'content-type': 'application/json' },
    })
  } catch (err) {
    // sendBeacon ignores the response; still answer so a normal POST gets JSON.
    return NextResponse.json({ loaded: false, error: String(err) }, { status: 502 })
  }
}
