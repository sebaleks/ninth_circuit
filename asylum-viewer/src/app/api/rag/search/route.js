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
  // Forward the incoming query string (e.g. ?include_timings=true) to Render so
  // the per-stage `timings` block survives the proxy. The response is already a
  // verbatim text passthrough below, so `timings` is preserved once requested.
  const upstream = await fetch(`${base}/search${new URL(request.url).search}`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body,
    signal: AbortSignal.timeout(60_000),
  })
  const payload = await upstream.text()
  return new NextResponse(payload, {
    status: upstream.status,
    headers: { 'content-type': 'application/json' },
  })
}
