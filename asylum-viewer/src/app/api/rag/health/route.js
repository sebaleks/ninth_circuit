import { NextResponse } from 'next/server'

export const runtime = 'nodejs'

export async function GET() {
  const base = process.env.RAG_API_URL
  if (!base) {
    return NextResponse.json(
      { error: 'RAG_API_URL not configured on the server' },
      { status: 500 },
    )
  }
  try {
    const upstream = await fetch(`${base}/health`, {
      signal: AbortSignal.timeout(60_000),
    })
    const payload = await upstream.text()
    return new NextResponse(payload, {
      status: upstream.status,
      headers: { 'content-type': 'application/json' },
    })
  } catch (err) {
    return NextResponse.json({ status: 'unreachable', error: String(err) }, { status: 502 })
  }
}
