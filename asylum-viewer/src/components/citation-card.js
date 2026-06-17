'use client'

// Semantic confidence colors, kept visually distinct from the brand accent (red #E3120B).
// Each entry maps a confidence.color name -> { text, dot } Tailwind classes.
const CONFIDENCE_CLASSES = {
  green: { text: 'text-emerald-600', dot: 'bg-emerald-500' },
  yellow: { text: 'text-amber-600', dot: 'bg-amber-500' },
  red: { text: 'text-rose-600', dot: 'bg-rose-500' },
}

export default function CitationCard({ citation, index }) {
  const { case_link, snippet, page, score, dense_score, case_disposition, case_pub_status, confidence } = citation
  // Trim very long snippets for the card view
  const trimmed = snippet.length > 280 ? snippet.slice(0, 280).trim() + '…' : snippet
  const filename = case_link.split('/').pop()?.replace('.pdf', '') ?? case_link
  // dense_score is the meaningful cosine relevance; fall back to the legacy RRF rank score.
  const relevance = Number.isFinite(dense_score) ? dense_score : score
  const confidenceClasses = confidence ? (CONFIDENCE_CLASSES[confidence.color] ?? CONFIDENCE_CLASSES.yellow) : null

  return (
    <a
      href={case_link}
      target="_blank"
      rel="noopener noreferrer"
      className="block border border-border bg-filter-bg hover:border-accent transition-colors p-3 text-xs"
    >
      <div className="flex items-center justify-between gap-2 mb-2">
        <div className="flex items-center gap-2 min-w-0">
          <span className="font-mono text-[10px] tracking-wider text-muted shrink-0">
            [{index}]
          </span>
          <span className="font-mono text-[11px] text-text truncate">{filename}</span>
        </div>
        <div className="flex items-center gap-2 shrink-0">
          {confidence && (
            <span
              title={confidence.tooltip}
              className={`flex items-center gap-1 text-[10px] font-mono tracking-wider ${confidenceClasses.text}`}
            >
              <span className={`inline-block w-2 h-2 rounded-full shrink-0 ${confidenceClasses.dot}`} />
              {confidence.label}
            </span>
          )}
          {case_pub_status && (
            <span className="text-[10px] font-mono tracking-wider text-muted uppercase">
              {case_pub_status}
            </span>
          )}
          <span className="text-[10px] font-mono text-muted">
            p.{page} · {Math.round(relevance * 100)}%
          </span>
        </div>
      </div>
      {case_disposition && (
        <div className="text-[10px] font-mono tracking-wider text-muted uppercase mb-1.5">
          {case_disposition}
        </div>
      )}
      <p className="text-text leading-relaxed whitespace-pre-wrap">{trimmed}</p>
    </a>
  )
}
