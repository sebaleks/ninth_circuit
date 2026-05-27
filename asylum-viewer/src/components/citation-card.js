'use client'

export default function CitationCard({ citation, index }) {
  const { case_link, snippet, page, score, case_disposition, case_pub_status } = citation
  // Trim very long snippets for the card view
  const trimmed = snippet.length > 280 ? snippet.slice(0, 280).trim() + '…' : snippet
  const filename = case_link.split('/').pop()?.replace('.pdf', '') ?? case_link

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
          {case_pub_status && (
            <span className="text-[10px] font-mono tracking-wider text-muted uppercase">
              {case_pub_status}
            </span>
          )}
          <span className="text-[10px] font-mono text-muted">
            p.{page} · {Math.round(score * 100)}%
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
