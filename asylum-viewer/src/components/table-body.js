import { getColumnWidth } from '@/lib/columns'

const ROW_HEIGHT = 44

export default function TableBody({ rows, columns, onRowClick, frozenCols = 0, frozenRows = 0, columnLeftOffsets = {}, theadHeight = 0 }) {
  const formatCell = (val, col) => {
    if (val === null || val === undefined) {
      return <span className="text-muted">&mdash;</span>
    }
    if (typeof val === 'boolean') {
      return (
        <span className={`inline-block px-2.5 py-1 rounded text-xs font-semibold tracking-wider ${
          val ? 'bg-yes-bg text-yes-text' : 'bg-no-bg text-no-text'
        }`}>
          {val ? 'YES' : 'NO'}
        </span>
      )
    }
    if (col === 'link' && String(val).startsWith('http')) {
      return (
        <a
          href={val}
          target="_blank"
          rel="noreferrer"
          onClick={e => e.stopPropagation()}
          className="text-accent no-underline font-medium hover:underline"
        >
          PDF &#8599;
        </a>
      )
    }
    return String(val)
  }

  if (rows.length === 0) {
    return (
      <tbody>
        <tr>
          <td colSpan={columns.length} className="text-center py-16 text-muted font-mono tracking-wider">
            NO MATCHING RECORDS
          </td>
        </tr>
      </tbody>
    )
  }

  return (
    <tbody>
      {rows.map((row, i) => {
        const isFrozenRow = i < frozenRows
        const isLastFrozenRow = frozenRows > 0 && i === frozenRows - 1
        const rowBg = i % 2 === 0 ? 'var(--color-surface)' : 'var(--color-row-alt)'

        return (
          <tr
            key={row.link || i}
            onClick={() => onRowClick(row)}
            style={isFrozenRow ? { position: 'sticky', top: theadHeight + i * ROW_HEIGHT, zIndex: 4 } : {}}
            className={[
              'border-b border-border transition-colors hover:bg-row-hover hover:[&>td:first-child]:border-l-2 hover:[&>td:first-child]:border-l-accent cursor-pointer',
              i % 2 === 0 ? 'bg-surface' : 'bg-row-alt',
              isLastFrozenRow ? 'shadow-[0_2px_6px_rgba(0,0,0,0.12)]' : '',
            ].filter(Boolean).join(' ')}
          >
            {columns.map((col, j) => {
              const isFrozenCol = j < frozenCols
              const isLastFrozenCol = frozenCols > 0 && j === frozenCols - 1

              return (
                <td
                  key={col}
                  title={String(row[col] ?? '')}
                  style={{
                    minWidth: getColumnWidth(col),
                    ...(isFrozenCol ? {
                      position: 'sticky',
                      left: columnLeftOffsets[col],
                      zIndex: isFrozenRow ? 6 : 2,
                      background: rowBg,
                    } : {}),
                  }}
                  className={[
                    'px-5 py-3 border-r border-border max-w-[280px] overflow-hidden text-ellipsis whitespace-nowrap align-middle text-sm',
                    isLastFrozenCol ? 'shadow-[2px_0_6px_rgba(0,0,0,0.12)]' : '',
                  ].filter(Boolean).join(' ')}
                >
                  {formatCell(row[col], col)}
                </td>
              )
            })}
          </tr>
        )
      })}
    </tbody>
  )
}
