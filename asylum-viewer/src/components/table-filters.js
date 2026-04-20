import { getFilterType } from '@/lib/columns'
import { BINARY_COLS } from '@/lib/filters'

export default function TableFilters({ columns, filters, onFilterChange, frozenCols = 0, columnLeftOffsets = {} }) {
  const renderFilter = (col) => {
    const filterType = getFilterType(col)
    if (filterType === 'none') return null

    const baseClass = 'w-full min-w-[70px] px-2.5 py-2 bg-surface border border-border text-text font-mono text-xs outline-none transition-colors focus:border-accent'

    if (filterType === 'binary') {
      return (
        <select
          value={filters[col] || ''}
          onChange={e => onFilterChange(col, e.target.value)}
          className={baseClass}
        >
          <option value="">All</option>
          {BINARY_COLS[col].map(opt => <option key={opt} value={opt}>{opt}</option>)}
        </select>
      )
    }

    if (filterType === 'boolean') {
      return (
        <select
          value={filters[col] || ''}
          onChange={e => onFilterChange(col, e.target.value)}
          className={baseClass}
        >
          <option value="">All</option>
          <option value="true">Yes</option>
          <option value="false">No</option>
          <option value="null">&mdash;</option>
        </select>
      )
    }

    if (filterType === 'date') {
      return (
        <input
          type="text"
          placeholder="2024 or 2024-03..."
          value={filters[col] || ''}
          onChange={e => onFilterChange(col, e.target.value)}
          className={`${baseClass} placeholder:text-muted`}
        />
      )
    }

    if (filterType === 'numeric') {
      return (
        <input
          type="number"
          placeholder="min..."
          value={filters[col] || ''}
          onChange={e => onFilterChange(col, e.target.value)}
          className={`${baseClass} placeholder:text-muted`}
        />
      )
    }

    return (
      <input
        type="text"
        placeholder="filter..."
        value={filters[col] || ''}
        onChange={e => onFilterChange(col, e.target.value)}
        className={`${baseClass} placeholder:text-muted`}
      />
    )
  }

  return (
    <tr>
      {columns.map((col, i) => {
        const isSticky = i < frozenCols
        const isLastFrozen = frozenCols > 0 && i === frozenCols - 1
        return (
          <th
            key={col}
            style={isSticky ? { position: 'sticky', left: columnLeftOffsets[col], zIndex: 20 } : {}}
            className={[
              'bg-filter-bg px-3 py-2 border-b-2 border-border border-r border-r-border',
              isLastFrozen ? 'shadow-[2px_0_6px_rgba(0,0,0,0.15)]' : '',
            ].filter(Boolean).join(' ')}
          >
            {renderFilter(col)}
          </th>
        )
      })}
    </tr>
  )
}
