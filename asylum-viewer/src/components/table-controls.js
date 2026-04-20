'use client'

import { useState, useRef, useEffect } from 'react'
import { COLUMN_GROUPS, getLabel } from '@/lib/columns'

export default function TableControls({
  columnOrder,
  hiddenColumns,
  onToggleColumn,
  frozenCols,
  frozenRows,
  onFrozenColsChange,
  onFrozenRowsChange,
  visibleCount,
  onResetLayout,
}) {
  const [showColPanel, setShowColPanel] = useState(false)
  const panelRef = useRef(null)

  useEffect(() => {
    const handler = (e) => {
      if (panelRef.current && !panelRef.current.contains(e.target)) {
        setShowColPanel(false)
      }
    }
    document.addEventListener('mousedown', handler)
    return () => document.removeEventListener('mousedown', handler)
  }, [])

  const selectedCount = columnOrder.length - hiddenColumns.size

  return (
    <div className="flex flex-col px-4 sm:px-7 bg-filter-bg border-b border-border text-xs font-mono shrink-0">

      {/* Row 1: Column visibility */}
      <div className="flex items-center py-2 border-b border-border">
      <div className="relative" ref={panelRef}>
        <button
          onClick={() => setShowColPanel(v => !v)}
          className="flex items-center gap-1.5 px-3 py-1.5 border border-border text-text hover:border-accent hover:text-accent transition-colors tracking-wider uppercase"
        >
          <span className="font-semibold">Filter Columns</span>
          <span className="bg-accent text-[#fff] rounded-full w-4 h-4 flex items-center justify-center text-[10px] leading-none">
            {selectedCount}
          </span>
        </button>

        {showColPanel && (
          <div className="absolute top-full left-0 mt-1 bg-surface border border-border shadow-lg z-50 min-w-[200px] max-h-[400px] overflow-y-auto">
            {/* Select all / Deselect all */}
            <div className="flex gap-3 px-3 py-2 border-b border-border bg-filter-bg">
              <button
                onClick={() => columnOrder.forEach(col => hiddenColumns.has(col) && onToggleColumn(col))}
                className="text-[11px] tracking-wider uppercase text-accent hover:underline"
              >
                Select All
              </button>
              <span className="text-muted">|</span>
              <button
                onClick={() => columnOrder.forEach(col => !hiddenColumns.has(col) && onToggleColumn(col))}
                className="text-[11px] tracking-wider uppercase text-accent hover:underline"
              >
                Deselect All
              </button>
            </div>
            {COLUMN_GROUPS.map(group => {
              const groupCols = group.columns.filter(c => columnOrder.includes(c))
              if (groupCols.length === 0) return null
              const allSelected = groupCols.every(c => !hiddenColumns.has(c))
              const noneSelected = groupCols.every(c => hiddenColumns.has(c))
              const groupChecked = allSelected ? true : noneSelected ? false : 'indeterminate'

              const toggleGroup = () => {
                if (allSelected) {
                  groupCols.forEach(c => onToggleColumn(c))
                } else {
                  groupCols.filter(c => hiddenColumns.has(c)).forEach(c => onToggleColumn(c))
                }
              }

              return (
                <div key={group.key} className="border-b-2 border-text/30 last:border-0">
                  {/* Group row */}
                  <label className="flex items-center gap-2 px-3 py-1.5 bg-th-bg hover:bg-row-hover cursor-pointer">
                    <input
                      type="checkbox"
                      checked={groupChecked === true}
                      ref={el => { if (el) el.indeterminate = groupChecked === 'indeterminate' }}
                      onChange={toggleGroup}
                      className="accent-accent w-3 h-3 cursor-pointer"
                    />
                    <span className="text-[10px] tracking-[0.12em] uppercase text-text font-semibold">
                      {group.label}
                    </span>
                    <span className="ml-auto text-muted text-[10px]">
                      {groupCols.filter(c => !hiddenColumns.has(c)).length}/{groupCols.length}
                    </span>
                  </label>
                  {/* Indented column rows */}
                  {groupCols.map(col => (
                    <label
                      key={col}
                      className="flex items-center gap-2 pl-7 pr-3 py-1.5 hover:bg-row-hover cursor-pointer"
                    >
                      <input
                        type="checkbox"
                        checked={!hiddenColumns.has(col)}
                        onChange={() => onToggleColumn(col)}
                        className="accent-accent w-3 h-3 cursor-pointer"
                      />
                      <span className="text-xs text-text tracking-wide">{getLabel(col)}</span>
                    </label>
                  ))}
                </div>
              )
            })}
          </div>
        )}
      </div>
      </div>

      {/* Row 2: Freeze controls + Reset */}
      <div className="flex items-center gap-3 py-2 tracking-wider uppercase">
        <span className="text-text font-semibold">Freeze :</span>
        <span className="text-text">Columns: upto</span>
        <input
          type="number"
          min={0}
          max={visibleCount}
          value={frozenCols}
          onChange={e => onFrozenColsChange(Math.max(0, Math.min(visibleCount, Number(e.target.value))))}
          className="w-10 px-1.5 py-1 bg-surface border border-border text-text text-center outline-none focus:border-accent"
        />
        <span className="text-muted mx-1">,</span>
        <span className="text-text">Rows: upto</span>
        <input
          type="number"
          min={0}
          max={50}
          value={frozenRows}
          onChange={e => onFrozenRowsChange(Math.max(0, Number(e.target.value)))}
          className="w-10 px-1.5 py-1 bg-surface border border-border text-text text-center outline-none focus:border-accent"
        />
      </div>

      {/* Row 3: Reset layout */}
      <div className="flex items-center gap-3 py-2 border-t border-border tracking-wider uppercase">
        <span className="text-text font-semibold">Reset :</span>
        <button
          onClick={onResetLayout}
          className="text-text font-mono text-xs tracking-wider uppercase hover:text-accent transition-colors"
        >
          Reset Layout
        </button>
      </div>
    </div>
  )
}
