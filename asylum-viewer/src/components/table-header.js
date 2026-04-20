'use client'

import { useState } from 'react'
import { COLUMN_GROUPS, getLabel, getColumnWidth } from '@/lib/columns'

export default function TableHeader({ columns, frozenCols = 0, columnLeftOffsets = {}, onColumnReorder }) {
  const [dragCol, setDragCol] = useState(null)
  const [dragOverCol, setDragOverCol] = useState(null)

  const handleDragStart = (e, col) => {
    setDragCol(col)
    e.dataTransfer.effectAllowed = 'move'
  }

  const handleDragOver = (e, col) => {
    e.preventDefault()
    e.dataTransfer.dropEffect = 'move'
    setDragOverCol(col)
  }

  const handleDrop = (e, col) => {
    e.preventDefault()
    if (dragCol && dragCol !== col) onColumnReorder(dragCol, col)
    setDragCol(null)
    setDragOverCol(null)
  }

  const handleDragEnd = () => {
    setDragCol(null)
    setDragOverCol(null)
  }

  return (
    <>
      {/* Group header row */}
      <tr>
        {COLUMN_GROUPS.map(group => {
          const visibleCols = group.columns.filter(c => columns.includes(c))
          if (visibleCols.length === 0) return null
          return (
            <th
              key={group.key}
              colSpan={visibleCols.length}
              className="bg-header-bg text-header-text px-5 py-3 text-left font-mono text-xs font-semibold tracking-[0.12em] uppercase border-b border-border border-r border-r-[#2e2c28] whitespace-nowrap"
            >
              {group.label}
            </th>
          )
        })}
      </tr>

      {/* Individual column labels — draggable to reorder */}
      <tr>
        {columns.map((col, i) => {
          const isSticky = i < frozenCols
          const isLastFrozen = frozenCols > 0 && i === frozenCols - 1
          const isDragging = dragCol === col
          const isDragOver = dragOverCol === col && dragCol !== col

          return (
            <th
              key={col}
              draggable
              onDragStart={e => handleDragStart(e, col)}
              onDragOver={e => handleDragOver(e, col)}
              onDrop={e => handleDrop(e, col)}
              onDragEnd={handleDragEnd}
              style={{
                minWidth: getColumnWidth(col),
                ...(isSticky ? { position: 'sticky', left: columnLeftOffsets[col], zIndex: 20 } : {}),
              }}
              className={[
                'bg-th-bg px-5 py-3 text-left font-mono text-xs font-medium tracking-[0.06em] uppercase text-text border-b border-border border-r border-r-border whitespace-nowrap',
                'cursor-grab select-none transition-opacity',
                isDragging ? 'opacity-30' : '',
                isDragOver ? 'border-l-2 border-l-accent' : '',
                isLastFrozen ? 'shadow-[2px_0_6px_rgba(0,0,0,0.15)]' : '',
              ].filter(Boolean).join(' ')}
            >
              <span className="flex items-center gap-1.5">
                <span className="text-muted opacity-40 text-[10px]">⠿</span>
                {getLabel(col)}
              </span>
            </th>
          )
        })}
      </tr>
    </>
  )
}
