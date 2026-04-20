'use client'

import { useState } from 'react'
import { COLUMN_GROUPS, getLabel, getColumnWidth } from '@/lib/columns'

function getColGroupKey(col) {
  return COLUMN_GROUPS.find(g => g.columns.includes(col))?.key ?? null
}

export default function TableHeader({ columns, frozenCols = 0, columnLeftOffsets = {}, onColumnReorder, onGroupReorder }) {
  const [dragCol, setDragCol] = useState(null)
  const [dragOverCol, setDragOverCol] = useState(null)
  const [dragGroup, setDragGroup] = useState(null)
  const [dragOverGroup, setDragOverGroup] = useState(null)

  // --- Column drag (within same group only) ---
  const handleColDragStart = (e, col) => {
    e.stopPropagation()
    setDragCol(col)
    e.dataTransfer.effectAllowed = 'move'
  }

  const handleColDragOver = (e, col) => {
    e.preventDefault()
    e.stopPropagation()
    if (dragCol && getColGroupKey(dragCol) === getColGroupKey(col)) {
      e.dataTransfer.dropEffect = 'move'
      setDragOverCol(col)
    } else {
      e.dataTransfer.dropEffect = 'none'
    }
  }

  const handleColDrop = (e, col) => {
    e.preventDefault()
    e.stopPropagation()
    if (dragCol && dragCol !== col && getColGroupKey(dragCol) === getColGroupKey(col)) {
      onColumnReorder(dragCol, col)
    }
    setDragCol(null)
    setDragOverCol(null)
  }

  const handleColDragEnd = () => {
    setDragCol(null)
    setDragOverCol(null)
  }

  // --- Group drag ---
  const handleGroupDragStart = (e, groupKey) => {
    setDragGroup(groupKey)
    e.dataTransfer.effectAllowed = 'move'
  }

  const handleGroupDragOver = (e, groupKey) => {
    e.preventDefault()
    if (dragGroup && dragGroup !== groupKey) {
      e.dataTransfer.dropEffect = 'move'
      setDragOverGroup(groupKey)
    }
  }

  const handleGroupDrop = (e, groupKey) => {
    e.preventDefault()
    if (dragGroup && dragGroup !== groupKey) onGroupReorder(dragGroup, groupKey)
    setDragGroup(null)
    setDragOverGroup(null)
  }

  const handleGroupDragEnd = () => {
    setDragGroup(null)
    setDragOverGroup(null)
  }

  return (
    <>
      {/* Group header row — draggable to reorder groups */}
      <tr>
        {COLUMN_GROUPS.map(group => {
          const visibleCols = group.columns.filter(c => columns.includes(c))
          if (visibleCols.length === 0) return null
          const isDraggingGroup = dragGroup === group.key
          const isDragOverGroup = dragOverGroup === group.key && dragGroup !== group.key
          return (
            <th
              key={group.key}
              colSpan={visibleCols.length}
              draggable
              onDragStart={e => handleGroupDragStart(e, group.key)}
              onDragOver={e => handleGroupDragOver(e, group.key)}
              onDrop={e => handleGroupDrop(e, group.key)}
              onDragEnd={handleGroupDragEnd}
              className={[
                'bg-header-bg text-header-text px-5 py-3 text-left font-mono text-xs font-semibold tracking-[0.12em] uppercase border-b border-border border-r border-r-[#2e2c28] whitespace-nowrap',
                'cursor-grab select-none transition-opacity',
                isDraggingGroup ? 'opacity-30' : '',
                isDragOverGroup ? 'border-l-2 border-l-accent' : '',
              ].filter(Boolean).join(' ')}
            >
              <span className="flex items-center gap-1.5">
                <span className="opacity-40 text-[10px]">⠿</span>
                {group.label}
              </span>
            </th>
          )
        })}
      </tr>

      {/* Individual column labels — draggable within same group */}
      <tr>
        {columns.map((col, i) => {
          const isSticky = i < frozenCols
          const isLastFrozen = frozenCols > 0 && i === frozenCols - 1
          const isDragging = dragCol === col
          const isDragOver = dragOverCol === col && dragCol !== col
          const sameGroup = dragCol ? getColGroupKey(dragCol) === getColGroupKey(col) : true
          const isDroppable = dragCol && dragCol !== col && sameGroup

          return (
            <th
              key={col}
              draggable
              onDragStart={e => handleColDragStart(e, col)}
              onDragOver={e => handleColDragOver(e, col)}
              onDrop={e => handleColDrop(e, col)}
              onDragEnd={handleColDragEnd}
              style={{
                minWidth: getColumnWidth(col),
                ...(isSticky ? { position: 'sticky', left: columnLeftOffsets[col], zIndex: 20 } : {}),
              }}
              className={[
                'bg-th-bg px-5 py-3 text-left font-mono text-xs font-medium tracking-[0.06em] uppercase text-text border-b border-border border-r border-r-border whitespace-nowrap',
                'cursor-grab select-none transition-opacity',
                isDragging ? 'opacity-30' : '',
                dragCol && !sameGroup ? 'opacity-40 cursor-not-allowed' : '',
                isDragOver && isDroppable ? 'border-l-2 border-l-accent' : '',
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
