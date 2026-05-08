import { VISIBLE_COLUMNS } from './columns'

const DEFAULT_VISIBLE = [
  'link',
  'published_status',
  'date_filed',
  'country_of_origin',
  'final_disposition',
]

export const DEFAULT_LAYOUT = {
  columnOrder: [
    ...DEFAULT_VISIBLE,
    ...VISIBLE_COLUMNS.filter(c => !DEFAULT_VISIBLE.includes(c)),
  ],
  hiddenColumns: VISIBLE_COLUMNS.filter(c => !DEFAULT_VISIBLE.includes(c)),
  frozenCols: 0,
  frozenRows: 0,
}

export async function loadPreferences(supabase) {
  const { data: { user } } = await supabase.auth.getUser()
  if (!user) return DEFAULT_LAYOUT

  const { data, error } = await supabase
    .from('user_preferences')
    .select('table_layout')
    .eq('user_id', user.id)
    .single()

  if (error || !data?.table_layout) return DEFAULT_LAYOUT

  const saved = data.table_layout

  // Merge new columns (added since last save) into saved order at their natural position
  const savedSet = new Set(saved.columnOrder ?? [])
  const mergedOrder = [...(saved.columnOrder ?? [])]
  VISIBLE_COLUMNS.forEach((col, i) => {
    if (!savedSet.has(col)) {
      const insertAt = Math.min(i, mergedOrder.length)
      mergedOrder.splice(insertAt, 0, col)
    }
  })

  return {
    columnOrder: mergedOrder,
    hiddenColumns: saved.hiddenColumns ?? DEFAULT_LAYOUT.hiddenColumns,
    frozenCols: saved.frozenCols ?? DEFAULT_LAYOUT.frozenCols,
    frozenRows: saved.frozenRows ?? DEFAULT_LAYOUT.frozenRows,
  }
}

export async function savePreferences(supabase, layout) {
  const { data: { user } } = await supabase.auth.getUser()
  if (!user) return

  await supabase
    .from('user_preferences')
    .upsert(
      { user_id: user.id, table_layout: layout, updated_at: new Date().toISOString() },
      { onConflict: 'user_id' }
    )
}
