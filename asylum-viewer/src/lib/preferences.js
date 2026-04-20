import { VISIBLE_COLUMNS } from './columns'

export const DEFAULT_LAYOUT = {
  columnOrder: VISIBLE_COLUMNS,
  hiddenColumns: [],
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
  return {
    columnOrder: saved.columnOrder ?? DEFAULT_LAYOUT.columnOrder,
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
