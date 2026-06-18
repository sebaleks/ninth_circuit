'use client'

// Controls for the opt-in "Result Cross Verification" feature. Sits between the
// results list and the search input. Three model states drive the buttons:
//   off     -> [Enable Result Cross Verification]
//   loading -> [Loading verifier…] (disabled)
//   ready   -> [Disable]  +  [Cross Verify] / [Stop] (while streaming)
//
// All colour cues are paired with text so they're never colour-only.

const BTN =
  'px-3 py-1.5 text-[11px] font-mono tracking-wider uppercase border transition-colors disabled:opacity-40 disabled:cursor-not-allowed'

export default function VerifyControls({
  verifyModel,
  streaming,
  note,
  canVerify,
  onEnable,
  onDisable,
  onVerify,
  onStop,
}) {
  return (
    <div className="border-t border-border px-3 py-2 space-y-2">
      {verifyModel === 'off' && (
        <button
          type="button"
          onClick={onEnable}
          className={`${BTN} w-full border-border text-text hover:border-accent hover:text-accent`}
          title="Loads the cross-encoder verifier on the server (takes a few seconds)"
        >
          Enable Result Cross Verification
        </button>
      )}

      {verifyModel === 'loading' && (
        <button
          type="button"
          disabled
          className={`${BTN} w-full border-border text-muted animate-pulse`}
        >
          Loading verifier…
        </button>
      )}

      {verifyModel === 'ready' && (
        <div className="flex items-center gap-2">
          {streaming ? (
            <button
              type="button"
              onClick={onStop}
              className={`${BTN} flex-1 border-accent text-accent hover:bg-accent/10`}
            >
              Stop
            </button>
          ) : (
            <button
              type="button"
              onClick={onVerify}
              disabled={!canVerify}
              className={`${BTN} flex-1 border-border text-text hover:border-accent hover:text-accent`}
              title="Re-score the current results with the cross-encoder"
            >
              Cross Verify
            </button>
          )}
          <button
            type="button"
            onClick={onDisable}
            className={`${BTN} border-border text-muted hover:text-text`}
            title="Unload the verifier and free server memory"
          >
            Disable
          </button>
        </div>
      )}

      {streaming && (
        <div className="flex items-center gap-2 text-muted text-[10px] font-mono tracking-wider animate-pulse">
          <span>VERIFYING</span>
          <span className="inline-block w-1 h-1 bg-muted rounded-full" />
          <span className="inline-block w-1 h-1 bg-muted rounded-full" />
          <span className="inline-block w-1 h-1 bg-muted rounded-full" />
        </div>
      )}

      {note && (
        <div className="text-[10px] font-mono tracking-wider text-muted leading-snug">
          {note}
        </div>
      )}
    </div>
  )
}
