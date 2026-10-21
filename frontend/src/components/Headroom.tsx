import { useEffect, useState } from "react"
import { Gauge } from "lucide-react"

import { Progress } from "@/components/ui/progress"
import { cn } from "@/lib/utils"
import type { HeadroomPrediction } from "@/lib/types"

/**
 * Live VRAM headroom preview for the current resolution / block choice.
 *
 * The numbers come from the backend's calibrated model, which was measured on
 * this machine by running the real transformer forward and the real VAE decode
 * for every (size, blocks) combination.
 */
export function Headroom({
  size,
  blocks,
}: {
  size: number
  blocks: number
}) {
  const [pred, setPred] = useState<HeadroomPrediction | null>(null)
  const [error, setError] = useState(false)

  useEffect(() => {
    let cancelled = false
    const url = `/api/headroom?size=${size}&blocks=${blocks}`
    fetch(url)
      .then((r) => (r.ok ? r.json() : Promise.reject(new Error(String(r.status)))))
      .then((d) => {
        if (!cancelled) {
          setPred(d)
          setError(false)
        }
      })
      .catch(() => {
        if (!cancelled) setError(true)
      })
    return () => {
      cancelled = true
    }
  }, [size, blocks])

  if (error) return null
  if (!pred) return null

  const pctUsed = pred.device_peak_gb / Math.max(pred.device_peak_gb + Math.max(pred.projected_free_gb, 0), 0.001)
  const warn = pred.verdict === "won't fit" || pred.verdict === "very tight"

  return (
    <div className="flex flex-col gap-2 rounded-lg border border-border bg-muted/30 p-3">
      <div className="flex items-center justify-between">
        <span className="flex items-center gap-1.5 text-xs font-medium text-muted-foreground">
          <Gauge className="size-3.5" />
          VRAM headroom
        </span>
        <span
          className={cn(
            "font-mono text-[11px] tabular-nums",
            warn ? "text-foreground" : "text-muted-foreground",
          )}
        >
          {pred.verdict}
        </span>
      </div>

      <Progress value={pctUsed * 100} className="h-1.5" />

      <div className="flex flex-wrap items-baseline justify-between gap-x-3 gap-y-0.5 font-mono text-[11px] tabular-nums text-muted-foreground">
        <span>needs {pred.device_peak_gb.toFixed(2)} GB</span>
        <span>free after: {pred.projected_free_gb.toFixed(2)} GB</span>
        <span>tensors {pred.torch_peak_gb.toFixed(2)} GB</span>
      </div>

      <div className="flex flex-wrap items-baseline justify-between gap-x-3 gap-y-0.5 font-mono text-[11px] tabular-nums text-muted-foreground">
        <span className={pred.encode_fits ? undefined : "text-destructive"}>
          encode needs {pred.encode_peak_gb.toFixed(2)} GB
        </span>
        <span className={pred.encode_fits ? undefined : "text-destructive"}>
          free after: {pred.encode_free_gb.toFixed(2)} GB
        </span>
      </div>

      <p className="text-[11px] leading-relaxed text-muted-foreground/80">
        {!pred.encode_fits
          ? "The prompt encoder alone does not fit on the card right now — close something else using VRAM, or the job will fail at the encode step."
          : pred.binding === "vae"
            ? "The VAE decode is the limit here, not the transformer — a larger resolution is what costs you, not the offload settings."
            : "The transformer dominates, so a larger resolution is what costs you here."}
        {warn && " This is very close to the card limit."}
      </p>
    </div>
  )
}
