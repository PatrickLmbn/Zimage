import { useCallback, useEffect, useState } from "react"
import { CircleAlert, Loader2, Send, Square, Trash2, X, ZoomIn } from "lucide-react"

import { Button } from "@/components/ui/button"
import { Textarea } from "@/components/ui/textarea"
import { Label } from "@/components/ui/label"
import { Slider } from "@/components/ui/slider"
import { Badge } from "@/components/ui/badge"
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog"
import { Progress } from "@/components/ui/progress"
import { Checkbox } from "@/components/ui/checkbox"
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
  AlertDialogTrigger,
} from "@/components/ui/alert-dialog"
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip"
import { Headroom } from "@/components/Headroom"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"
import { cn } from "@/lib/utils"
import type { GalleryItem, Job, Settings } from "@/lib/types"

const PRESETS = [
  "a cute cat sitting on a windowsill",
  "a neon-lit rainy tokyo street at night, reflections on wet asphalt",
  "an old library with towering wooden shelves, dust in sunbeams",
  "a lone astronaut walking across a vast red desert, cinematic",
  "a macro shot of a dew-covered spider web at dawn",
  "a brutalist concrete house on a cliff above the ocean, overcast",
]

const ACTIVE_JOB_KEY = "zit.activeJob"

/** Small circled "i" that explains a control on hover. */
function InfoHint({ text }: { text: string }) {
  return (
    <Tooltip>
      <TooltipTrigger
        render={
          <button
            type="button"
            aria-label="What is this?"
            className="inline-flex shrink-0 items-center text-muted-foreground/60 transition-colors hover:text-foreground focus-visible:ring-2 focus-visible:ring-ring focus-visible:outline-none"
          />
        }
      >
        <CircleAlert className="size-4" />
      </TooltipTrigger>
      <TooltipContent side="right" className="max-w-72 text-sm leading-relaxed">
        {text}
      </TooltipContent>
    </Tooltip>
  )
}

/** Label + optional info icon on one row. */
function FieldLabel({
  htmlFor,
  children,
  hint,
}: {
  htmlFor?: string
  children: React.ReactNode
  hint: string
}) {
  return (
    <div className="flex items-center gap-2">
      <Label htmlFor={htmlFor}>{children}</Label>
      <InfoHint text={hint} />
    </div>
  )
}

export function GeneratePanel({
  settings,
  onDone,
  onJobState,
  disabled,
}: {
  settings: Settings | null
  onDone: (job: Job) => void
  onJobState: (job: Job | null) => void
  disabled: boolean
}) {
  const [prompt, setPrompt] = useState(PRESETS[0])
  const [steps, setSteps] = useState(8)
  const [size, setSize] = useState(512)
  const [seed, setSeed] = useState(-1)
  const [busy, setBusy] = useState(false)
  const [job, setJob] = useState<Job | null>(null)
  const [cancelling, setCancelling] = useState(false)

  const sizes = settings?.sizes ?? [256, 384, 512, 640, 768]
  const maxSteps = settings?.max_steps ?? 50

  // Persist the active job id so a page refresh can reattach to it.
  const setActive = useCallback((id: string | null) => {
    try {
      if (id) window.localStorage.setItem(ACTIVE_JOB_KEY, id)
      else window.localStorage.removeItem(ACTIVE_JOB_KEY)
    } catch {
      /* private mode / storage disabled */
    }
  }, [])

  // On mount: reattach to a job that was running before the refresh.
  useEffect(() => {
    let id: string | null = null
    try {
      id = window.localStorage.getItem(ACTIVE_JOB_KEY)
    } catch {
      return
    }
    if (!id) return
    let cancelled = false
    const reattach = async () => {
      try {
        const res = await fetch(`/api/jobs/${id}`)
        if (!res.ok) {
          setActive(null)
          return
        }
        const j = (await res.json()) as Job
        if (cancelled) return
        setJob(j)
        if (j.state === "queued" || j.state === "running") {
          setBusy(true)
          onJobState(j)
        } else {
          setActive(null)
        }
      } catch {
        /* backend down; nothing to reattach to */
      }
    }
    reattach()
    return () => {
      cancelled = true
    }
  }, [setActive, onJobState])

  // Poll the active job until it reaches a terminal state.
  useEffect(() => {
    if (!job || (job.state !== "queued" && job.state !== "running")) return
    const id = window.setInterval(async () => {
      try {
        const res = await fetch(`/api/jobs/${job.id}`)
        if (!res.ok) return
        const next = (await res.json()) as Job
        setJob(next)
        if (next.state === "done" || next.state === "error") {
          window.clearInterval(id)
          setBusy(false)
          setCancelling(false)
          setActive(null)
          onJobState(null)
          if (next.state === "done") onDone(next)
        } else {
          onJobState(next)
        }
      } catch {
        /* backend restarting; keep polling */
      }
    }, 700)
    return () => window.clearInterval(id)
  }, [job, onDone, onJobState, setActive])

  const submit = useCallback(async () => {
    if (!prompt.trim() || busy) return
    setBusy(true)
    setCancelling(false)
    setJob(null)
    try {
      const res = await fetch("/api/generate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          prompt: prompt.trim(),
          steps,
          size,
          seed,
          guidance: 0,
        }),
      })
      if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: res.statusText }))
        throw new Error(err.detail ?? "request failed")
      }
      const data = (await res.json()) as { job_id: string }
      setActive(data.job_id)
      const queued: Job = {
        id: data.job_id,
        state: "queued",
        prompt: prompt.trim(),
        created_at: Date.now() / 1000,
        started_at: null,
        finished_at: null,
        progress: 0,
        stage: "queued",
        image_url: null,
        error: null,
        metrics: null,
      }
      setJob(queued)
      onJobState(queued)
    } catch (e) {
      setBusy(false)
      setJob({
        id: "err",
        state: "error",
        prompt,
        created_at: Date.now() / 1000,
        started_at: null,
        finished_at: null,
        progress: 0,
        stage: "error",
        image_url: null,
        error: e instanceof Error ? e.message : String(e),
        metrics: null,
      })
    }
  }, [prompt, steps, size, seed, busy, setActive, onJobState])

  // Stop: asks the backend to abort. A running job stops at the next step
  // boundary; a queued job is cancelled immediately.
  const stop = useCallback(async () => {
    if (!job || job.id === "err") return
    setCancelling(true)
    try {
      const res = await fetch(`/api/jobs/${job.id}`, { method: "DELETE" })
      if (!res.ok) {
        const err = await res.json().catch(() => ({}))
        throw new Error(err.detail ?? "could not cancel")
      }
    } catch (e) {
      setCancelling(false)
      setJob((j) =>
        j
          ? {
              ...j,
              error: e instanceof Error ? e.message : String(e),
            }
          : j,
      )
    }
  }, [job])

  const isActive = job?.state === "queued" || job?.state === "running"

  return (
    <div className="flex h-full flex-col gap-6">
      {/* Scrollable control area. The job status and the Generate/Stop button
          stay pinned below it so they are always reachable, even when the
          controls are taller than the viewport. */}
      <div className="flex min-h-0 flex-1 flex-col gap-6 overflow-y-auto pr-1">
      <div className="flex flex-col gap-2">
        <FieldLabel
          htmlFor="prompt"
          hint="The description of the image you want. The text encoder turns this into conditioning tokens for the transformer. Longer, more descriptive prompts usually give more detail. The encode runs on the GPU and takes about a second, so prompt length barely affects the wait."
        >
          Prompt
        </FieldLabel>
        <Textarea
          id="prompt"
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          placeholder="Describe the image you want..."
          rows={6}
          className="resize-none font-mono text-sm leading-relaxed"
          disabled={busy}
        />
        <div className="flex flex-wrap gap-2">
          {PRESETS.map((p) => (
            <button
              key={p}
              type="button"
              onClick={() => setPrompt(p)}
              disabled={busy}
              className={cn(
                "rounded-full border border-border px-3 py-1.5 text-xs text-muted-foreground",
                "transition-colors hover:bg-accent hover:text-accent-foreground",
                "disabled:pointer-events-none disabled:opacity-40",
              )}
            >
              {p.length > 34 ? p.slice(0, 34) + "…" : p}
            </button>
          ))}
        </div>
      </div>

      <div className="flex flex-col gap-2">
        <FieldLabel
          hint="Width and height of the output in pixels. Bigger means more detail but much more VRAM and time — on this 4GB card 512 is the safe maximum, and each step gets slower as this grows."
        >
          Resolution
        </FieldLabel>
        <Select value={String(size)} onValueChange={(v) => setSize(Number(v))} disabled={busy}>
          <SelectTrigger className="w-full">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            {sizes.map((s) => (
              <SelectItem key={s} value={String(s)}>
                {s} × {s}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      </div>

      <Headroom size={size} blocks={1} />

      <div className="flex flex-col gap-3">
        <div className="flex items-center justify-between">
          <FieldLabel
            htmlFor="steps"
            hint="How many denoising passes the model makes. More steps = a cleaner, more finished image, and each step adds about 4 seconds. Z-Image-Turbo is a distilled model, so 8 steps already looks good; going to 20 rarely helps much."
          >
            Steps
          </FieldLabel>
          <Badge variant="secondary" className="font-mono tabular-nums">
            {steps}
          </Badge>
        </div>
        <Slider
          id="steps"
          min={1}
          max={maxSteps}
          step={1}
          value={steps}
          onValueChange={(v) => setSteps(v as number)}
          disabled={busy}
        />
      </div>

      <div className="flex flex-col gap-2">
        <FieldLabel
          htmlFor="seed"
          hint="The random number that starts the generation. The same prompt + same seed + same settings gives the exact same image, so you can reproduce a result you like. Leave it at -1 for a new random seed every time."
        >
          Seed
        </FieldLabel>
        <div className="flex gap-2">
          <input
            id="seed"
            type="number"
            value={seed}
            onChange={(e) => setSeed(Number(e.target.value))}
            disabled={busy}
            className="h-9 w-full rounded-md border border-input bg-transparent px-3 py-1 font-mono text-sm tabular-nums outline-none focus-visible:ring-2 focus-visible:ring-ring disabled:opacity-50"
          />
          <Tooltip>
            <TooltipTrigger
              render={
                <Button
                  type="button"
                  variant="outline"
                  size="icon"
                  onClick={() => setSeed(-1)}
                  disabled={busy}
                />
              }
            >
              <Square className="size-3.5" />
            </TooltipTrigger>
            <TooltipContent>Use a new random seed</TooltipContent>
          </Tooltip>
        </div>
      </div>
      </div>

      {job && (
        <div
          className={cn(
            "flex flex-col gap-2 rounded-lg border p-3",
            job.state === "error" ? "border-destructive/40" : "border-border",
          )}
        >
          <div className="flex items-center justify-between gap-2 text-sm">
            <span
              className={cn(
                "truncate",
                job.state === "error" ? "text-destructive" : "text-muted-foreground",
              )}
            >
              {job.stage}
            </span>
            <span className="font-mono tabular-nums text-muted-foreground">
              {Math.round(job.progress * 100)}%
            </span>
          </div>
          <Progress value={job.progress * 100} />
          {job.metrics && (
            <p className="font-mono text-[11px] tabular-nums text-muted-foreground">
              {job.metrics.total_seconds ?? job.metrics.generation_seconds}s total ·{" "}
              {job.metrics.seconds_per_step}s/step · seed {job.metrics.seed}
            </p>
          )}
          {job.error && (
            <p className="text-sm text-destructive break-words">{job.error}</p>
          )}
        </div>
      )}

      <div className="mt-auto flex gap-2">
        {isActive ? (
          <Button
            className="flex-1"
            variant="outline"
            onClick={stop}
            disabled={cancelling}
          >
            {cancelling ? (
              <>
                <Loader2 className="size-4 animate-spin" />
                Stopping
              </>
            ) : (
              <>
                <X className="size-4" />
                Stop
              </>
            )}
          </Button>
        ) : (
          <Button
            className="flex-1"
            onClick={submit}
            disabled={busy || !prompt.trim() || disabled}
          >
            <Send className="size-4" />
            Generate
          </Button>
        )}
      </div>
    </div>
  )
}

export function GalleryGrid({
  items,
  activeId,
  onChanged,
}: {
  items: GalleryItem[]
  activeId: string | null
  onChanged: () => void
}) {
  const [busyId, setBusyId] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [selected, setSelected] = useState<Set<string>>(new Set())
  const [bulkBusy, setBulkBusy] = useState(false)

  const toggle = (jobId: string, checked: boolean) => {
    setSelected((prev) => {
      const next = new Set(prev)
      if (checked) next.add(jobId)
      else next.delete(jobId)
      return next
    })
  }

  const allSelected = items.length > 0 && items.every((i) => selected.has(i.job_id))

  const toggleAll = (checked: boolean) => {
    setSelected(checked ? new Set(items.map((i) => i.job_id)) : new Set())
  }

  const remove = async (jobId: string) => {
    setBusyId(jobId)
    setError(null)
    try {
      const res = await fetch(`/api/images/${jobId}`, { method: "DELETE" })
      if (!res.ok) {
        const body = await res.json().catch(() => null)
        throw new Error(body?.detail ?? `HTTP ${res.status}`)
      }
      setSelected((prev) => {
        const next = new Set(prev)
        next.delete(jobId)
        return next
      })
      onChanged()
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setBusyId(null)
    }
  }

  const removeSelected = async () => {
    const ids = [...selected]
    if (ids.length === 0) return
    setBulkBusy(true)
    setError(null)
    try {
      const res = await fetch("/api/images/delete", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ job_ids: ids }),
      })
      if (!res.ok) {
        const body = await res.json().catch(() => null)
        throw new Error(body?.detail ?? `HTTP ${res.status}`)
      }
      const result = (await res.json()) as {
        deleted: string[]
        missing: string[]
        busy: string[]
        count: number
      }
      if (result.busy.length > 0) {
        setError(
          `${result.busy.length} image${result.busy.length === 1 ? "" : "s"} still generating — skipped`,
        )
      }
      // Keep the still-running ones selected so the user can retry them.
      setSelected(new Set(result.busy))
      onChanged()
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setBulkBusy(false)
    }
  }

  if (items.length === 0) {
    return (
      <div className="flex h-full min-h-40 items-center justify-center rounded-lg border border-dashed border-border">
        <p className="text-base text-muted-foreground">No images yet</p>
      </div>
    )
  }
  return (
    <div className="flex flex-col gap-3">
      <div className="flex items-center gap-3">
        <label className="flex items-center gap-2 text-xs text-muted-foreground select-none">
          <Checkbox
            checked={allSelected}
            indeterminate={selected.size > 0 && !allSelected}
            onCheckedChange={toggleAll}
            aria-label="Select all images"
          />
          {selected.size > 0
            ? `${selected.size} selected`
            : "Select all"}
        </label>
        {selected.size > 0 && (
          <AlertDialog>
            <AlertDialogTrigger
              render={
                <Button variant="destructive" size="sm" disabled={bulkBusy} />
              }
            >
              {bulkBusy ? (
                <Loader2 className="size-3.5 animate-spin" />
              ) : (
                <Trash2 className="size-3.5" />
              )}
              Delete {selected.size}
            </AlertDialogTrigger>
            <AlertDialogContent>
              <AlertDialogHeader>
                <AlertDialogTitle>
                  Delete {selected.size} image{selected.size === 1 ? "" : "s"}?
                </AlertDialogTitle>
                <AlertDialogDescription>
                  This permanently removes the image files from disk. It cannot be
                  undone.
                </AlertDialogDescription>
              </AlertDialogHeader>
              <AlertDialogFooter>
                <AlertDialogCancel>Cancel</AlertDialogCancel>
                <AlertDialogAction onClick={removeSelected}>
                  Delete
                </AlertDialogAction>
              </AlertDialogFooter>
            </AlertDialogContent>
          </AlertDialog>
        )}
      </div>
      {error && (
        <p className="text-xs text-destructive break-words">{error}</p>
      )}
      <div className="grid grid-cols-1 gap-5 sm:grid-cols-2">
      {items.map((item) => (
        <figure
          key={item.job_id}
          className={cn(
            "group relative overflow-hidden rounded-lg border bg-background transition-colors",
            selected.has(item.job_id)
              ? "border-foreground"
              : item.job_id === activeId
                ? "border-foreground"
                : "border-border",
          )}
        >
          <Dialog>
            <DialogTrigger
              render={
                <button
                  type="button"
                  className="group relative block w-full cursor-zoom-in overflow-hidden"
                  aria-label="View image larger"
                />
              }
            >
              <img
                src={item.url}
                alt={item.prompt || "generated image"}
                className="max-h-64 w-full object-cover"
                loading="lazy"
              />
              <span className="pointer-events-none absolute right-2 bottom-2 flex size-8 items-center justify-center rounded-md border border-border bg-background/80 opacity-0 backdrop-blur transition-opacity group-hover:opacity-100">
                <ZoomIn className="size-4" />
              </span>
            </DialogTrigger>
            <DialogContent className="sm:max-w-[min(96vw,72rem)]">
              <DialogHeader>
                <DialogTitle>
                  {item.prompt || "Generated image"}
                </DialogTitle>
                {item.metrics && (
                  <DialogDescription className="font-mono text-[11px] tabular-nums">
                    {item.metrics.size}px · {item.metrics.steps} steps ·{" "}
                    {item.metrics.seconds_per_step}s/step
                    {item.metrics.total_seconds ? ` · ${item.metrics.total_seconds}s total` : ""} ·{" "}
                    {item.metrics.peak_vram_gb}GB VRAM
                  </DialogDescription>
                )}
              </DialogHeader>
              <img
                src={item.url}
                alt={item.prompt || "generated image"}
                className="mx-auto h-[72vh] w-auto max-w-full rounded-lg object-contain"
              />
            </DialogContent>
          </Dialog>
          <label
            className="absolute top-2 left-2 flex size-8 cursor-pointer items-center justify-center rounded-md border border-border bg-background/80 backdrop-blur transition-opacity group-hover:opacity-100 focus-within:opacity-100"
            onClick={(e) => e.stopPropagation()}
          >
            <Checkbox
              checked={selected.has(item.job_id)}
              onCheckedChange={(checked) => toggle(item.job_id, checked)}
              aria-label={`Select image ${item.job_id}`}
            />
          </label>
          <Button
            variant="secondary"
            size="icon"
            aria-label="Delete image"
            disabled={busyId === item.job_id}
            onClick={() => remove(item.job_id)}
            className="absolute top-2 right-2 size-8 opacity-0 transition-opacity group-hover:opacity-100 focus-visible:opacity-100 disabled:opacity-50"
          >
            {busyId === item.job_id ? (
              <Loader2 className="size-4 animate-spin" />
            ) : (
              <Trash2 className="size-4" />
            )}
          </Button>
          <figcaption className="flex flex-col gap-1.5 px-3.5 py-3">
            <p
              className={cn(
                "line-clamp-2 text-sm leading-relaxed",
                item.prompt ? "" : "text-muted-foreground italic",
              )}
            >
              {item.prompt || "No prompt recorded"}
            </p>
            {item.metrics ? (
              <p className="font-mono text-[11px] text-muted-foreground tabular-nums">
                {item.metrics.size}px · {item.metrics.steps} steps ·{" "}
                {item.metrics.seconds_per_step}s/step
                {item.metrics.total_seconds ? ` · ${item.metrics.total_seconds}s total` : ""} ·{" "}
                {item.metrics.peak_vram_gb}GB VRAM
              </p>
            ) : (
              <p className="font-mono text-[11px] text-muted-foreground tabular-nums">
                no metrics recorded
              </p>
            )}
          </figcaption>
        </figure>
      ))}
      </div>
    </div>
  )
}