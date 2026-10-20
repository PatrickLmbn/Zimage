import { useCallback, useEffect, useState } from "react"
import { Moon, Sun, Image as ImageIcon } from "lucide-react"

import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { ScrollArea } from "@/components/ui/scroll-area"
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip"
import { GeneratePanel, GalleryGrid } from "@/components/GeneratePanel"
import { SystemMonitor } from "@/components/SystemMonitor"
import type { GalleryItem, Health, Job, Settings, SystemStats } from "@/lib/types"

function useTheme() {
  const [dark, setDark] = useState(() => {
    const saved = localStorage.getItem("theme")
    if (saved) return saved === "dark"
    return window.matchMedia("(prefers-color-scheme: dark)").matches
  })
  useEffect(() => {
    document.documentElement.classList.toggle("dark", dark)
    localStorage.setItem("theme", dark ? "dark" : "light")
  }, [dark])
  return { dark, toggle: () => setDark((d) => !d) }
}

export default function App() {
  const { dark, toggle } = useTheme()
  const [health, setHealth] = useState<Health | null>(null)
  const [settings, setSettings] = useState<Settings | null>(null)
  const [gallery, setGallery] = useState<GalleryItem[]>([])
  const [activeId, setActiveId] = useState<string | null>(null)
  const [system, setSystem] = useState<SystemStats | null>(null)
  const [activeJob, setActiveJob] = useState<Job | null>(null)

  const refreshGallery = useCallback(async () => {
    try {
      const res = await fetch("/api/gallery")
      if (res.ok) setGallery((await res.json()) as GalleryItem[])
    } catch {
      /* backend not up yet */
    }
  }, [])

  // Poll health while the model loads, then keep it fresh.
  useEffect(() => {
    let alive = true
    const tick = async () => {
      try {
        const res = await fetch("/api/health")
        if (res.ok && alive) {
          const h = (await res.json()) as Health
          setHealth(h)
          if (h.status === "ready" || h.status === "error") refreshGallery()
        }
      } catch {
        if (alive) setHealth((h) => h ?? null)
      }
    }
    tick()
    const id = window.setInterval(tick, 2000)
    return () => {
      alive = false
      window.clearInterval(id)
    }
  }, [refreshGallery])

  // Realtime system telemetry. 1 Hz keeps it readable without hammering the box.
  useEffect(() => {
    let alive = true
    const tick = async () => {
      try {
        const res = await fetch("/api/system")
        if (res.ok && alive) setSystem((await res.json()) as SystemStats)
      } catch {
        /* backend not up yet */
      }
    }
    tick()
    const id = window.setInterval(tick, 1000)
    return () => {
      alive = false
      window.clearInterval(id)
    }
  }, [])

  useEffect(() => {
    fetch("/api/settings")
      .then((r) => (r.ok ? r.json() : null))
      .then((s) => setSettings(s as Settings | null))
      .catch(() => {})
  }, [])

  const onDone = useCallback(
    (job: Job) => {
      setActiveId(job.id)
      refreshGallery()
    },
    [refreshGallery],
  )

  const ready = health?.status === "ready"
  const loading = health?.status === "loading" || health === null

  return (
    <TooltipProvider>
      <div className="flex min-h-screen flex-col bg-background lg:h-screen lg:overflow-hidden">
        <header className="sticky top-0 z-10 border-b border-border bg-background/80 backdrop-blur">
          <div className="mx-auto flex h-14 max-w-screen-2xl items-center gap-3 px-6">
            <ImageIcon className="size-4 text-muted-foreground" />
            <h1 className="text-sm font-medium tracking-tight">Z-Image-Turbo</h1>
            <Badge variant="outline" className="font-mono text-[10px]">
              INT4 · sdnq
            </Badge>

            <div className="ml-auto flex items-center gap-2">
              {loading && (
                <Badge variant="secondary" className="gap-1.5 text-[10px]">
                  <span className="size-1.5 animate-pulse rounded-full bg-current" />
                  loading model
                </Badge>
              )}
              {ready && activeJob && (
                <Badge variant="secondary" className="gap-1.5 text-[10px]">
                  <span className="size-1.5 animate-pulse rounded-full bg-current" />
                  generating
                </Badge>
              )}
              {ready && !activeJob && (
                <Badge variant="secondary" className="text-[10px]">
                  ready
                </Badge>
              )}
              {health?.status === "error" && (
                <Badge variant="destructive" className="text-[10px]">
                  error
                </Badge>
              )}
              <Tooltip>
                <TooltipTrigger
                  render={
                    <Button variant="ghost" size="icon" onClick={toggle} />
                  }
                >
                  {dark ? <Sun className="size-4" /> : <Moon className="size-4" />}
                </TooltipTrigger>
                <TooltipContent>Toggle theme</TooltipContent>
              </Tooltip>
            </div>
          </div>
        </header>

        <main className="mx-auto grid w-full max-w-screen-2xl min-h-0 flex-1 grid-cols-1 gap-6 px-6 py-6 lg:grid-cols-[380px_minmax(0,1fr)_340px] lg:grid-rows-[minmax(0,1fr)] lg:overflow-hidden">
          {/* Left: generation controls */}
          <aside className="flex min-h-0 lg:pr-1">
            <div className="flex min-h-0 flex-1 flex-col rounded-xl border border-border bg-card p-5">
              {health?.status === "error" ? (
                <div className="flex flex-col gap-3">
                  <p className="text-sm font-medium">Model failed to load</p>
                  <p className="text-xs break-words text-destructive">{health.error}</p>
                </div>
              ) : (
                <GeneratePanel
                  settings={settings}
                  onDone={onDone}
                  onJobState={setActiveJob}
                  disabled={!ready}
                />
              )}
            </div>
          </aside>

          {/* Centre: gallery */}
          <section className="flex min-h-0 min-w-0 flex-col rounded-xl border border-border bg-card p-5">
            <div className="mb-4 flex items-center justify-between">
              <h2 className="text-base font-medium">Gallery</h2>
              <Button variant="ghost" size="sm" onClick={refreshGallery}>
                Refresh
              </Button>
            </div>
            <ScrollArea className="min-h-0 flex-1 pr-3">
              <GalleryGrid items={gallery} activeId={activeId} />
            </ScrollArea>
          </section>

          {/* Right: system monitor */}
          <aside className="flex min-h-0 lg:pl-1">
            <SystemMonitor stats={system} className="h-full flex-1" />
          </aside>
        </main>

        <footer className="mx-auto w-full max-w-screen-2xl shrink-0 px-6 pb-4">
          <p className="text-[11px] text-muted-foreground">
            Text encoder stays resident on CPU (one load, no per-request cost) · only N
            transformer blocks are resident in VRAM at a time
          </p>
        </footer>
      </div>
    </TooltipProvider>
  )
}