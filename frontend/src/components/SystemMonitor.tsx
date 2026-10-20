import { useEffect, useRef, useState } from "react"
import {
  Activity,
  Cpu,
  Gauge,
  HardDrive,
  MemoryStick,
  Thermometer,
  Zap,
} from "lucide-react"

import { Progress } from "@/components/ui/progress"
import type { SystemStats } from "@/lib/types"
import { cn } from "@/lib/utils"

/** A slim labelled bar. `percent` drives the fill; `value` is the readout. */
function Meter({
  icon,
  label,
  value,
  title,
  percent,
  alert,
}: {
  icon: React.ReactNode
  label: string
  value: string
  title?: string
  percent: number
  alert?: boolean
}) {
  return (
    <div className="flex flex-col gap-0.5" title={title}>
      <div className="flex items-center justify-between gap-2 text-[11px]">
        <span className="flex items-center gap-1.5 text-muted-foreground">
          {icon}
          {label}
        </span>
        <span
          className={cn(
            "font-mono tabular-nums",
            alert ? "text-foreground" : "text-muted-foreground",
          )}
        >
          {value}
        </span>
      </div>
      <Progress value={Math.max(0, Math.min(100, percent))} className="h-1" />
    </div>
  )
}

/** A compact stat tile for single-value readouts. */
function Tile({
  icon,
  label,
  value,
}: {
  icon: React.ReactNode
  label: string
  value: string
}) {
  return (
    <div className="flex items-center gap-1 rounded-md border border-border px-1.5 py-1 text-[10px]">
      <span className="shrink-0 text-muted-foreground">{icon}</span>
      <span className="truncate text-muted-foreground">{label}</span>
      <span className="ml-auto font-mono tabular-nums">{value}</span>
    </div>
  )
}

/**
 * A compact time-series plot. `values` are the samples, `max` the value
 * that maps to the top of the plot. Grows to fill spare card height.
 */
function Sparkline({
  values,
  max,
  format,
  label,
}: {
  values: number[]
  max: number
  format: (v: number) => string
  label: string
}) {
  const latest = values.length > 0 ? values[values.length - 1] : 0
  return (
    <div className="flex min-h-16 flex-1 flex-col gap-0.5">
      <div
        className="flex min-h-6 flex-1 items-end gap-px rounded-sm border-b border-border/60 [background-image:repeating-linear-gradient(to_top,transparent_0,transparent_calc(25%-1px),var(--color-border)_calc(25%-1px),var(--color-border)_25%)]"
      >
        {values.length === 0 ? (
          <div className="h-full w-full animate-pulse rounded-sm bg-muted/50" />
        ) : (
          values.map((v, i) => (
            <div
              key={i}
              className="flex-1 rounded-t-[1px] bg-foreground/70"
              style={{ height: `${Math.max(2, max > 0 ? (v / max) * 100 : 0)}%` }}
              title={`${format(v)}`}
            />
          ))
        )}
      </div>
      <div className="flex items-center justify-between text-[10px] text-muted-foreground">
        <span>{label}</span>
        <span className="font-mono tabular-nums">{format(latest)}</span>
      </div>
    </div>
  )
}

export function SystemMonitor({
  stats,
  className,
}: {
  stats: SystemStats | null
  className?: string
}) {
  // Time-series history for the plots.
  const [history, setHistory] = useState<number[]>([])
  const [ramHistory, setRamHistory] = useState<number[]>([])
  const lastPush = useRef(0)

  useEffect(() => {
    if (!stats) return
    const now = Date.now()
    if (now - lastPush.current < 1000) return
    lastPush.current = now
    setHistory((h) => [...h.slice(-59), stats.gpu?.util_percent ?? 0])
    setRamHistory((h) => [...h.slice(-59), stats.ram_used_mb])
  }, [stats])

  if (!stats) {
    return (
      <div
        className={cn(
          "flex flex-col gap-3 rounded-xl border border-border bg-card p-4",
          className,
        )}
      >
        <div className="flex items-center gap-1.5 text-xs text-muted-foreground">
          <Activity className="size-3.5" />
          System
        </div>
        <div className="h-20 animate-pulse rounded-md bg-muted/50" />
      </div>
    )
  }

  const gpu = stats.gpu
  const perCore = stats.cpu_per_core ?? []
  const coreCols = perCore.length > 12 ? 16 : 8

  return (
    <div
      className={cn(
        "flex min-h-0 flex-col gap-3 rounded-xl border border-border bg-card p-4",
        className,
      )}
    >
      <div className="flex items-center justify-between">
        <span className="flex items-center gap-1.5 text-xs font-medium">
          <Activity className="size-3.5" />
          System
        </span>
        <span className="font-mono text-[10px] tabular-nums text-muted-foreground">
          {gpu?.name ?? "no GPU"}
        </span>
      </div>

      {/* Time-series plots. These grow to absorb spare card height so the
          panel fills the column instead of leaving dead space at the bottom. */}
      <Sparkline
        values={history}
        max={100}
        format={(v) => `${v.toFixed(0)}%`}
        label={`GPU load · last ${history.length}s`}
      />
      <Sparkline
        values={ramHistory}
        max={Math.max(stats.ram_total_mb, ...ramHistory, 1)}
        format={(v) => `${(v / 1024).toFixed(1)} GB`}
        label={`RAM · last ${ramHistory.length}s`}
      />

      <div className="flex flex-1 flex-col justify-between gap-3">
        <Meter
          icon={<Cpu className="size-3" />}
          label={`CPU · ${stats.cpu_cores ?? "?"} threads`}
          value={`${stats.cpu_percent.toFixed(0)}%`}
          percent={stats.cpu_percent}
        />

        <Meter
          icon={<MemoryStick className="size-3" />}
          label="RAM"
          value={`${(stats.ram_used_mb / 1024).toFixed(1)} / ${(
            stats.ram_total_mb / 1024
          ).toFixed(1)} GB`}
          title={`${(stats.ram_available_mb / 1024).toFixed(1)} GB available`}
          percent={stats.ram_percent}
          alert={stats.ram_percent > 90}
        />

        <Meter
          icon={<HardDrive className="size-3" />}
          label="Swap"
          value={`${(stats.swap_used_mb / 1024).toFixed(1)} / ${(
            stats.swap_total_mb / 1024
          ).toFixed(1)} GB`}
          title="zram — high usage means RAM pressure"
          percent={stats.swap_percent}
          alert={stats.swap_percent > 75}
        />

        {gpu?.available && (
          <Meter
            icon={<Gauge className="size-3.5" />}
            label="VRAM"
            value={`${gpu.vram_used_gb?.toFixed(2) ?? "?"} / ${
              gpu.vram_total_gb?.toFixed(2) ?? "?"
            } GB`}
            title={`${gpu.vram_free_gb?.toFixed(2) ?? "?"} GB free · ${
              gpu.memory_percent?.toFixed(0) ?? "?"
            }% of the card in use`}
            percent={gpu.memory_percent ?? 0}
          />
        )}

        <div className="grid grid-cols-3 gap-1.5 text-[10px]">
          <Tile
            icon={<Thermometer className="size-3" />}
            label="Temp"
            value={gpu?.temp_c != null ? `${gpu.temp_c.toFixed(0)}°C` : "—"}
          />
          <Tile
            icon={<Zap className="size-3" />}
            label="Power"
            value={gpu?.power_w != null ? `${gpu.power_w.toFixed(0)}W` : "—"}
          />
          <Tile
            icon={<HardDrive className="size-3" />}
            label="Disk"
            value={
              stats.free_gb != null ? `${stats.free_gb.toFixed(0)}G` : "—"
            }
          />
        </div>
      </div>

      {/* Per-core CPU load: one bar per logical thread. */}
      {perCore.length > 0 && (
        <div className="flex flex-col gap-1">
          <div className="flex items-center justify-between text-[10px] text-muted-foreground">
            <span>CPU per core</span>
            <span className="font-mono tabular-nums">
              {perCore.filter((v) => v > 50).length}/{perCore.length} busy
            </span>
          </div>
          <div
            className="grid gap-1"
            style={{ gridTemplateColumns: `repeat(${coreCols}, minmax(0, 1fr))` }}
          >
            {perCore.map((v, i) => (
              <div
                key={i}
                className="relative h-6 overflow-hidden rounded-[2px] bg-muted/60"
                title={`core ${i}: ${v.toFixed(0)}%`}
              >
                <div
                  className="absolute inset-x-0 bottom-0 rounded-[2px] bg-foreground/70"
                  style={{ height: `${Math.max(2, v)}%` }}
                />
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Disk usage */}
      {stats.total_gb != null && stats.used_gb != null && (
        <Meter
          icon={<HardDrive className="size-3" />}
          label="Disk"
          value={`${stats.used_gb.toFixed(0)} / ${stats.total_gb.toFixed(0)} GB`}
          percent={stats.percent ?? 0}
        />
      )}
    </div>
  )
}
