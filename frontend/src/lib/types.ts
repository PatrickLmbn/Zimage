export type JobState = "queued" | "running" | "done" | "error"

export interface Job {
  id: string
  state: JobState
  prompt: string
  created_at: number
  started_at: number | null
  finished_at: number | null
  progress: number
  stage: string
  image_url: string | null
  error: string | null
  metrics: {
    generation_seconds: number
    seconds_per_step: number
    peak_vram_gb: number
    rss_mb: number
    seed: number
    size: number
    steps: number
    blocks?: number
  } | null
}

export interface Health {
  status: "ready" | "loading" | "error"
  model: string
  device: string
  error: string | null
  rss_mb: number
  vram_free_gb: number | null
}

export interface Settings {
  sizes: number[]
  default_size: number
  default_steps: number
  max_steps: number
  blocks: number[]
  active_blocks?: number
  notes: Record<string, string>
}

export interface GalleryItem {
  job_id: string
  url: string
  prompt: string
  created_at: number
  metrics: Job["metrics"]
}

export interface GpuStats {
  available: boolean
  name?: string
  util_percent?: number | null
  /** VRAM occupancy: the share of the card that is allocated. */
  memory_percent?: number | null
  /** Memory-controller busy time, NOT VRAM occupancy. */
  memory_controller_percent?: number | null
  vram_total_gb?: number | null
  vram_used_gb?: number | null
  vram_free_gb?: number | null
  temp_c?: number | null
  power_w?: number | null
  clock_mhz?: number | null
}

/** Predicted VRAM outcome of a generation, from the backend's calibrated model. */
export interface HeadroomPrediction {
  size: number
  blocks: number
  /** Peak PyTorch tensor bytes (what the gallery's peak-VRAM metric shows). */
  torch_peak_gb: number
  /** Peak whole-card VRAM (what the System card's VRAM bar shows). */
  device_peak_gb: number
  projected_used_gb: number
  projected_free_gb: number
  /** Which phase sets the peak: "vae" or "transformer". */
  binding: "vae" | "transformer"
  verdict: "won't fit" | "very tight" | "tight" | "ok" | "comfortable"
  fits: boolean
  explain?: string
}

export interface SystemStats {
  ts: number
  cpu_percent: number
  cpu_cores: number | null
  /** Per-logical-CPU busy percentage, index 0 = first thread. */
  cpu_per_core?: number[]
  ram_total_mb: number
  ram_used_mb: number
  ram_available_mb: number
  ram_percent: number
  swap_total_mb: number
  swap_used_mb: number
  swap_percent: number
  total_gb?: number
  used_gb?: number
  free_gb?: number
  percent?: number
  gpu: GpuStats
}