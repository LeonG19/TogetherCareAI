export interface RuntimeEnv {
  BACKEND_URL?: string;
  API_KEY?: string;
  AB_ENABLED?: string;
  AB_EVERY_N?: string;
}

declare global {
  interface Window {
    __ENV?: RuntimeEnv;
  }
}

function runtimeEnv(key: keyof RuntimeEnv): string | undefined {
  return typeof window !== "undefined" ? window.__ENV?.[key] : undefined;
}

export function getBackendUrl(): string {
  const url =
    runtimeEnv("BACKEND_URL") ||
    import.meta.env.VITE_BACKEND_URL ||
    "http://localhost:8000";
  return url.replace(/\/$/, "");
}

export function getApiKey(): string | undefined {
  return runtimeEnv("API_KEY") || import.meta.env.VITE_API_KEY || undefined;
}

// Mirrors the original app's env-var check exactly, including its apparent
// inverted-default bug: A/B mode is ON unless this is set to something other
// than "0". Preserved for behavioral parity with the live app - see README.
export function getAbEnabled(): boolean {
  const v = runtimeEnv("AB_ENABLED") ?? import.meta.env.VITE_AB_ENABLED ?? "0";
  return v === "0";
}

export function getAbEveryN(): number {
  const v = runtimeEnv("AB_EVERY_N") ?? import.meta.env.VITE_AB_EVERY_N ?? "5";
  const n = parseInt(v, 10);
  return Number.isFinite(n) && n > 0 ? n : 5;
}
