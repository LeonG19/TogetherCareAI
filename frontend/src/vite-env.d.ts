/// <reference types="vite/client" />

interface ImportMetaEnv {
  readonly VITE_BACKEND_URL?: string;
  readonly VITE_API_KEY?: string;
  readonly VITE_AB_ENABLED?: string;
  readonly VITE_AB_EVERY_N?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
