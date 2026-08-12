// Cloudflare Worker entrypoint for the Containers deployment. Routes every
// request to a Durable-Object-managed instance of the frontend Docker image.
//
// NOTE: Cloudflare Containers is a newer product and its API has moved fast.
// This targets the `@cloudflare/containers` SDK pattern documented at
// https://developers.cloudflare.com/containers/ as of early 2026 - verify
// against the current docs before deploying, since it may have changed since.
import { Container, getContainer } from "@cloudflare/containers";

interface Env {
  FRONTEND_CONTAINER: DurableObjectNamespace<FrontendContainer>;
  BACKEND_URL?: string;
  API_KEY?: string;
  AB_ENABLED?: string;
  AB_EVERY_N?: string;
}

export class FrontendContainer extends Container<Env> {
  defaultPort = 8080;
  sleepAfter = "10m";

  // Forwarded into the container process as real env vars, which
  // docker/entrypoint.sh then writes into dist/env-config.js at startup.
  get envVars() {
    return {
      BACKEND_URL: this.env.BACKEND_URL ?? "",
      API_KEY: this.env.API_KEY ?? "",
      AB_ENABLED: this.env.AB_ENABLED ?? "0",
      AB_EVERY_N: this.env.AB_EVERY_N ?? "5",
    };
  }
}

export default {
  async fetch(request: Request, env: Env): Promise<Response> {
    const container = getContainer(env.FRONTEND_CONTAINER);
    return container.fetch(request);
  },
};
