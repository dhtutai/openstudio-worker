# openstudio-worker

RunPod serverless worker for Openstudio: one open model per endpoint (FLUX.1 Schnell for images, Wan 2.1 1.3B for video), optional LoRA, outputs saved to private Supabase Storage.

Built for linux/amd64 by GitHub Actions and published to `ghcr.io/<owner>/openstudio-worker`. Configuration (including Supabase credentials) is set as environment variables on the RunPod endpoint, never in this repo.
