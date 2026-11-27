# openstudio-worker

RunPod serverless worker for Openstudio: one open model per endpoint (FLUX.1 Schnell for images, Wan 2.1 1.3B for video), optional LoRA, outputs saved to private Supabase Storage.

Built for linux/amd64 by GitHub Actions and published to `ghcr.io/<owner>/openstudio-worker`. Configuration (including Supabase credentials) is set as environment variables on the RunPod endpoint, never in this repo.

FLUX 3 Image Klein is built separately from `Dockerfile.flux3`. It needs `HF_TOKEN` for an account that accepted the model terms; the FLUX Kommunity License applies to the weights.
