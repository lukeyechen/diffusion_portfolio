# Google Cloud Run deployment

This repository now contains a lightweight FastAPI backend for the upgraded short-horizon portfolio engine. The Cloud Run image intentionally installs `requirements-api.txt` instead of the full Streamlit/torch stack so cold starts and storage are smaller.

## One-time Google Cloud setup

1. Create or select a Google Cloud project and enable billing.
2. Install the Google Cloud CLI and sign in.
3. Enable the required APIs:

```bash
gcloud services enable run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com
```

## Deploy from the repository root

Set your project and region:

```bash
gcloud config set project YOUR_PROJECT_ID
gcloud config set run/region us-east1
```

Deploy directly from source:

```bash
gcloud run deploy diffusion-portfolio-api \
  --source . \
  --allow-unauthenticated \
  --cpu 1 \
  --memory 2Gi \
  --min-instances 0 \
  --max-instances 2 \
  --concurrency 4 \
  --timeout 900
```

Cloud Run will build the root `Dockerfile` and return a service URL such as:

```text
https://diffusion-portfolio-api-XXXXXXXXXX-ue.a.run.app
```

Test the health endpoint:

```bash
curl https://YOUR_SERVICE_URL/health
```

Interactive OpenAPI documentation is available at:

```text
https://YOUR_SERVICE_URL/docs
```

## Portfolio request

```bash
curl -X POST "https://YOUR_SERVICE_URL/portfolio/recommendation" \
  -H "Content-Type: application/json" \
  -d '{
    "tickers": ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN"],
    "holding_period": "1 week",
    "gamma": 3.0,
    "max_weight": 0.40,
    "turnover_penalty_bps": 25.0
  }'
```

For the validated defaults, `rebalance_step_pct` may be omitted. The API uses 50% for one week and 100% for other horizons; the validated short-horizon presets are one week and two weeks.

## Security before public release

The prototype allows unauthenticated calls and permissive CORS. Before a public App Store release, add authentication or an API gateway, rate limiting, restricted browser origins if needed, logging/monitoring, and any required licensed-data controls. Never ship Bloomberg credentials inside the iPhone app.
