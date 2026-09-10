# Diffusion Portfolio Mobile

Flutter client for the private `diffusion-portfolio-api` Cloud Run service.

## Security model for the first test build

The Cloud Run service remains IAM-private. The app accepts a short-lived Google
identity token at runtime and keeps it only in memory. No service-account key,
password, or long-lived token is compiled into the app.

In Google Cloud Shell, obtain the service URL and a short-lived token:

```bash
gcloud run services describe diffusion-portfolio-api \
  --region=us-east1 \
  --format='value(status.url)'

gcloud auth print-identity-token
```

Paste those values into the locked connection panel in the app. The next
authentication milestone is interactive Google sign-in with server-side token
verification.

## Cloud builds

- `.github/workflows/mobile-android.yml` builds installable Android APK and Play
  Store AAB artifacts on pushes that change `mobile/**`.
- `.github/workflows/mobile-ios-check.yml` is manual-only and compiles the iOS app
  without signing on a hosted macOS runner.

The workflows generate the standard Flutter platform wrappers before compiling,
so a developer does not need Flutter installed just to obtain the build artifact.
