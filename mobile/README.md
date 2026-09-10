# Diffusion Portfolio Mobile

Flutter client for the authenticated `diffusion-portfolio-api` Cloud Run service.

## Security model

The user taps **Sign in with Google**. The mobile app sends Google's signed
OpenID Connect ID token to the API over HTTPS. FastAPI verifies the token's
signature, issuer, audience, expiry, verified email, and account identifier,
then enforces the `ALLOWED_GOOGLE_EMAILS` allowlist. No password,
service-account key, or long-lived token is compiled into the app.

The API URL is still obtained with:

```bash
gcloud run services describe diffusion-portfolio-api \
  --region=us-east1 \
  --format='value(status.url)'
```

Google Cloud Run must allow requests to reach FastAPI so FastAPI can perform
end-user authentication. Sensitive `/v1/**` endpoints reject missing, invalid,
expired, or unauthorized Google ID tokens.

## OAuth configuration

1. Register Android package `com.lukeyechen.diffusion_portfolio_mobile` with the
   release signing SHA-1 recorded in the Android build artifact.
2. Create a Web OAuth client for the backend.
3. Set GitHub Actions variable `GOOGLE_SERVER_CLIENT_ID` to the Web client ID.
4. Set Cloud Build substitution `_GOOGLE_OAUTH_CLIENT_IDS` to the same Web
   client ID.
5. Keep `_ALLOWED_GOOGLE_EMAILS=st.yeyo@gmail.com` unless access should change.

## Cloud builds

- `.github/workflows/mobile-android.yml` builds installable Android APK and Play
  Store AAB artifacts on pushes that change `mobile/**`. The artifact includes
  the signing report needed for Android OAuth registration.
- `.github/workflows/mobile-ios-check.yml` is manual-only and compiles the iOS app
  without signing on a hosted macOS runner.

The workflows generate the standard Flutter platform wrappers before compiling,
so a developer does not need Flutter installed just to obtain the build artifact.
