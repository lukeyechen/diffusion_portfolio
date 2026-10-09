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
3. Use the same Web client ID in the mobile client and Cloud Build
   `_GOOGLE_OAUTH_CLIENT_IDS` substitution.
4. Store `_ALLOWED_GOOGLE_EMAILS` privately as a Cloud Build trigger
   substitution. Do not commit personal email addresses to the repository.
5. Add `https://lukeyechen.github.io` to the Web OAuth client's Authorized
   JavaScript origins for the PWA.

## iPhone PWA

`.github/workflows/mobile-web.yml` builds the Flutter web app and deploys it to:

`https://lukeyechen.github.io/diffusion_portfolio/`

On iPhone, open that address in Safari, tap **Share**, and select **Add to Home
Screen**. The installed PWA uses Google Sign-In and the same protected Cloud Run
API as the Android app. The web page itself contains no portfolio data; protected
API requests still require an approved Google account.

## Private API allowlist

The committed `cloudbuild.yaml` contains only a safe placeholder. Configure the
real semicolon-separated `_ALLOWED_GOOGLE_EMAILS` value on the
`deploy-diffusion-portfolio` Cloud Build trigger before running a deployment.

## Cloud builds

- `.github/workflows/mobile-android.yml` builds installable Android APK and Play
  Store AAB artifacts on pushes that change `mobile/**`. The artifact includes
  the signing report needed for Android OAuth registration.
- `.github/workflows/mobile-ios-check.yml` is manual-only and compiles the iOS app
  without signing on a hosted macOS runner.
- `.github/workflows/mobile-web.yml` deploys the free iPhone-installable PWA to
  GitHub Pages.

The workflows generate the standard Flutter platform wrappers before compiling,
so a developer does not need Flutter installed just to obtain the build artifact.


## Feasible Tuning

The iPhone PWA and Android share the Feasible Tuning page and authenticated
`/v1/feasible-tuning` endpoint. Run jointly calibrates c and epsilon on data
before the backtest start, freezes the winning pair, and compares Trace tuning
with Classical MV and Old Portfolio. The default replay start is six calendar
months before today. Fixed b, Same-sample ratio, manual c/epsilon/maximum-a
controls, the cap-active column, and the duplicate Trace-versus-old table are
omitted. Latest tuning values use an em dash for unused parameters.

Deploy the updated API as well as the mobile builds before using this page.

Stock entry is always visible under **Enter stocks / download from Yahoo**.
Enter comma- or space-separated ticker symbols, then tap **Download from Yahoo &
run** below calibration settings. The API downloads Yahoo Finance history for
those stocks before calibrating and replaying; no saved Portfolio dataset is
required. This input flow is shared by the iPhone PWA and Android.
