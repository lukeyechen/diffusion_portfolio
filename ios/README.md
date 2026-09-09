# iPhone SwiftUI client

This folder is the starter source for the native iPhone client. It is designed to call the same FastAPI backend used by the web/research stack, so the iPhone does not reimplement the diffusion mathematics.

## Xcode setup

1. On a Mac, create a new **iOS App** project named `DiffusionPortfolio` using SwiftUI.
2. Set the deployment target to iOS 17 or newer.
3. Add the Swift files from this folder to the Xcode project.
4. In `APIConfig.swift`, replace `YOUR_CLOUD_RUN_URL` with the HTTPS URL returned by Cloud Run.
5. Build and run on the iPhone simulator or a physical iPhone.

The first mobile milestone contains the Portfolio screen only. Method Comparison, Backtest, Diagnostics, saved portfolios, and push notifications can be added after the shared API is stable.
