# IPO Watch

A Streamlit app for completed U.S. IPO listings and price charts. Search by company or ticker, filter by year/month, exclude inferred SPACs, and open embedded TradingView or external Yahoo charts.

## Run locally

```sh
cd ipo_watch
python -m pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Deploy to Streamlit Community Cloud

Select this GitHub repository, its default branch, and `ipo_watch/streamlit_app.py` as the main file. Use Python 3.12. No API keys or app secrets are needed. Review the Streamlit app's sharing settings before deployment; repository visibility and app visibility are separate.

## Automatic new IPOs

- The app fetches the current-year Stock Analysis IPO directory with a 15-minute cache on visits and interactions. It merges new rows with saved history.
- `.github/workflows/update-ipo-watch.yml` updates the committed snapshot daily at **23:30 UTC**. GitHub Actions must be enabled and permitted to write repository contents. Run it immediately with **Actions → Update IPO listings → Run workflow**. This branch must be merged into the default branch to activate the schedule. Scheduled workflows may be delayed by GitHub and only run from the default branch. Inactive public repositories may have scheduled workflows disabled by GitHub.
- New calendar years are picked up automatically. Completed listings only; the app does not treat expected IPOs as completed.
- If the source is unavailable or the parser finds no completed listings, the last successful snapshot is retained. A failed scheduled check is visible in GitHub Actions. Source coverage is not a guarantee of all exchange listings.
- Streamlit runtime writes are a convenience cache, not durable storage. The repository snapshot is the durable fallback when the app restarts. A daily commit normally causes Streamlit to update from GitHub.

The bundled snapshot includes the original 2026 directory plus verified November 2 listing NARA. NARA's offer price is left blank because it was not present in the primary-source evidence used for that addition: https://www.sec.gov/Archives/edgar/data/2138186/000135445726001043/NARA_8A_Cert_0002138186.pdf

## Data and charts

Source: https://stockanalysis.com/ipos/2026/ (year changes automatically). Direct listings and SPACs may be included. SPAC labels are heuristics from company names. Offer/reference prices are not necessarily the prices retail buyers could obtain; corporate actions can affect comparisons. Charts are hosted by TradingView; verify company and exchange. External links provide fallback charts.

## Checks

```sh
python -m unittest discover -s tests
```

No credentials, private Sites access tokens, or personal holdings are included.
