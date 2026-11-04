from datetime import datetime, timezone
from urllib.parse import urlencode, quote
import streamlit as st
from ipo_data import read_snapshot, refresh

st.set_page_config(page_title='IPO Watch', page_icon='📈', layout='wide')
st.markdown('''<style>
.block-container{padding-top:2rem;max-width:1500px}h1{letter-spacing:-1.3px}div[data-testid="stMetric"]{border-bottom:1px solid #263548;padding-bottom:12px}.caption{color:#98aabd;font-size:13px}.kicker{color:#72dfb5;letter-spacing:2px;font-size:11px;font-weight:600}.company{font-size:23px;font-weight:500;margin:8px 0}.offer{font-size:29px;margin-top:4px}a{color:#72dfb5!important}
</style>''', unsafe_allow_html=True)

@st.cache_data(ttl=900, show_spinner=False)
def load_snapshot():
    try:
        return refresh(), None
    except Exception:
        return read_snapshot(), 'The live source is unavailable. Showing the last saved list; you can retry with Refresh.'

left, right = st.columns([5, 1])
with left:
    st.markdown('<div class="kicker">IPO WATCH · NEW TO THE MARKET</div>', unsafe_allow_html=True)
    st.title('Every debut. One place.')
    st.caption('Explore U.S. listings and follow their price since going public.')
with right:
    if st.button('↻ Refresh listings', width='stretch'):
        load_snapshot.clear()

with st.spinner('Checking IPO listings…'):
    snapshot, warning = load_snapshot()
rows = snapshot['rows']
if warning: st.warning(warning)
updated = datetime.fromisoformat(snapshot['updated_at']).strftime('%b %d, %Y · %H:%M UTC')
st.caption(f"{len(rows)} saved listings · Last successful update: {updated} · Source coverage and delays may vary")
st.divider()

filter_col, chart_col = st.columns([1, 2.1], gap='large')
with filter_col:
    st.subheader('IPO directory')
    search = st.text_input('Search company or ticker', placeholder='e.g. MANE or Veradermics', key='search')
    years = sorted({r['date'][:4] for r in rows} | {str(datetime.now(timezone.utc).year)}, reverse=True)
    a, b = st.columns(2)
    year = a.selectbox('Year', years, key='year')
    month_names = ['All months', 'January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
    month = b.selectbox('Month', month_names, key='month')
    exclude = st.checkbox('Exclude tagged SPACs', key='exclude_spacs')
    visible = [r for r in rows if r['date'].startswith(year)
               and (month == 'All months' or int(r['date'][5:7]) == month_names.index(month))
               and (not exclude or r['kind'] != 'SPAC')
               and search.casefold() in (r['ticker']+' '+r['name']).casefold()]
    visible.sort(key=lambda r: (r['date'], r['ticker']), reverse=True)
    st.caption(f'{len(visible)} matching listings')
    if not visible:
        st.info('No matching listings. Try another month or search.')
        selected = None
    else:
        lookup = {r['date']+':'+r['ticker']: r for r in visible}
        # A changing option list invalidates an old selection safely.
        if st.session_state.get('company') not in lookup:
            st.session_state['company'] = next(iter(lookup))
        key = st.selectbox('Select company', list(lookup), format_func=lambda k: f"{lookup[k]['ticker']} · {lookup[k]['name']}", key='company')
        selected = lookup[key]
        with st.expander('View filtered directory'):
            st.dataframe([{'Listed':r['date'], 'Ticker':r['ticker'], 'Company':r['name'], 'Offer / reference':r['offer']} for r in visible], hide_index=True, width='stretch')
with chart_col:
    if selected:
        st.markdown('<div class="kicker">SELECTED LISTING</div>', unsafe_allow_html=True)
        st.subheader(selected['name'])
        st.caption(f"{selected['ticker']} · Listed {selected['date']}")
        offer = '—' if selected['offer'] is None else f"${selected['offer']:,.2f}"
        st.metric('IPO offer / directory reference price', offer)
        st.caption('May differ from the first available trading price.')
        query = urlencode({'symbol':selected['ticker'], 'interval':'D', 'range':'ALL', 'theme':'dark', 'style':'3', 'timezone':'America/New_York', 'withdateranges':'1', 'hideideas':'1', 'locale':'en'})
        st.iframe('https://s.tradingview.com/widgetembed/?'+query, height=470, alt='Selected IPO stock-price chart')
        a,b = st.columns(2)
        a.link_button('Open Yahoo chart ↗', f"https://finance.yahoo.com/quote/{quote(selected['ticker'])}/chart/", width='stretch')
        b.link_button('Company details ↗', f"https://stockanalysis.com/stocks/{quote(selected['ticker'].lower())}/", width='stretch')
        st.caption('TradingView supplies price history. Verify the chart company and exchange; use Yahoo if a symbol is unavailable.')
st.divider()
st.caption('New listings are checked when the app opens or refreshes (15-minute cache). The included GitHub Actions workflow checks daily while the app is closed, once enabled in the repository. SPAC tags are inferred from names; direct listings may be included. Data is informational, and completeness is not guaranteed.')
st.markdown(f"[Listing source]({snapshot['source']})")
