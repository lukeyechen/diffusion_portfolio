from pathlib import Path

# Method Comparison (Legacy), upgraded so Section A itself uses the requested
# short holding periods. Sections B-E remain the original Legacy research tools,
# and Section F (Historical/Gaussian/Neural true same-frequency OOS) is appended
# from Method_Comparison_Legacy_Upgraded.py.
_legacy_path = Path("views") / "Method_Comparison.py"
_legacy_source = _legacy_path.read_text(encoding="utf-8")

_a_marker = (
    "# ============================================================\n"
    "# A. Return-horizon theoretical study\n"
    "# ============================================================\n"
)
_b_marker = (
    "# ============================================================\n"
    "# B. Selected-horizon monthly OOS comparison\n"
    "# ============================================================\n"
)

_a_start = _legacy_source.index(_a_marker)
_b_start = _legacy_source.index(_b_marker)

_short_a = r'''# ============================================================
# A. Short return-horizon theoretical study
# ============================================================
st.divider()
st.header("A. Return-Horizon Study")
st.caption(
    "Compares theoretical diffusion feasibility across 1W, 2W, 1M, and 3M holding-period returns."
)

_a_save_col, _a_download_col = st.columns([1, 2])
with _a_save_col:
    if st.button("💾 Save A results", key="save_section_a", use_container_width=True):
        if "mc_horizon_table" not in st.session_state:
            st.warning("Run Section A first, then save the results.")
        else:
            st.session_state["section_a_saved_zip"] = _build_section_a_export_zip(
                st.session_state["mc_horizon_table"]
            )
            st.success("Section A results package prepared.")

with _a_download_col:
    if "section_a_saved_zip" in st.session_state:
        st.download_button(
            "⬇ Download A results (.zip)",
            data=st.session_state["section_a_saved_zip"],
            file_name="section_A_short_return_horizon_study.zip",
            mime="application/zip",
            key="download_section_a",
            use_container_width=True,
        )

if st.button(
    "Run Return-Horizon Study",
    type="primary",
    key="run_return_horizon_study",
    use_container_width=True,
):
    st.session_state["mc_stop_requested"] = False
    if source != "Yahoo Finance":
        st.error("The automatic 1W/2W/1M/3M study requires Yahoo Finance.")
    else:
        try:
            from core.data import download_yahoo_returns as _short_download_yahoo_returns
            from core.short_horizon_portfolio import (
                aggregate_nonoverlapping as _short_aggregate_nonoverlapping,
            )

            with st.spinner("Computing 1W / 2W / 1M / 3M theoretical horizons..."):
                _weekly = _short_download_yahoo_returns(
                    ticker_list,
                    start=start_date,
                    end=None,
                    interval="1wk",
                )[ticker_list].dropna()
                _monthly = _short_download_yahoo_returns(
                    ticker_list,
                    start=start_date,
                    end=None,
                    interval="1mo",
                )[ticker_list].dropna()

                horizon_data = {
                    "1W": _weekly,
                    "2W": _short_aggregate_nonoverlapping(_weekly, 2),
                    "1M": _monthly,
                    "3M": _short_aggregate_nonoverlapping(_monthly, 3),
                }
                table = return_horizon_study(horizon_data, beta=float(beta))

            st.session_state.pop("section_a_saved_zip", None)
            st.session_state["mc_horizon_table"] = table
        except Exception as exc:
            st.error(f"Return-horizon study failed: {exc}")

if "mc_horizon_table" in st.session_state:
    table = st.session_state["mc_horizon_table"]
    st.dataframe(
        table.style.format(
            {
                "N/n": "{:.4f}",
                "b*": "{:.4f}",
                "b*/n": "{:.4f}",
                "T": "{:.4f}",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

    chart_df = table.copy()
    fig = px.bar(
        chart_df,
        x="Horizon",
        y="b*/n",
        text_auto=".3f",
        title="Theoretical feasibility ratio by short holding period",
    )
    fig.add_hline(
        y=1.0,
        line_dash="dash",
        annotation_text="Boundary b*/n = 1",
    )
    st.plotly_chart(fig, use_container_width=True)


'''

_patched_source = _legacy_source[:_a_start] + _short_a + _legacy_source[_b_start:]
_legacy_code = compile(_patched_source, str(_legacy_path), "exec")
exec(_legacy_code, globals(), globals())

# Append the already-tested Section F short-horizon Legacy OOS comparison,
# without executing the original Legacy page a second time.
_extra_path = Path("views") / "Method_Comparison_Legacy_Upgraded.py"
_extra_source = _extra_path.read_text(encoding="utf-8")
_tail_marker = "import io\nimport zipfile\n"
_tail_start = _extra_source.index(_tail_marker)
_extra_tail = _extra_source[_tail_start:]
exec(compile(_extra_tail, str(_extra_path), "exec"), globals(), globals())
