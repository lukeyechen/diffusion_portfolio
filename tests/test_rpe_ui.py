from pathlib import Path
import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest

PAGE=Path(__file__).resolve().parents[1]/'views/RPE.py'


def test_rpe_uses_saved_monthly_data_and_retains_results():
    script=f'''import streamlit as st
from pathlib import Path
if st.sidebar.radio('Page',['RPE','Other'])=='RPE':
    p=Path({str(PAGE)!r})
    exec(compile(p.read_text(),str(p),'exec'),globals(),globals())
else: st.write('Other page')
'''
    app=AppTest.from_string(script)
    x=pd.DataFrame(np.random.default_rng(8).normal(.005,.05,(126,5)),columns=list('ABCDE'),index=pd.date_range('2010-01-31',periods=126,freq='ME'))
    app.session_state['shared_current_window']={'full_returns':x,'holding_period':'1 month'}
    app.run()
    next(w for w in app.text_input if w.label=='Backtest / strategy replay start date').set_value(str(x.index[124].date())).run()
    next(w for w in app.number_input if w.label=='Posterior draws').set_value(100).run()
    next(b for b in app.button if b.label=='Run RPE comparison').click().run(timeout=120)
    assert not app.exception and not app.error
    assert len(app.dataframe)==2
    assert 'RPE predictive (TC)' in app.dataframe[1].value.Method.values
    original=app.dataframe[1].value.copy()
    app.sidebar.radio[0].set_value('Other').run()
    app.sidebar.radio[0].set_value('RPE').run()
    assert not app.exception
    pd.testing.assert_frame_equal(original,app.dataframe[1].value)


def test_nonmonthly_saved_data_is_not_silently_used():
    app=AppTest.from_file(PAGE)
    app.session_state['shared_current_window']={'full_returns':pd.DataFrame(np.ones((130,5))), 'holding_period':'1 week'}
    app.run()
    assert not app.exception
    assert not app.button
    assert any('1 month interval' in w.value for w in app.info)
