from .core import nrcs_nesting_eqn
import pandas as pd
import numpy as np

ratio_to_24h = pd.DataFrame(np.arange(start=0, stop=241, step=1), columns = ['time']).set_index(['time'])


dur_precip_df_noaa = pd.DataFrame([['05m',0.1464],['10m',0.2252]],columns = ['time','ratio']).set_index('time')


cumulative_ratio = {}
cumulative_ratio['0'] = 0
cumulative_ratio['6'] = 0.0608
cumulative_ratio['9'] = 0.1216
cumulative_ratio['10.5'] = 0.1802
cumulative_ratio['11'] = 0.1993
cumulative_ratio['11.5'] = 0.2466
cumulative_ratio['11.75'] = 0.3041
cumulative_ratio['11.875'] = 0.3615
cumulative_ratio['11.9167'] = 0.3874

nrcs_result = pd.read_csv('nrcs_results.csv').to_dict()['ratio']

def test_nrcs_example():
    t_step = np.arange(start=0, stop=241, step=1)
    result = []
    for t in t_step:
        result = nrcs_nesting_eqn(t,cumulative_ratio,dur_precip_df_noaa)
        assert abs(result - nrcs_result[t]) < 0.00008, f'function is not returning prescribed cumulative distribution at time step {t}, expected value is {nrcs_result[t]} but result is {result}'
