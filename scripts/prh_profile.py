"""
Module to run profile on like so:

python prh_profile.py ; snakeviz program.prof
"""
import sys
import timeit
import cProfile
import functools
import pstats
import pandas
import numpy as np
sys.path.append("../")
np.bool8 = np.bool
from plotly import express as px
from plotly.subplots import make_subplots
import hill_test
import dghf

def run_many_times(repeats,x_y_kw,bounds_n):
    """
    Fit the data multiple times

    :param repeats: how many times to repeat
    :param x_y_kw: list; each element like <x,y,kw of expected params>
    :param bounds_n: bounds for hill coefficient
    :return: nothing, just runs a bunch
    """
    for _ in range(repeats):
        for x,y,_ in x_y_kw:
            dghf.fit(x=x, y=y, bounds=bounds_n)

def run():
    """
    Runs the profiling code
    """
    simulated_data = hill_test.MyTestCase().simulated_data
    # ignore the all nan data set
    x_y_kw = [s[0] for s in simulated_data if
              not set(s[0][-1].values()) == set([np.nan])]
    # all of the simulated data has positive hill coefficient
    bounds_n = [0, np.inf]
    profiler = cProfile.Profile()
    profiler.enable()
    run_many_times(repeats=10,x_y_kw=x_y_kw,bounds_n=bounds_n)
    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats('cumtime')
    stats.dump_stats('program.prof')


def plot_error_and_time(kw_fit_arr, x_y_kw, title, bounds_n, x="coarse_n",
                        time_repeats=10,
                        convert_v=None, height=600, width=400, log_x=False):
    """

    :param kw_fit_arr: list, length M, each element a dictionary of fit params
    :param x_y_kw:  list, length N, each element (x vals, y val, expected kw)
    :param title: for plot
    :param bounds_n:  bounds on hill coefficient
    :param x: what to plot on x value
    :param time_repeats: how many times to repeat
    :param convert_v: dictionary, each element converts from key of
    kw_fit_arr elements to function to convert
    :param height: px, of plot
    :param width:  px, of plot
    :param log_x:  if x should be plotted log-style
    :return: nothing
    """
    df_error_n, df_time_n = error_values(kw_fit_arr=kw_fit_arr, x_y_kw=x_y_kw,
                                         bounds_n=bounds_n,
                                         time_repeats=time_repeats)
    x_vals = sorted(set([e for kw in kw_fit_arr for e in kw.keys()]))
    if convert_v is not None:
        for d in [df_error_n, df_time_n]:
            for x in x_vals:
                d[x] = [convert_v[x](e) for e in d[x]]
            d.sort_values(by=x_vals, inplace=True)
    df_error_median_n = df_error_n[
        ["Error (%)", "Parameter set"] + x_vals].groupby(
        ["Parameter set"] + x_vals).median().reset_index()
    df_time_and_error = df_error_median_n.merge(df_time_n,
                                                on=["Parameter set"] + x_vals)
    # Create subplots
    category_orders = {x: sorted(set(df_time_and_error[x]))}
    fig = make_subplots(rows=2, cols=1, horizontal_spacing=0.15)
    fig.add_trace(
        px.box(x=x, y="Error (%)", points="all", data_frame=df_time_and_error,
               log_y=True, category_orders=category_orders).data[0], row=1,
        col=1)
    fig.add_trace(px.box(df_time_n, x=x, y="Time (s)", points="all",
                         category_orders=category_orders).data[0], row=2, col=1)
    fig.update_layout(autosize=False, height=height, width=width, title=title)
    log = 'log' if log_x else None
    fig.update_xaxes(title_text=x, row=1, col=1, type=log)
    fig.update_yaxes(title_text="Error (%)", type='log', row=1, col=1)
    fig.update_xaxes(title_text=x, row=2, col=1, type=log)
    fig.update_yaxes(title_text="Time (s)", row=2, col=1)
    fig.show(renderer="iframe")


def error_values(kw_fit_arr, x_y_kw, bounds_n=None, time_repeats=0):
    """

    :param kw_fit_arr: see plot_error_and_time
    :param x_y_kw: see plot_error_and_time
    :param bounds_n: see plot_error_and_time
    :param time_repeats: see plot_error_and_time
    :return: tuple, one df of errors and one df of times
    """
    errors = []
    times = []
    for kw_params in kw_fit_arr:
        for i, (x, y, kw) in enumerate(x_y_kw):
            _f_callable = functools.partial(dghf.fit,x=x, y=y, bounds_n=bounds_n, **kw_params)
            kw_fit = _f_callable()
            if time_repeats > 0:
                time_avg = timeit.timeit(_f_callable, number=time_repeats)
            else:
                time_avg = np.nan
            times.append(
                {"Parameter set": i, "Time (s)": time_avg, **kw_params})
            for k, v in kw_fit.items():
                v_expected = kw[k]
                v_calculate = v
                errors.append({"Parameter set": i, "Parameter": k,
                               "Error (%)": 100 * abs((v_expected - v_calculate) / (v_expected)),
                               **kw_params})
    return pandas.DataFrame(errors), pandas.DataFrame(times)


if __name__ == "__main__":
    run()
