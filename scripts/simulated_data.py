"""
script o create simulated data
"""
import numpy as np
import dghf


def _get_x_y_k(log10_i=-9,log10_f=-4,noise_scale=0,n_zero=0,n_nan=0,n_size=11,**kw):
    """

    :param log10_i: which magnitude to start (e.g., -6 is 1 uM
    :param log10_f:  which magnitude to end  (e.g., -3 is 1 mM)
    :param noise_scale: scale of noise
    :param n_zero: number of points at zero concentration
    :param n_nan: number of points with nan
    :param kw: see hill_log_Ka
    :return: tuple of (x values, y values, kw)
    """
    x = np.logspace(log10_i, log10_f, num=n_size,base=10,endpoint=True)
    y = dghf.hill_log_Ka(x=x, **kw) + \
        np.random.normal(loc=0,scale=noise_scale,size=x.size)
    if n_zero > 0:
        # add in the zero points
        x = [0 for _ in range(n_zero)] + list(x)
        y = list(kw["min_v"] + np.random.normal(loc=0,scale=noise_scale,size=n_zero)) + list(y)
    if n_nan > 0:
        x = [min(x)] * n_nan + list(x)
        y = [np.nan] * n_nan + list(y)
    return np.array(x),np.array(y),kw


def simulated_data():
    """

    :return: list of tuples, where each tuple is like

    ( (x values, y values, expected parameters), dictionary of error terms
    for  _assert_close_kw)
    """
    kw_fit_1 = {'min_v':10, 'max_v':100, 'log_K_a':np.log(1e-6),'n':1}
    x_y_k_err = [
        [_get_x_y_k(-7, -4, **kw_fit_1),{'atol':1e-4}],
        [_get_x_y_k(-7, -4, noise_scale=5,**kw_fit_1),
         {'rtol':0.15,'atol_signal':9}],
        [_get_x_y_k(-7, -4, noise_scale=5, n_zero=2,**kw_fit_1),
         {'rtol':0.15,'atol_signal':9}],
        [_get_x_y_k(-7, -4, noise_scale=5, n_zero=2,n_nan=2, **kw_fit_1),
         {'rtol':0.5, 'atol_signal':9}],
        # oops all nans
        [[[np.nan] * 11, [np.nan]*11,
           {'min_v':np.nan, 'max_v':np.nan, 'log_K_a':np.nan,'n':np.nan}], {}],
        # small number
        [_get_x_y_k(-7, -4, n_size=4,**kw_fit_1), {"atol":1e-4}],
        # try only going to 10 uM (-5)
        [_get_x_y_k(-7, -5, noise_scale=5, min_v=10, max_v=100, log_K_a=np.log(1e-6),n=1),
         {'rtol':0.16,'atol_signal':9}],
        # try only 8 points
        [_get_x_y_k(-7, -4, noise_scale=5,n_size=8,
                    min_v=10, max_v=100, log_K_a=np.log(1e-6), n=1),
         {'rtol':0.23, 'atol_signal':26}],
    ]
    return x_y_k_err
