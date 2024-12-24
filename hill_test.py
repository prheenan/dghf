"""
Testing module for hill fitter
"""
import unittest
import warnings
import logging
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import numpy as np
import plotly.graph_objects as go
from scipy.stats import linregress
import dghf
from scripts import canvass_download

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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


class MyTestCase(unittest.TestCase):

    def __init__(self,*args,**kw):
        super().__init__(*args,**kw)
        self.i_subtest = 0
        np.random.seed(42)
        self.simulated_data = self._simulated_data()

    def _assert_close_kw(self,kw_found,kw_expected,atol=0,rtol=1e-6,
                         atol_signal=None,rtol_signal=None):
        """

        :param kw_found: fitted keyworkds
        :param kw_expected:  what we expect
        :param atol: input to np.testing.assert_allclose for hill/logKa
        :param rtol: input to np.testing.assert_allclose for hill/logKa
        :param atol_signal: input to np.testing.assert_allclose for min/max
        :param rtol_signal: input to np.testing.assert_allclose for min/max
        :return: asserts true/value if available
        """
        if atol_signal is None:
            atol_signal = atol
        if rtol_signal is None:
            rtol_signal = rtol
        signal_names = ['min_v', 'max_v',]
        non_signal_names = ['log_K_a', 'n']
        for names,atol_i,rtol_i in [ [non_signal_names,atol,rtol],
                                     [signal_names,atol_signal,rtol_signal]]:
            fit_found = [kw_found[n] for n in names]
            fit_expected = [ kw_expected[n] for n in names]
            with self.subTest(self.i_subtest):
                np.testing.assert_allclose(fit_found,fit_expected, atol=atol_i,
                                           rtol=rtol_i)
                self.i_subtest += 1

    def _simulated_data(self):
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
            [ [[np.nan] * 11, [np.nan]*11,
               dict(min_v=np.nan, max_v=np.nan, log_K_a=np.nan,n=np.nan)], {}],
            # small number
            [_get_x_y_k(-7, -4, n_size=4,**kw_fit_1), {"atol":1e-4}],
            # try only going to 10 uM (-5)
            [_get_x_y_k(-7, -5, noise_scale=5, **dict(min_v=10, max_v=100, log_K_a=np.log(1e-6),n=1)),
             {'rtol':0.16,'atol_signal':9}],
            # try only 8 points
            [_get_x_y_k(-7, -4, noise_scale=5,n_size=8,
                        **dict(min_v=10, max_v=100, log_K_a=np.log(1e-6), n=1)),
             {'rtol':0.23, 'atol_signal':26}],

        ]
        return x_y_k_err

    def test_prepared_data(self):
        """

        :return: nothing, test prepared data
        """
        self.i_subtest = 0
        x_y_k_err = self.simulated_data
        # only specify final bounds as :
        # final hill coefficient, which should be positive
        for (x,y,kw_expected),kw_err in x_y_k_err[::-1]:
            kw_fit = dghf.fit(x,y,bounds_n=[0,np.inf])
            self._assert_close_kw(kw_found=kw_fit, kw_expected=kw_expected,
                                  **kw_err)

    def test_overflow_inducing_runs(self):
        """
        test runs which should induced overflow
        """
        self.i_subtest = 0
        x_y_k_err = self.simulated_data
        for (x,y,_),__ in x_y_k_err[::-1]:
            with self.subTest(i=self.i_subtest):
                with warnings.catch_warnings(category=RuntimeWarning):
                    warnings.simplefilter("error",category=RuntimeWarning)
                    dghf.fit(x,y,bounds_n=[0,np.inf],coarse_n=2)
            self.i_subtest += 1

    def test_bounds(self):
        """
        Tested bounded fit problems
        """
        x_y_k_err = self.simulated_data
        for (x,y,kw_expected),kw_err in x_y_k_err[::-1]:
            for name in dghf.param_names_order():
                bounds = { 'bounds_min_v':[None, None],
                           'bounds_max_v':[None, None],
                           'bounds_log_K_a':[None, None],
                           'bounds_n':[0, np.inf]}
                bounds[f'bounds_{name}'] = [kw_expected[name],kw_expected[name]]
                kw_fit = dghf.fit(x,y,**bounds)
                self._assert_close_kw(kw_found=kw_fit, kw_expected=kw_expected,
                                      **kw_err)

    def test_canvass(self):
        """
        Test the canvass data set, makng sure the residuals and R2 look OK
        """
        self.i_subtest = 0
        df_canvass = \
            canvass_download.read_canvass_data(out_dir="./out/test/cache_canvass")
        x_y_dict = {id_v: {'x':df_v["Concentration (M)"].to_numpy(),
                           'y':df_v["Activity (%)"].to_numpy()}
                    for id_v, df_v in df_canvass.groupby("Curve ID")}
        ids = sorted(set(df_canvass["Curve ID"]))
        x_y = [x_y_dict[i] for i in ids]
        n_pool = cpu_count() - 1
        n_curves = len(x_y)
        with Pool(n_pool) as p:
            all_fit_kw = list(tqdm(p.imap(dghf._fit_multiprocess, x_y),
                                   total=n_curves,
                                   desc="Fitting CANVASS"))
        y_pred_arr = []
        for (x_y_kw), kw in tqdm(zip(x_y, all_fit_kw),desc="Predict CANVASS",
                                 total=n_curves):
            y_pred = dghf.hill_log_Ka(x=x_y_kw['x'], **kw)
            y_pred_arr.append(y_pred)
        stats = [linregress(x=x_y_kw['x'], y=y_pred_i)
                 for x_y_kw,y_pred_i in tqdm(zip(x_y,y_pred_arr),
                                             desc="Regress CANVASS",
                                             total=n_curves)]
        median_error = [np.median(np.abs((kw['y'] - y_pred_i)))
                        for kw, y_pred_i in zip(x_y, y_pred_arr)]
        all_r2 = [s.rvalue ** 2 for s in stats]
        error_med, error_90 = np.percentile(median_error,[50,90])
        r2_med, r2_25 = np.percentile(all_r2,[50,25])
        logger.info("test_canvass:: R2 median/25th: {:.3f}/{:.3f}".format(r2_med,r2_25))
        logger.info("test_canvass:: residual median/90th: {:.2f}/{:.2f}".format(error_med,error_90))
        with self.subTest(self.i_subtest):
            assert r2_med >= 0.63
        self.i_subtest += 1
        with self.subTest(self.i_subtest):
            assert r2_25 >= 0.145
        self.i_subtest += 1
        with self.subTest(self.i_subtest):
            assert error_med <= 1.3
        self.i_subtest += 1
        with self.subTest(self.i_subtest):
            assert error_90 <= 4.3
        self.i_subtest += 1


def _debug_plot(x,y,kw_fit):
    """

    :param x: x values
    :param y: y values
    :param kw_fit:  fitted x and y
    :return:  nothing, gives plot
    """
    x_interp = np.logspace(min(np.log10(x / 10)), max(np.log10(x * 10)), base=10)
    y_interp = dghf.hill_log_Ka(x=x_interp, **kw_fit)
    fig = go.Figure()
    # Add the first line with markers
    fig.add_trace(go.Scatter(x=x, y=y, mode='markers', name="Data"))
    # Add the second line without markers
    fig.add_trace(go.Scatter(x=x_interp, y=y_interp, mode='lines', name="Fit"))
    fig.update_layout(xaxis_type="log")
    fig.show()




if __name__ == '__main__':
    unittest.main()
