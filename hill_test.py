"""
Testing module for hill fitter
"""
import unittest
import warnings
import platform
import logging
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import numpy as np
import plotly.graph_objects as go
from scipy.stats import linregress
import dghf
from scripts import canvass_download
from scripts import prh_profile

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

    @classmethod
    def setUpClass(cls):
        np.random.seed(42)
        cls.simulated_data = simulated_data()
        cls.df_canvass = canvass_download.\
            read_canvass_data(out_dir="./out/test/cache_canvass")

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

    def test_00_prepared_data(self):
        """

        :return: nothing, test prepared data
        """
        self.i_subtest = 0
        x_y_k_err = MyTestCase.simulated_data
        # only specify final bounds as :
        # final hill coefficient, which should be positive
        for (x,y,kw_expected),kw_err in x_y_k_err[::-1]:
            kw_fit = dghf.fit(x,y,bounds_n=[0,np.inf])
            self._assert_close_kw(kw_found=kw_fit, kw_expected=kw_expected,
                                  **kw_err)

    def test_01_overflow_inducing_runs(self):
        """
        test runs which should induced overflow
        """
        self.i_subtest = 0
        x_y_k_err = MyTestCase.simulated_data
        for (x,y,_),__ in x_y_k_err[::-1]:
            with self.subTest(i=self.i_subtest):
                with warnings.catch_warnings(category=RuntimeWarning):
                    warnings.simplefilter("error",category=RuntimeWarning)
                    dghf.fit(x,y,bounds_n=[0,np.inf],coarse_n=2)
            self.i_subtest += 1

    def test_02a_simulated_bounds(self):
        """
        Tested bounded fit problems
        """
        x_y_k_err = MyTestCase.simulated_data
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
                # if I add in the expected bounds, should stll be OK
                bounds_oracle = { 'bounds_min_v':[-20, 20],
                                  'bounds_max_v':[80, 120],
                                  'bounds_log_K_a':[kw_expected["log_K_a"]-np.log(10),
                                                    kw_expected["log_K_a"]+np.log(10)],
                                  'bounds_n':[kw_expected["n"]*0.8, kw_expected["n"]*1.2]}
                kw_fit_oracle = dghf.fit(x,y,**bounds_oracle)
                self._assert_close_kw(kw_found=kw_fit_oracle, kw_expected=kw_expected,
                                      **kw_err)
                # more realistic bounds should also be OK
                x_nz = np.array(x)[np.where(np.array(x) >0)[0]]
                if x_nz.size > 0:
                    bounds_realistic = { 'bounds_min_v':[-20, 20],
                                         'bounds_max_v':[80, 120],
                                         'bounds_log_K_a':[np.log(min(x_nz)),
                                                           np.log(max(x_nz))],
                                         'bounds_n':[0, 5]}
                    kw_fit_realistic = dghf.fit(x,y,**bounds_realistic)
                    self._assert_close_kw(kw_found=kw_fit_realistic,
                                          kw_expected=kw_expected,
                                          **kw_err)

    def test_02a_inactivity(self):
        """
        test that inactive dataframes are marked as such
        """
        # get a bunch of inactive data
        x_y = canvass_download.\
            read_xy_from_assay_cid(df=MyTestCase.df_canvass,
                                   cid_assay=canvass_download.inactive_cid_assays())
        for x,y in x_y:
            kw_fit_with_range = dghf.fit(inactive_range=[-50, 50],x=x,y=y)
            min_y, max_y = min(y), max(y)
            dy = max_y - min_y
            with self.subTest(i=self.i_subtest):
                assert kw_fit_with_range['min_v'] >= min_y - dy
            self.i_subtest += 1
            with self.subTest(i=self.i_subtest):
                assert kw_fit_with_range['max_v'] <= max_y + dy
            self.i_subtest += 1
            with self.subTest(i=self.i_subtest):
                assert kw_fit_with_range["log_K_a"] == np.log(max(x[np.where(x > 0)[0]]))
            self.i_subtest += 1


    def test_03_canvass_exemplars(self):
        """
        Test exemplars chosen from the canvass set to be very high quality
        """
        cid_assay = canvass_download.exemplar_cid_assays()
        x_y = canvass_download.demo_x_y_data(df=MyTestCase.df_canvass)
        self.i_subtest = 0
        cid_assay_lower = set( [(134827994,1347393)])
        for (x,y),(cid,assay) in zip(x_y,cid_assay):
            lower = (cid,assay in cid_assay_lower)
            r2_thresh = 0.94 if lower else 0.98
            slope_thresh = 0.94 if lower else 0.98
            resid_thresh = 3
            kw_for_fit = dict(x=x, y=y, bounds_n=[0,np.inf])
            # fit with and without inactive range;
            kw_fit_with_range = dghf.fit(inactive_range=[-50, 50], **kw_for_fit)
            kw_fit_without_range = dghf.fit(**kw_for_fit)
            keys = sorted(set(kw_fit_with_range.keys()))
            # fits shouldn't change much
            with self.subTest(self.i_subtest):
                np.testing.assert_allclose([kw_fit_with_range[k] for k in keys],
                                           [kw_fit_without_range[k] for k in
                                            keys], rtol=0.05)
            self.i_subtest += 1
            # fits were chosen such that they should be very high quality
            for kw in [kw_fit_with_range, kw_fit_without_range]:
                y_pred_i = dghf.hill_log_Ka(x=x, **kw)
                stats = linregress(x=y, y=y_pred_i)
                r2 =  stats.rvalue**2
                median_resid = np.median(np.abs(y_pred_i - y))
                logger.info( "test_canvass_exemplars:: resid/r2/slope: {:.1f}/{:.3f}/{:.3f}".\
                             format(median_resid,r2,stats.slope))
                with self.subTest(self.i_subtest):
                    assert median_resid < resid_thresh, median_resid
                self.i_subtest += 1
                with self.subTest(self.i_subtest):
                    assert r2 > r2_thresh , r2
                self.i_subtest += 1
                with self.subTest(self.i_subtest):
                    assert stats.slope > slope_thresh , stats.slope
                self.i_subtest += 1

    def test_04_timing(self):
        """
        On my local computer only, check the timing
        """
        self.i_subtest = 0
        comp = platform.uname().node.split(".")[0]
        logger.info("test_04_timing:: {:s}".format(comp))
        if comp != 'Patricks-MBP-3':
            return
        n_repeats = 10
        x_y_kw = [ (x,y,kw_expected)
                   for (x,y,kw_expected),kw_err in MyTestCase.simulated_data
                   if not all(np.isnan(sorted(kw_expected.values())))]
        _, df_time_n = prh_profile.\
            error_values(kw_fit_arr=[dict() for _ in range(n_repeats)],
                         bounds_n=[0, np.inf], time_repeats=10,
                         x_y_kw=x_y_kw)
        median_per_run = df_time_n.groupby("Parameter set").\
            median()["Time/run (s/run)"]
        median_of_median = np.median(median_per_run)
        max_of_median = np.max(median_per_run)
        logger.info("test_04_timing:: median-median/max-median: {:.3f}s/{:.3f}s".\
            format(median_of_median, max_of_median))
        with self.subTest(i=self.i_subtest):
            assert median_of_median < 0.03, median_of_median
        self.i_subtest += 1
        with self.subTest(i=self.i_subtest):
            assert max_of_median < 0.035, max_of_median
        self.i_subtest += 1


    def test_99_canvass(self):
        """
        Test the canvass data set, makng sure the residuals and R2 look OK
        """
        self.i_subtest = 0
        x_y_dict = {id_v: {'x':df_v["Concentration (M)"].to_numpy(),
                           'y':df_v["Activity (%)"].to_numpy()}
                    for id_v, df_v in MyTestCase.df_canvass.groupby("Curve ID")}
        ids = sorted(set(MyTestCase.df_canvass["Curve ID"]))
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
        stats = [ _safe_regress(x=x_y_kw['y'], y=y_pred_i)
                 for x_y_kw,y_pred_i in tqdm(zip(x_y,y_pred_arr),
                                             desc="Regress CANVASS",
                                             total=n_curves)]
        median_error = [np.median(np.abs((kw['y'] - y_pred_i)))
                        for kw, y_pred_i in zip(x_y, y_pred_arr)]
        all_r2 = [s.rvalue ** 2 if s is not None else np.nan for s in stats ]
        error_med, error_90 = np.nanpercentile(median_error,[50,90])
        r2_med, r2_25 = np.nanpercentile(all_r2,[50,25])
        fraction_nan = sum(np.isnan(all_r2)) / len(all_r2)
        logger.info("test_99_canvass:: R2 median/25th: {:.3f}/{:.3f}".format(r2_med,r2_25))
        logger.info("test_99_canvass:: residual median/90th: {:.2f}/{:.2f}".format(error_med,error_90))
        logger.info("test_99_canvass:: fraction nan {:.3f}".format(fraction_nan))
        with self.subTest(self.i_subtest):
            assert fraction_nan < 0.01, fraction_nan
        self.i_subtest += 1
        with self.subTest(self.i_subtest):
            assert r2_med >= 0.58, r2_med
        self.i_subtest += 1
        with self.subTest(self.i_subtest):
            assert r2_25 >= 0.32, r2_25
        self.i_subtest += 1
        with self.subTest(self.i_subtest):
            assert error_med <= 1.3, error_med
        self.i_subtest += 1
        with self.subTest(self.i_subtest):
            assert error_90 <= 4.3
        self.i_subtest += 1

def _safe_regress(x,y):
    """

    :param x: x value
    :param y: y value
    :return: linregress object, unless error then None
    """
    try:
        return linregress(x=x,y=y)
    except ValueError:
        return None

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



if __name__ == '__main__':
    unittest.main()
