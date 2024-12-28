"""
Testing module for hill fitter
"""
import unittest
import warnings
import platform
import json
import logging
import tempfile
import pandas
from tqdm import tqdm
from click.testing import CliRunner
import numpy as np
import plotly.graph_objects as go
from scipy.stats import linregress
import dghf
from scripts import canvass_download
from scripts import simulated_data
from scripts import prh_profile

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MyTestCase(unittest.TestCase):

    def __init__(self,*args,**kw):
        """

        :param args:  see unittest.TestCase __init__
        :param kw: see unittest.TestCase __init__
        """
        super().__init__(*args,**kw)
        self.i_subtest = 0

    @classmethod
    def setUpClass(cls):
        """
        download the data and make the simulated data shared by all the tests
        """
        np.random.seed(42)
        cls.simulated_data = simulated_data.simulated_data()
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
            kw_for_fit = {'x':x, 'y':y, 'bounds_n':[0,np.inf]}
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
            median()["Time/curve (s/fit)"]
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

    def test_05_demo_data(self):
        """
        make sure the demo plotting works
        """
        all_cids = (canvass_download.inactive_cid_assays() +
                    canvass_download.exemplar_cid_assays())
        x_y = canvass_download.demo_x_y_data(df=MyTestCase.df_canvass,
                                             cid_assay=all_cids)
        all_fit_kw = [dghf.fit(x, y) for x, y in x_y]
        dghf.gallery_plot(x_y=x_y, all_fit_kw=all_fit_kw)

    def test_98_canvass_subset(self):
        runner = CliRunner()
        x_y_dict = dghf._ids_to_xy(in_df=MyTestCase.df_canvass)
        subset = sorted(x_y_dict.keys())[:30]
        x_y = [x_y_dict[i] for i in subset]
        df_subset = MyTestCase.df_canvass[MyTestCase.df_canvass["Curve ID"].isin(subset)].\
            sort_values(by="Curve ID",ignore_index=True)
        bounds_n = [0,np.inf]
        df_fit_directly = pandas.DataFrame([dghf.fit(**tmp,bounds_n=bounds_n)
                                            for tmp in x_y])
        # test the full stack
        with tempfile.NamedTemporaryFile(suffix=".csv") as in_f, \
             tempfile.NamedTemporaryFile(suffix=".csv") as out_f_csv, \
             tempfile.NamedTemporaryFile(suffix=".json") as out_f_json:
            # output a subset of the data
            df_subset.to_csv(in_f.name, index=False)
            arg_bounds = ['--bounds_n',*bounds_n]
            # call the function for csv
            self._invoke(runner, args=["fit-file",
                                       "--input_file", in_f.name,
                                       "--output_file",out_f_csv.name,
                                       *arg_bounds])
            df_found_csv = pandas.read_csv(out_f_csv.name)
            # call the function for json
            runner.invoke(dghf.cli,["fit-file",
                                         "--input_file",in_f.name,
                                         "--output_file",out_f_json.name,
                                         *arg_bounds],
                          catch_exceptions=False)
            # read in the json
            with open(out_f_json.name,'r',encoding="utf8") as f:
                json_found = json.load(f)
            # make sure CSV = JSON = what we would expect from calling directly
            df_json = pandas.DataFrame(json_found)
            cols_to_check = ["min_v", "max_v", "log_K_a", "n"]
            # make sure that the json and csv data are the same
            with self.subTest(i=self.i_subtest):
                np.testing.assert_allclose(df_json[cols_to_check],
                                           df_found_csv[cols_to_check])
            self.i_subtest += 1
            # directly check that if we call the code directly the results
            # are the same (with 5%, since algorithms are random)
            with self.subTest(i=self.i_subtest):
                np.testing.assert_allclose(df_json[cols_to_check],
                                           df_fit_directly[cols_to_check],rtol=0.05)
            self.i_subtest += 1
            # redundant but go ahead and check csv as well
            with self.subTest(i=self.i_subtest):
                np.testing.assert_allclose(df_found_csv[cols_to_check],
                                           df_fit_directly[cols_to_check],rtol=0.05)
            self.i_subtest += 1

    def test_99_canvass(self):
        """
        Test the canvass data set, makng sure the residuals and R2 look OK
        """
        self.i_subtest = 0
        x_y, _, all_fit_kw = dghf._fit_df(in_df=MyTestCase.df_canvass,
                                          n_jobs=-1,col_x="Concentration (M)",
                                          col_y="Activity (%)",col_id="Curve ID")
        n_curves = len(x_y)
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


    def _invoke(self,runner, args):
        """

        :param runner: cli runner
        :param args: argumnets to provide to runner
        :return:  nothing, tests the function returns no error
        """
        with self.subTest(i=self.i_subtest):
            result = runner.invoke(dghf.cli, args, catch_exceptions=False)
        self.i_subtest += 1
        with self.subTest(i=self.i_subtest):
            assert result.exit_code == 0
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





if __name__ == '__main__':
    unittest.main()
