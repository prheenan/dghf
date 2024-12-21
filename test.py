"""
Testing module for hill fitter
"""
import unittest
import warnings

import numpy as np
import plotly.graph_objects as go
import dghf
from scripts import canvass_download

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
        bounds = [[None,None],[None,None],[None,None],[0,np.inf]]
        for (x,y,kw_expected),kw_err in x_y_k_err[::-1]:
            kw_fit = dghf.fit(x,y,bounds=bounds)
            self._assert_close_kw(kw_found=kw_fit, kw_expected=kw_expected,
                                  **kw_err)

    def test_overflow_inducing_runs(self):
        """
        test runs which should induced overflow
        """
        self.i_subtest = 0
        x_y_k_err = self.simulated_data
        bounds = [[None,None],[None,None],[None,None],[0,np.inf]]
        for (x,y,_),__ in x_y_k_err[::-1]:
            with self.subTest(i=self.i_subtest):
                with warnings.catch_warnings(category=RuntimeWarning):
                    warnings.simplefilter("error",category=RuntimeWarning)
                    dghf.fit(x,y,bounds=bounds,coarse_n=2)
            self.i_subtest += 1

    def test_bounds(self):
        """
        Tested bounded fit problems
        """
        x_y_k_err = self.simulated_data
        for (x,y,kw_expected),kw_err in x_y_k_err[::-1]:
            for i,name in enumerate(dghf.param_names_order()):
                bounds = [[None, None], [None, None], [None, None], [0, np.inf]]
                bounds[i] = [kw_expected[name],kw_expected[name]]
                kw_fit = dghf.fit(x,y,bounds=bounds)
                self._assert_close_kw(kw_found=kw_fit, kw_expected=kw_expected,
                                      **kw_err)

    def test_canvass(self):
        """
        Test the canvass data set

        import plotly
        np.bool8 = np.bool
        from plotly import express as px
        df_canvass["[M]"] = df_canvass["Concentration (M)"]
        df_canvass["%"] = df_canvass["Activity (%)"]
        df_canvass["ID"] = df_canvass["PUBCHEM_CID"].astype(int)
        df_canvass["assay"] = df_canvass["Assay"].astype(str)
        fig = px.line(df_canvass,log_x=True,x="[M]",facet_col_wrap=20,
                      facet_row_spacing=0.01,
                      y="%",facet_col="ID",color="assay",height=2000,width=2000,
                      facet_col_spacing=0.01,range_y=[-100,200])
        fig.show()
        """
        canvass_download.read_canvass_data(out_dir="./out/test/cache_canvass")


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
