"""
Module to help download all the canvass data
"""
import time
import os
import re

import numpy as np
import requests
import pandas
from tqdm import tqdm


def process_canvass_df(d,aid=""):
    """

    :param d: pubchem data frame given by
    f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/assay/aid/{aid}/CSV?sid={csv}"
    :param aid: assay id
    :return: processed data frame with the columns we care about
    """
    df_skiprow = d.iloc[1:, :].copy()
    cols = list(df_skiprow.columns)
    cols_activity = [c for c in cols if c.startswith("Activity at ")]
    all_matches = [re.match(r"Activity at ([\.\d+]+) uM",c)
                   for c in cols_activity]
    assert all(m is not None for m in all_matches)
    activity_to_concentration_M = { c:float(m.group(1))/1e6
                                    for c,m in zip(cols_activity,all_matches)}
    cols_meta = ['PUBCHEM_ACTIVITY_OUTCOME', 'Curve_Description',
                 'Fit_LogAC50', 'Fit_HillSlope', 'Fit_R2']
    cols_id = ['PUBCHEM_ACTIVITY_URL','PUBCHEM_SID',
               'PUBCHEM_RESULT_TAG','PUBCHEM_CID']
    df_skiprow["Curve ID"] = ["___".join(str(row[c]) for c in cols_id
                                         if str(row[c]) != "nan") + f"___{aid}"
                              for i,row in df_skiprow.iterrows()]
    df_melted = pandas.melt(df_skiprow[ ["Curve ID"] + cols_activity],
                            id_vars="Curve ID",
                            value_vars=cols_activity, var_name="Concentration",
                            value_name="Activity (%)").dropna(subset=["Activity (%)"]).copy()
    df_melted["Concentration (M)"] = df_melted["Concentration"].map(activity_to_concentration_M)
    df_melted["Activity (%)"] = df_melted["Activity (%)"].astype(float)
    df_to_ret = df_skiprow[["Curve ID"] + cols_id + cols_meta].\
        merge(df_melted.drop("Concentration",axis="columns"),on="Curve ID")
    return df_to_ret.sort_values(by=["Curve ID","Concentration (M)"],ignore_index=True)

def read_canvass_data(out_dir="./out/test/cache_canvass",aids=None,
                      random_sample=None,random_seed=None,n_max_assays=None):
    """
    The default aids were found as follows

    - https://tripod.nih.gov/canvass/ to get the paper title used below
    - Search for the following in pubchem
    - Canvass: A Crowd-Sourced, Natural-Product Screening Library ...
    - "Linked Data sets" -> Bioactivities -> Summary (Search Results) -> CSV
    - Copy unique aids
    - There may be a way to get them programatically (TBD?)

    :param out_dir: where to output/cache the data. If None (not recommended), won't cache
    :param aids: list of ais to use (defaults to canvass
    :param random_sample: how many to randomly sample without replacement
    :param random_seed: only used if random sample is not none, passed to np.random.seed
    :param n_max_assays: maximum number of assays to read. If none, reads all
    :return: all canvass data
    """
    if out_dir is not None:
        if not os.path.isdir(out_dir):
            os.makedirs(out_dir)
    if aids is None:
        aids = [1347345,
                1347346,
                1347347,
                1347348,
                1347349,
                1347350,
                1347351,
                1347352,
                1347353,
                1347354,
                1347355,
                1347356,
                1347357,
                1347358,
                1347359,
                1347360,
                1347361,
                1347362,
                1347363,
                1347364,
                1347365,
                1347366,
                1347367,
                1347368,
                1347369,
                1347370,
                1347371,
                1347372,
                1347373,
                1347374,
                1347375,
                1347376,
                1347377,
                1347378,
                1347379,
                1347380,
                1347381,
                1347387,
                1347388,
                1347389,
                1347390,
                1347391,
                1347392,
                1347393,
                1347394,
                1347396,
                1347400,
                1347401,
                1347402]
    time_sleep = 4
    all_dfs = []
    for i_aid,aid in enumerate(tqdm(aids)):
        if n_max_assays is not None and i_aid >= n_max_assays:
            break
        file_v = os.path.join(out_dir,f"{aid}.csv") if out_dir is not None else None
        if file_v is not None and (not os.path.isfile(file_v)):
            return_v = requests.\
                get(f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/assay/aid/{aid}/CSV",
                    timeout=60)
            assert return_v.ok
            if file_v is not None:
                # then save it out
                with open(file_v,'w',encoding="utf-8") as f:
                    f.write(return_v.text)
            time.sleep(time_sleep)
        # file exists
        df_csv = process_canvass_df(pandas.read_csv(file_v),aid)
        df_csv.insert(loc=0,column="Assay",value=aid)
        all_dfs.append(df_csv)
    df_cat = pandas.concat(d for d in all_dfs)
    if random_sample is not None:
        if random_seed is not None:
            np.random.seed(random_seed)
        random_ids = np.random.choice(sorted(set(df_cat["Curve ID"])),
                                      size=random_sample,replace=False)
        df_cat = df_cat[df_cat["Curve ID"].isin(random_ids)].copy()
    return df_cat

def read_xy_from_assay_cid(df,cid_assay=None):
    """

    :param df:  data frame like from read_canvass_data
    :param cid_assay: list, length N, of cids/assay pairs, like exemplar_cid_assays
    :return: list, length N, each element is X and Y
    """
    if cid_assay is None:
        cid_assay = exemplar_cid_assays()
    dict_cid_assay = {(cid, a): [df_i["Concentration (M)"].to_numpy(),
                                 df_i["Activity (%)"].to_numpy()]
                      for (cid, a), df_i in df.groupby(["PUBCHEM_CID", "Assay"])}
    data_subset = [dict_cid_assay[float(c), int(a)]
                   for c,a in cid_assay]
    return data_subset

def inactive_cid_assays():
    """

    :return: list, each elemnet tuple of (compound id, assay) of inactive curves
    """
    return [
        [3037629, 1347357],
        [134827992, 1347352],
        [134828028, 1347368],
        [134827947,1347371],
        [12110448, 1347364],
        [73659,1347370],
        [134827997,1347375]
    ]


def exemplar_cid_assays():
    """

    :return: list, each elemnet tuple of (compound id, assay) of active curves
    """
    return [
        [21123718, 1347368],
        [71452522, 1347364],
        [638024, 1347373],
        [44543726, 1347391],
        [11169934, 1347365],
        [134827994, 1347393],
        [134827991, 1347351],
        [134828011, 1347391],
        [135419370, 1347350]
    ]

def demo_x_y_data(df=None,cid_assay=None):
    """

    :param df: dataframe to read from; defaults to all canvass data
    :param cid_assay: list of length N; each element like <cid, assay>
    :return: list of length N, each element is x and y
    """
    if df is None:
        df = read_canvass_data()
    if cid_assay is None:
        cid_assay = exemplar_cid_assays()
    x_y = read_xy_from_assay_cid(df, cid_assay=cid_assay)
    return x_y

def run():
    """

    :return:
    """
    df_cat = read_canvass_data(out_dir="../out/test/cache_canvass")
    df_cat.to_csv("canvass_data.csv",index=False)


if __name__ == "__main__":
    run()
