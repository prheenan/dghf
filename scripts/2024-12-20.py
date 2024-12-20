import sys
import time
import os
import re
import requests
import pandas
from tqdm import tqdm
import dghf
sys.path.append("../")


def process_df(d):
    """

    :param d: pubchem data frame given by
    f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/assay/aid/{aid}/CSV?sid={csv}"
    :return: processed data frame with the columns we care about
    """
    df_skiprow = d.iloc[1:, :].copy()
    cols = list(df_skiprow.columns)
    cols_activity = [c for c in cols if c.startswith("Activity at ")]
    all_matches = [re.match(r"Activity at ([\.\d+]+) uM",c)
                   for c in cols_activity]
    assert all(m is not None for m in all_matches)
    activity_to_concentration_M = dict([ [c,float(m.group(1))/1e6]
                                         for c,m in zip(cols_activity,all_matches)])
    cols_meta = ['PUBCHEM_ACTIVITY_OUTCOME', 'Curve_Description',
                 'Fit_LogAC50', 'Fit_HillSlope', 'Fit_R2']
    cols_id = ['PUBCHEM_ACTIVITY_URL','PUBCHEM_SID',
               'PUBCHEM_CID', 'PUBCHEM_RESULT_TAG']
    df_skiprow["Curve ID"] = ["___".join(str(row[c]) for c in cols_id
                                         if str(row[c]) != "nan")
                              for i,row in df_skiprow.iterrows()]
    df_melted = pandas.melt(df_skiprow[ ["Curve ID"] + cols_activity],
                            id_vars="Curve ID",
                            value_vars=cols_activity, var_name="Concentration",
                            value_name="Activity (%)").dropna(subset=["Activity (%)"]).copy()
    df_melted["Concentration (M)"] = df_melted["Concentration"].map(activity_to_concentration_M)
    df_melted["Activity (%)"] = df_melted["Activity (%)"].astype(float)
    df_to_ret = df_skiprow[["Curve ID"] + cols_meta].\
        merge(df_melted.drop("Concentration",axis="columns"),on="Curve ID")
    return df_to_ret.sort_values(by=["Curve ID","Concentration (M)"],ignore_index=True)

def read_canvass_data(f_aid_and_sid,out_dir = "./out/cache_canvass"):
    """

    :param f_aid_and_sid: file of the aid and sids we care about
    :param out_dir: where to output the data
    :return:
    """
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    aids = set(pandas.read_csv(f_aid_and_sid)["aid"])
    time_sleep = 4
    all_dfs = []
    for aid in tqdm(aids):
        file_v = os.path.join(out_dir,f"{aid}.csv")
        if not os.path.isfile(file_v):
            return_v = requests.get(f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/assay/aid/{aid}/CSV",
                                    timeout=60)
            assert return_v.ok
            with open(file_v,'w',encoding="utf-8") as f:
                f.write(return_v.text)
            time.sleep(time_sleep)
        # file exists
        df_csv = process_df(pandas.read_csv(file_v))
        all_dfs.append(df_csv)
    df_cat = pandas.concat(d for d in all_dfs)
    return df_cat

def run():
    """
    This file was created in the following manner:

    - https://tripod.nih.gov/canvass/
    - Search for the following in pubchem
    - Canvass: A Crowd-Sourced, Natural-Product Screening Library for Exploring Biological Space
    - Click "Linked Data sets" -> Bioactivities -> Summary (Search Results) -> CSV
    - Then get unique aids
    - OR!
    - papaer hss pubmed ID Canvass: A Crowd-Sourced, Natural-Product Screening Library for Exploring Biological Space
    """
    df_cat = read_canvass_data(f_aid_and_sid="aid.csv",
                               out_dir="./out/cache_canvass")
    for _, df_curve in tqdm(df_cat.groupby("Curve ID")):
        dghf.fit(x=df_curve["Concentration (M)"].to_numpy(),
                 y=df_curve["Activity (%)"].to_numpy())


if __name__ == "__main__":
    run()
