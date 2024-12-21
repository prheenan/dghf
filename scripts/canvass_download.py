import time
import os
import re
import requests
import pandas
from tqdm import tqdm


def process_canvass_df(d):
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
    df_to_ret = df_skiprow[["Curve ID"] + cols_id + cols_meta].\
        merge(df_melted.drop("Concentration",axis="columns"),on="Curve ID")
    return df_to_ret.sort_values(by=["Curve ID","Concentration (M)"],ignore_index=True)

def read_canvass_data(out_dir="./out/cache_canvass",aids=None):
    """
    The default aids were found as follows

    - https://tripod.nih.gov/canvass/ to get the paper title used below
    - Search for the following in pubchem
    - Canvass: A Crowd-Sourced, Natural-Product Screening Library ...
    - "Linked Data sets" -> Bioactivities -> Summary (Search Results) -> CSV
    - Copy unique aids
    - There may be a way to get them programatically (TBD?)

    :param out_dir: where to output/cache the data. If None (not recommended), won't cache
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
    for aid in tqdm(aids):
        file_v = os.path.join(out_dir,f"{aid}.csv") if out_dir is not None else None
        if file_v is not None and (not os.path.isfile(file_v)):
            return_v = requests.get(f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/assay/aid/{aid}/CSV",
                                    timeout=60)
            assert return_v.ok
            if file_v is not None:
                # then save it out
                with open(file_v,'w',encoding="utf-8") as f:
                    f.write(return_v.text)
            time.sleep(time_sleep)
        # file exists
        df_csv = process_canvass_df(pandas.read_csv(file_v))
        df_csv.insert(loc=0,column="Assay",value=aid)
        all_dfs.append(df_csv)
    df_cat = pandas.concat(d for d in all_dfs)
    return df_cat

def run():
    """

    :return:
    """
    df_cat = read_canvass_data(out_dir="../out/test/cache_canvass")
    df_cat.to_csv("canvass_data.csv",index=False)


if __name__ == "__main__":
    run()
