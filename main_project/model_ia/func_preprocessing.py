# Imports

import os
import numpy as np
import pandas as pd
import xarray as xr
import intake
import fsspec
import cftime
from xarray.coding.cftimeindex import CFTimeIndex
from collections import Counter
import gc

# Functions
def read_cmip6_spacialized(path):
    ds = xr.open_dataset(path)
    ds = ds.squeeze(dim="dcpp_init_year")
    ds = ds.sel(time=slice('1901-01-01T00:00:00.000000000', '2012-01-01T00:00:00.000000000'))
    ds = ds.sel(lat=slice(5, 38.2), lon=slice(65, 98))

    return ds

def read_cmip6_ocean_index(path_dmi, path_enso):
    # DMI
    df_dmi = pd.read_csv(path_dmi)
    df_dmi["time"] = pd.to_datetime(df_dmi["time"])
    df_dmi = df_dmi.drop(columns=["dcpp_init_year"])
    df_dmi.set_index("time", inplace=True)
    df_dmi.rename(columns={"tos_mean": "miroc6_dmi"}, inplace=True)
    df_dmi = df_dmi.loc["1901-01-01":"2012-01-01"]
    # ENSO 3.4
    df_enso = pd.read_csv(path_enso)
    df_enso["time"] = pd.to_datetime(df_enso["time"])
    df_enso = df_enso.drop(columns=["dcpp_init_year"])
    df_enso.set_index("time", inplace=True)
    df_enso.rename(columns={"tos_mean": "miroc6_enso34"}, inplace=True)

    return df_dmi, df_enso

def read_crop_yield(path):
    new_time = pd.date_range(start="1901", end="2012", freq="YS")  # YS = Year Start

    # Crop model
    ds = xr.open_dataset(path, decode_times=False)
    ds = ds.assign_coords(time=new_time)
    ds = ds.sortby("lat")
    ds = ds.sel(lat=slice(5, 38.2), lon=slice(65, 98))

    return ds

def interpolation_ds(ds_ref, ds_interp):
    ds_ref = ds_ref.interp(coords={"lon": ds_interp.lon.values, 
                                   "lat": ds_interp.lat.values},
                                   method="linear")
    
    if "yield_ric" in ds_ref:
        ds_ref["yield_ric"] = ds_ref["yield_ric"].fillna(-4)
    if "mrsos_mean" in ds_ref:
        ds_ref["mrsos_mean"] = ds_ref["mrsos_mean"].fillna(-40)

    print(ds_ref.dims)

    return ds_ref

def ocn_model_mean(df_list):
    df = pd.concat(df_list, axis=1)
    df_cnn = df
    df_cnn["dmi"] = (df["miroc6_dmi"] + df["noresm_dmi"]) / 2
    df_cnn["enso34"] = (df["miroc6_enso34"] + df["noresm_enso34"]) / 2

    df_cnn = df_cnn.drop(columns=["miroc6_dmi", "noresm_dmi", "miroc6_enso34", "noresm_enso34"])
    
    return df_cnn