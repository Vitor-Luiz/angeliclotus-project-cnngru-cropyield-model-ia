# Imports

# DataTree Imports
import os
import numpy as np
import pandas as pd
import xarray as xr
import intake
import fsspec
import cftime
from xarray import DataTree
from xarray.coding.cftimeindex import CFTimeIndex
from xmip.preprocessing import combined_preprocessing 
from xmip.postprocessing import _parse_metric
from collections import Counter
import gc

# Functions

def build_facet_dict(model=None, variables=None, table=None, experiments=None):

    facet_dict = { 
        "source_id": model,
        "variable_id": variables,
        #"member_id":"r1i1p1f1",  # all members
        "table_id": table,
        "grid_label": "gn",
        "experiment_id": experiments,
        "require_all_on":"source_id"
        }
    
    return facet_dict

def preprocess_time(ds, experiment_id, calendar="proleptic_gregorian", start_date="1901-01-01", end_date="2012-12-31"):
    """
    Preprocess a dataset to standardize its time dimension and compute annual mean.
    
    Parameters:
    - ds (xr.Dataset): Input dataset to preprocess.
    - experiment_id (str): Experiment identifier (e.g., 'historical', 'ssp585').
    - calendar (str): Calendar to enforce.
    - start_date (str): Start date for slicing (used only for historical data).
    - end_date (str): End date for slicing (used only for historical data).
    
    Returns:
    - xr.Dataset: Preprocessed dataset with annual means.
    """
    # Ensure time is monotonic
    if not ds.indexes["time"].is_monotonic_increasing:
        ds = ds.sortby("time")
    
    # Handle CFTimeIndex and enforce calendar
    if isinstance(ds.indexes["time"], CFTimeIndex):

        # Convert to the desired calendar
        ds = ds.convert_calendar(calendar, align_on="year")

        # Apply slicing only for the historical experiment
        if experiment_id == "historical":
            ds = ds.sel(time=slice(start_date, end_date))
    else:
        ds["time"] = ds.indexes["time"].to_datetimeindex()
        if experiment_id == "historical":
            ds = ds.sel(time=slice(start_date, end_date))
    
    # Compute annual mean
    ds = ds.resample(time="YS").mean()
    
    return ds

def readin_cmip6_to_datatree(facet_dict):
    """
    Read in CMIP6 data and convert it to a DataTree structure.
    
    Parameters:
    - facet_dict (dict): Dictionary of facets to filter the CMIP6 catalog.
    
    Returns:
    - dict: A DataTree structure containing the datasets.
    """
    # Open an intake catalog containing the Pangeo CMIP cloud data
    col = intake.open_esm_datastore("https://storage.googleapis.com/cmip6/pangeo-cmip6.json")
    
    # Create a subset of the catalog using facet search
    cat = col.search(
        source_id=facet_dict["source_id"],
        variable_id=facet_dict["variable_id"],
        table_id=facet_dict["table_id"],
        grid_label=facet_dict["grid_label"],
        experiment_id=facet_dict["experiment_id"],
        require_all_on=facet_dict["require_all_on"]
    )
    
    # Set aggregation attributes
    cat.esmcat.aggregation_control.groupby_attrs = ["source_id", "experiment_id"]
    
    # Define preprocessing function with experiment_id
    kwargs = dict(
        preprocess=lambda ds: preprocess_time(
            ds, experiment_id=facet_dict["experiment_id"]
        ),  # Apply time preprocessing
        xarray_open_kwargs=dict(use_cftime=True),  # Ensure all datasets use the same time index
        storage_options={"token": "anon"},  # Anonymous/public authentication to Google Cloud Storage
    )
    
    # Convert the sub-catalog into a dictionary of datasets
    dt = cat.to_dataset_dict(**kwargs)
    
    return dt

def add_ensemble_mean(ds):
    """
    Add a member_id_mean variable to the dataset, which is the mean across all ensemble members.
    This calculates the mean for each data variable individually.
    """
    # Cria um novo dataset com as médias das variáveis ao longo de 'member_id'
    mean_ds = ds.mean(dim="member_id", keep_attrs=True)
    
    # Adiciona cada variável média ao dataset original com um sufixo "_mean"
    for var_name in mean_ds.data_vars:
        ds[f"{var_name}_mean"] = mean_ds[var_name]
    
    return ds

def adjust_coordinates(ds):
    """
    Adjust longitude and latitude coordinates for a dataset:
    - Longitude adjusted to range [-180, 180].
    - Latitude adjusted to range [-90, 90].
    - Handles cases where coordinates are 1D, 2D, or 3D.
    - Removes member_id for 3D coordinates and flattens x and y for 2D or 3D.
    - Sorts by longitude after adjustments.
    
    Parameters:
    - ds (xarray.Dataset): The input dataset to be adjusted.
    
    Returns:
    - xarray.Dataset: The dataset with adjusted latitude and longitude.
    """
    # Identify the names of longitude and latitude coordinates
    longitude_name = None
    latitude_name = None

    for coord in ds.coords:
        if coord in ["lon", "longitude"]:
            longitude_name = coord
        elif coord in ["lat", "latitude"]:
            latitude_name = coord

    # Adjust longitude
    if longitude_name:
        if ds[longitude_name].ndim == 3:  # Case 3D
            # Remove member_id dimension, flatten x and y
            adjusted_longitude = ds[longitude_name].isel(member_id=0).values.flatten()
            adjusted_longitude = ((adjusted_longitude + 180) % 360 - 180)  # Apply the equation
            ds = ds.assign_coords({longitude_name: adjusted_longitude})
        elif ds[longitude_name].ndim == 2:  # Case 2D
            # Flatten x and y
            adjusted_longitude = ds[longitude_name].values.flatten()
            adjusted_longitude = ((adjusted_longitude + 180) % 360 - 180)  # Apply the equation
            ds = ds.assign_coords({longitude_name: adjusted_longitude})
        elif ds[longitude_name].ndim == 1:  # Case 1D
            # Only adjust to the range [-180, 180]
            ds = ds.assign_coords({longitude_name: ((ds[longitude_name] + 180) % 360 - 180)})

        # Sort by longitude
        ds = ds.sortby(longitude_name)
    else:
        print("No longitude coordinate ('lon' or 'longitude') found in the dataset.")

    # Adjust latitude
    if latitude_name:
        if ds[latitude_name].ndim == 3:  # Case 3D
            # Remove member_id dimension, flatten x and y
            adjusted_latitude = ds[latitude_name].isel(member_id=0).values.flatten()
            ds = ds.assign_coords({latitude_name: adjusted_latitude})
        elif ds[latitude_name].ndim == 2:  # Case 2D
            # Flatten x and y
            adjusted_latitude = ds[latitude_name].values.flatten()
            ds = ds.assign_coords({latitude_name: adjusted_latitude})
        elif ds[latitude_name].ndim == 1:  # Case 1D
            # No adjustment needed for latitude, as it is already in [-90, 90]
            pass

        # Sort by latitude
        ds = ds.sortby(latitude_name)
    else:
        print("No latitude coordinate ('lat' or 'latitude') found in the dataset.")

    return ds

def crop_to_india(ds):
    """
    Crop dataset to the latitude and longitude bounds of India.
    """
    return ds.sel(lat=slice(5, 40), lon=slice(65, 98))

def process_and_export(dt, output_dir, crop_to_india_applicable=True):
    """
    Process and export CMIP6 datasets, with optional cropping to India's region.
    
    Parameters:
    - dt: DataTree containing datasets.
    - output_dir: Directory to save the NetCDF files.
    - crop_to_india_applicable: Boolean flag to apply cropping only to specific datasets.
    """

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    for key, ds in dt.items():
        # Adjust longitude
        ds = adjust_coordinates(ds)
        
        # Crop to India's region, if applicable
        if crop_to_india_applicable:
            ds = crop_to_india(ds)
                
        # Add ensemble means (does not compute inter-model means)
        ds = add_ensemble_mean(ds)
        
        # Create the file path for saving the NetCDF file
        filename = f"{key}.nc"  # File name based on the DataTree key
        filepath = os.path.join(output_dir, filename)
        
        # Save the dataset in NetCDF format
        ds.to_netcdf(filepath)
        print(f"Saved: {filepath}")

        # Deleting ds for optimization
        del ds
        gc.collect()

# Function to process MIROC6 data
def process_miroc6(filepath, output_path, experiment):
    """
    Process MIROC6 data to calculate Nino3.4 SST anomalies and Dipole Mode Index (DMI).
    Args:
        filepath (str): Path to the MIROC6 dataset.
        output_path (str): Directory to save the output files.
    """
    # Open dataset
    ds = xr.open_dataset(filepath)

    # Convert longitude
    ds["x"] = ((ds["x"] + 180) % 360 - 180)
    ds = ds.sortby("x")

    # Nino3.4 region
    nino34_ds = ds.tos_mean.sel(y=slice(-5, 5), x=slice(-170, -120))
    nino34_mean = nino34_ds.squeeze("dcpp_init_year").mean(dim=["x", "y"])

    # Historical period
    nino34_hist = nino34_mean.sel(time=slice('1850-01-01T00:00:00.000000000', '1900-01-01T00:00:00.000000000'))

    # Anomalies
    nino34_anom = nino34_mean.sel(time=slice('1901-01-01T00:00:00.000000000', '2012-01-01T00:00:00.000000000'))
    anomalies = nino34_anom - nino34_hist.mean(dim="time")

    # Save anomalies
    anomalies_df = anomalies.to_dataframe()
    anomalies_df.to_csv(f"{output_path}/miroc6_nino34_anomalies.{experiment}.csv")

    # Calculate DMI
    western_iod = ds.tos_mean.sel(y=slice(-10, 10), x=slice(50, 70)).squeeze("dcpp_init_year").mean(dim=["x", "y"])
    eastern_iod = ds.tos_mean.sel(y=slice(-10, 0), x=slice(90, 110)).squeeze("dcpp_init_year").mean(dim=["x", "y"])
    dmi = western_iod - eastern_iod

    # Save DMI
    dmi_df = dmi.to_dataframe()
    dmi_df.to_csv(f"{output_path}/miroc6_dmi.{experiment}.csv")

# Function to process NorESM-MM data
def process_noresm(filepath, output_path, experiment):
    """
    Process NorESM-MM data to calculate Nino3.4 SST anomalies and Dipole Mode Index (DMI).
    Args:
        filepath (str): Path to the NorESM-MM dataset.
        output_path (str): Directory to save the output files.
    """
    # Open dataset
    ds = xr.open_dataset(filepath)

    # Convert longitude and latitude
    ds["i"] = ((ds["i"] + 180) % 360 - 180)
    ds = ds.sortby("i")
    ds["j"] = ((ds["j"] + 90) % 385 - 90)
    ds = ds.sortby("j")

    # Nino3.4 region
    nino34_ds = ds.tos_mean.sel(j=slice(-5, 5), i=slice(-170, -120))
    nino34_mean = nino34_ds.squeeze("dcpp_init_year").mean(dim=["i", "j"])

    # Historical period
    nino34_hist = nino34_mean.sel(time=slice('1850-01-01T00:00:00.000000000', '1900-01-01T00:00:00.000000000'))

    # Anomalies
    nino34_anom = nino34_mean.sel(time=slice('1901-01-01T00:00:00.000000000', '2012-01-01T00:00:00.000000000'))
    anomalies = nino34_anom - nino34_hist.mean(dim="time")

    # Save anomalies
    anomalies_df = anomalies.to_dataframe()
    anomalies_df.to_csv(f"{output_path}/noresm_nino34_anomalies.{experiment}.csv")

    # Calculate DMI
    western_iod = ds.tos_mean.sel(j=slice(-10, 10), i=slice(50, 70)).squeeze("dcpp_init_year").mean(dim=["i", "j"])
    eastern_iod = ds.tos_mean.sel(j=slice(-10, 0), i=slice(90, 110)).squeeze("dcpp_init_year").mean(dim=["i", "j"])
    dmi = western_iod - eastern_iod

    # Save DMI
    dmi_df = dmi.to_dataframe()
    dmi_df.to_csv(f"{output_path}/noresm_dmi.{experiment}.csv")