import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.shapereader as shpreader
import xarray as xr
import geopandas as gpd

def plot_maps(y_test_mean, predictions_mean, dif_percent):
    """
    Generate a figure with 1 row and 3 columns containing:
      - Map of the mean of y_test
      - Map of the mean of predictions
      - Map of the percentage difference between y_test and predictions
      
    Each map is plotted using contourf with contour lines and the India boundary overlaid.
    """
    
    # Set the map projection (PlateCarree is suitable for geographic data)
    projection = ccrs.PlateCarree()
    
    # Create the figure and axes: 1 row x 3 columns
    fig, axes = plt.subplots(1, 3, figsize=(20, 6), subplot_kw={'projection': projection})
    
    # Load the India shapefile from the local directory
    shapefile = r'C:\Users\Usuario\Documents\Python Scripts\angelic_lothus\data\India_Shape\india_ds.shp'
    india_shp = gpd.read_file(shapefile)

    print(india_shp.crs)  # Para verificar o CRS do shapefile
    print(y_test_mean.coords)  # Para verificar as coordenadas do xarray
    print(predictions_mean.coords)  # Para verificar as coordenadas do xarray

    if india_shp.crs is None:
    # Substitua "EPSG:XXXX" pelo CRS correto do shapefile (por exemplo, EPSG:4326 para lat/lon)
        india_shp = india_shp.set_crs(epsg=4326)

    # Reproject shapefile to PlateCarree CRS if necessary
    if india_shp.crs is not None and india_shp.crs.to_string() != 'EPSG:4326':
        india_shp = india_shp.to_crs(epsg=4326)  # EPSG:4326 é equivalente ao PlateCarree

    print(india_shp.crs)  # Para verificar o CRS do shapefile
    
    # List of datasets and titles for each map
    datasets = [y_test_mean, predictions_mean, dif_percent]
    titles = ["Mean of y_test", "Mean of Predictions", "Percentage Difference (%)"]
    cmaps = ['viridis', 'viridis', 'coolwarm']
    limits = [[0, 6], [0, 6], [-25, 25]]
    
    # Loop over each axis and dataset
    for ax, data, title, cmap, limit in zip(axes, datasets, titles, cmaps, limits):
        # Extract coordinates: assuming dimensions are 'lon' and 'lat'
        lon = data['lon']
        lat = data['lat']
        
        # Define explicit levels based on the provided limits
        levels = np.linspace(limit[0], limit[1], 10)
        
        # Plot filled contours (contourf) with defined levels
        cf = ax.contourf(lon, lat, data, levels=levels, transform=ccrs.PlateCarree(), cmap=cmap)
        
        # Add coastlines and gridlines
        ax.coastlines(resolution='10m', color='black', linewidth=1)
        gl = ax.gridlines(draw_labels=True, linestyle='--', alpha=0.5)
        gl.top_labels = False
        gl.right_labels = False
        
        # Add the India shapefile to the current axis
        india_shp.plot(ax=ax, edgecolor='black', facecolor='none', linewidth=1.5, transform=ccrs.PlateCarree())

        # Set the title for the current axis
        ax.set_title(title, fontsize=12)
        
        # Add a colorbar for the current plot
        cbar = fig.colorbar(cf, ax=ax, orientation='vertical', shrink=0.7)
    
    plt.tight_layout()
    plt.show()