# Local import
import func_cmip6 as fcmip6

# Lists Models
## Atmospheric Label
# Models
atm_MODELS = ["NorESM2-MM", "MIROC6"]
# List of CMIP6 experiments
atm_EXPERIMENTS = ["historical", "ssp245", "ssp585"]
# 
# List of variables to retrieve
atm_VARIABLES = ["tas", "pr", "hurs"]  # tas: temperature, pr: precipitation rate, tos: sea surface temperature
#
atm_TABLE = "Amon"

## Oceanic Label
# Models
ocean_MODELS = ["NorESM2-MM", "MIROC6"]
# List of CMIP6 experiments
ocean_EXPERIMENTS = ["historical", "ssp245", "ssp585"]
# 
# List of variables to retrieve
ocean_VARIABLES = "tos"  # tos: sea surface temperature
#
ocean_TABLE = "Omon"

## Land Label
# Models
land_MODELS = ["NorESM2-MM", "MIROC6"]
# List of CMIP6 experiments
land_EXPERIMENTS = ["historical", "ssp585"]
# 
# List of variables to retrieve
land_VARIABLES = "mrsos"  # mrsos: soil moisture in the surface layer
#
land_TABLE = "day"

# Dictinaries
# Atmospheric
atm_dict = fcmip6.build_facet_dict(model=atm_MODELS, variables=atm_VARIABLES, table=atm_TABLE, experiments=atm_EXPERIMENTS)

# Oceanic
ocean_dict = fcmip6.build_facet_dict(model=ocean_MODELS, variables=ocean_VARIABLES, table=ocean_TABLE, experiments=ocean_EXPERIMENTS)

# Land
land_dict = fcmip6.build_facet_dict(model=land_MODELS, variables=land_VARIABLES, table=land_TABLE, experiments=land_EXPERIMENTS)

## DataTree

dt_atm = fcmip6.readin_cmip6_to_datatree(atm_dict)
dt_ocean = fcmip6.readin_cmip6_to_datatree(ocean_dict)
dt_land = fcmip6.readin_cmip6_to_datatree(land_dict)

## Keys

# Atmospheric
print("Atmospheric")
print([key for key in dt_atm.keys()])

# Oceanic
print("Oceanic")
print([key for key in dt_ocean.keys()])

# Land
print("Land")
print([key for key in dt_land.keys()])

# Process Data
fcmip6.process_and_export(dt_atm,
                          output_dir=r"C:\Users\Usuario\Documents\Python Scripts\angelic_lothus\cmip6_data_atm", 
                          crop_to_india_applicable=True)

fcmip6.process_and_export(dt_ocean,
                          output_dir=r"C:\Users\Usuario\Documents\Python Scripts\angelic_lothus\cmip6_data_ocean_oficial", 
                          crop_to_india_applicable=False)

fcmip6.process_and_export(dt_land,
                          output_dir=r"C:\Users\Usuario\Documents\Python Scripts\angelic_lothus\cmip6_data_land", 
                          crop_to_india_applicable=True)

# Usage
miroc6_files = [r"C:\Users\Usuario\Documents\Python Scripts\angelic_lothus\cmip6_data_ocean_oficial\MIROC6.historical.nc", 
                r"C:\Users\Usuario\Documents\Python Scripts\angelic_lothus\cmip6_data_ocean_oficial\MIROC6.ssp245.nc",
                r"C:\Users\Usuario\Documents\Python Scripts\angelic_lothus\cmip6_data_ocean_oficial\MIROC6.ssp585.nc"]

noresm_files = [r"C:\Users\Usuario\Documents\Python Scripts\angelic_lothus\cmip6_data_ocean_oficial\NorESM2-MM.historical.nc", 
                r"C:\Users\Usuario\Documents\Python Scripts\angelic_lothus\cmip6_data_ocean_oficial\NorESM2-MM.ssp245.nc", 
                r"C:\Users\Usuario\Documents\Python Scripts\angelic_lothus\cmip6_data_ocean_oficial\NorESM2-MM.ssp585.nc"]

experiments = ["historical", "ssp245", "ssp585"]

output_dir = r"C:\Users\Usuario\Documents\Python Scripts\angelic_lothus\cmip6_data_ocean"

for file, experiment in zip(miroc6_files, experiments):
    fcmip6.process_miroc6(file, output_dir, experiment)

for file, experiment in zip(noresm_files, experiments):
    fcmip6.process_noresm(file, output_dir, experiment)