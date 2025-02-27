# Spacial Imports
import xarray as xr
import numpy as np

# Model Imports
import tensorflow as tf
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split, ParameterGrid
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D, Flatten, LSTM, Dense, Dropout, Concatenate, GRU, RepeatVector, Reshape
)

# Visualization
import matplotlib.pyplot as plt

# Local Imports
import func_preprocessing as fpp
import func_model as fm
import func_analysis as fa

# Preprocessing Data
### Path Spacial Input Variabels

## Atmospheric variables + Land Variables
# MIROC6 historical and NorESM2-MM historical
list_atm_lnd = [r"C:\Users\Usuario\Documents\Python Scripts\angelic_lothus\cmip6_data_atm\MIROC6.historical.nc",
                r"C:\Users\Usuario\Documents\Python Scripts\angelic_lothus\cmip6_data_atm\NorESM2-MM.historical.nc",
                r"C:\Users\Usuario\Documents\Python Scripts\angelic_lothus\cmip6_data_land\MIROC6.historical.nc",
                r"C:\Users\Usuario\Documents\Python Scripts\angelic_lothus\cmip6_data_land\NorESM2-MM.historical.nc"]

### Ocean Variables - Dipole Mode Index + ENSO 3.4
list_ocn = [
            [r"C:\Users\Usuario\Documents\Python Scripts\angelic_lothus\cmip6_data_ocean\miroc6_dmi.historical.csv",
            r"C:\Users\Usuario\Documents\Python Scripts\angelic_lothus\cmip6_data_ocean\miroc6_nino34_anomalies.historical.csv",
            "miroc6"],
            [r"C:\Users\Usuario\Documents\Python Scripts\angelic_lothus\cmip6_data_ocean\noresm_dmi.historical.csv",
            r"C:\Users\Usuario\Documents\Python Scripts\angelic_lothus\cmip6_data_ocean\noresm_nino34_anomalies.historical.csv",
            "noresm"]
            ]

### Crop Yield variables - output
list_cyr = [r"C:\Users\Usuario\Documents\Python Scripts\angelic_lothus\agMIP_rice\EPIC-Boku\EPIC-Boku.Rice\EPIC-Boku.Rice\pgfv2\epic-boku_pgfv2_hist_default_firr_yield_ric_annual_1901_2012.nc4",
            r"C:\Users\Usuario\Documents\Python Scripts\angelic_lothus\agMIP_rice\LPJ-GUESS\LPJ-GUESS.Rice\LPJ-GUESS.Rice\pgfv2\lpj-guess_pgfv2_hist_default_firr_yield_ric_annual_1901_2012.nc4",
            r"C:\Users\Usuario\Documents\Python Scripts\angelic_lothus\agMIP_rice\LPJmL\LPJmL.Rice\LPJmL.Rice\pgfv2\lpjml_pgfv2_hist_default_firr_yield_ric_annual_1901_2012.nc4",
            r"C:\Users\Usuario\Documents\Python Scripts\angelic_lothus\agMIP_rice\PEPIC\PEPIC.Rice\PEPIC.Rice\pgfv2\pepic_pgfv2_hist_default_firr_yield_ric_annual_1901_2012.nc4"]

# Open Files
ds_atm_lnd = [fpp.read_cmip6_spacialized(path_atm_lnd) for path_atm_lnd in list_atm_lnd]

df_ocn = [list(fpp.read_cmip6_ocean_index(path_ocn[0], path_ocn[1], path_ocn[2])) for path_ocn in list_ocn]

ds_cyr = [fpp.read_crop_yield(path_cyr) for path_cyr in list_cyr]

# Interpolation for same grid points
interp_list = ds_atm_lnd + ds_cyr

interp_variables = [fpp.interpolation_ds(interp_list[0], ds_interp) for ds_interp in interp_list]

# Emsemble Model Mean

ds_cnn = interp_variables[0]
ds_cnn = ds_cnn.drop_vars(["hurs", "pr", "tas", "height", "dcpp_init_year"])
ds_cnn = ds_cnn.drop_dims("bnds")

ds_cnn["hurs_mean"] = (interp_variables[0]["hurs_mean"] + interp_variables[1]["hurs_mean"]) / 2
ds_cnn["pr_mean"] = (interp_variables[0]["pr_mean"] + interp_variables[1]["pr_mean"]) / 2
ds_cnn["tas_mean"] = (interp_variables[0]["tas_mean"] + interp_variables[1]["tas_mean"]) / 2
ds_cnn["mrsos_mean"] = (interp_variables[2]["mrsos_mean"] + interp_variables[3]["mrsos_mean"]) / 2
ds_cnn["yield_rice"] = (interp_variables[4]["yield_ric"] + interp_variables[5]["yield_ric"] + interp_variables[6]["yield_ric"] + interp_variables[7]["yield_ric"]) / 4

ds_cnn = ds_cnn.drop_vars(["member_id", "height", "dcpp_init_year", "depth"])

print(ds_cnn.yield_rice.max())
print(ds_cnn.yield_rice.min())

df_ocn = [item for sublist in df_ocn for item in sublist]
df_cnn = fpp.ocn_model_mean(df_ocn)
print(df_cnn)

#Convolutional Neural Network (CNN + GRU)
## Arrays
# Input
india_tas = ds_cnn["tas_mean"].values
india_pr = ds_cnn["pr_mean"].values
india_hurs = ds_cnn["hurs_mean"].values
india_mrsos = ds_cnn["mrsos_mean"].values
enso34 = df_cnn["enso34"].values
dmi = df_cnn["dmi"].values
# Output
yr = ds_cnn["yield_rice"].values

## Normalization
#Input
india_tas = fm.normalization(india_tas)
india_pr = fm.normalization(india_pr)
india_hurs = fm.normalization(india_hurs)
india_mrsos = fm.normalization(india_mrsos)
enso34 = fm.normalization(enso34)
dmi = fm.normalization(dmi)
# Output
yield_rice = fm.normalization(yr)

# Climate Inputs (tas_mean, pr_mean, hurs_mean, mrsos_mean)
X_climatic = np.stack([india_tas, india_pr, india_hurs, india_mrsos], axis=-1)

# Temporal Remote Imputs (ENSO34, DMI)
X_indices = np.stack([enso34, dmi], axis=-1)
# Add the sample dimension to create the correct shape (samples, timesteps, features)
X_indices = np.tile(X_indices, (112, 1, 1))  # Repeat for 112 samples

# Output (Rice Yield)
y = yield_rice
y = np.expand_dims(yield_rice, axis=-1)  # shape (112, 23, 23, 1)

print("Shape of X_climatic:", X_climatic.shape)  # (samples, 23, 23, 4)
print("Shape of X_indices:", X_indices.shape)    # (samples, 112, 2)
print("Shape of y:", y.shape)                    # (samples, lat, lon, 1)

# Split for training, validation, and testing
X_climatic_train, X_climatic_temp, X_indices_train, X_indices_temp, y_train, y_temp = train_test_split(
    X_climatic, X_indices, y, test_size=0.2, random_state=42
)

X_climatic_val, X_climatic_test, X_indices_val, X_indices_test, y_val, y_test = train_test_split(
    X_climatic_temp, X_indices_temp, y_temp, test_size=0.5, random_state=42
)

# Verify the shapes
print("X_climatic_train shape:", X_climatic_train.shape)
print("X_indices_train shape:", X_indices_train.shape)
print("y_train shape:", y_train.shape)

print("X_climatic_val shape:", X_climatic_val.shape)
print("X_indices_val shape:", X_indices_val.shape)
print("y_val shape:", y_val.shape)

print("X_climatic_test shape:", X_climatic_test.shape)
print("X_indices_test shape:", X_indices_test.shape)
print("y_test shape:", y_test.shape)

# Input 1: Spatial data (X_climatic)
input_spatial = Input(shape=(X_climatic.shape[1], X_climatic.shape[2], X_climatic.shape[3]))
x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_spatial)
x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
x = Flatten()(x)  # Flatten to combine with the temporal output later

# Input 2: Temporal indices (X_indices)
input_indices = Input(shape=(X_indices.shape[1], X_indices.shape[2]))  # Shape: (timesteps, features)
y = GRU(32, activation='relu', return_sequences=False)(input_indices)  # Extract temporal features - Gated Recurrent Unit (RNN)

# Combine both sub-networks
combined = Concatenate()([x, y])
z = Dense(1024, activation='relu')(combined)  # Combine features into a fully connected layer
z = Dense(X_climatic.shape[1] * X_climatic.shape[2], activation='linear')(z)  # Map to spatial dimensions

# Reshape back to (lat, lon, 1)
output = Reshape((X_climatic.shape[1], X_climatic.shape[2], 1))(z)

# Build the model
model = Model(inputs=[input_spatial, input_indices], outputs=output)

# Compile the model
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

print(model.summary())

# Training
history = model.fit(
    [X_climatic_train, X_indices_train],  # 2 Inputs
    y_train,  # output
    epochs=30,
    batch_size=32,
    validation_data=([X_climatic_val, X_indices_val], y_val)
)

# Evaluation
loss, mae = model.evaluate([X_climatic_test, X_indices_test], y_test, verbose=1)
print(f"Test Loss: {loss}, Test MAE: {mae}")

# Plot training and validation loss
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss (MSE)')
plt.legend()
plt.grid(True)
plt.show()

# Verifying Shapes
print("Shape de X_climatic_test:", X_climatic_test.shape)  # have to be (n_samples, 23, 23, 4)
print("Shape de X_indices_test:", X_indices_test.shape)    # have to be (n_samples, 112, 2)
print("Shape de y_test:", y_test.shape)                    # have to be (n_samples, 23, 23, 1)

# Evaluating the model
loss, mae = model.evaluate([X_climatic_test, X_indices_test], y_test, verbose=1)
print(f"Test Loss: {loss}, Test MAE: {mae}")

# Prediction
predictions = model.predict([X_climatic_test, X_indices_test])

# Shape of prediction
print("Shape of predictions", predictions.shape)  # Have to be (n_samples, 23, 23, 1)
print(y_test.shape)
print(predictions.shape)

# R2 score
r2 = r2_score(y_test.flatten(), predictions.flatten())
print(f"R2 Score: {r2}")

y_test_denorm = fm.denormalize(y_test, yr)

predictions_denorm = fm.denormalize(predictions, yr)

#  'y_test' and 'predictions' are arrays NumPy or tensors
# Making the coords of time, lat and lon
time = range(y_test_denorm.shape[0])  # time index
lat = range(y_test_denorm.shape[1])   # latitude index
lon = range(y_test_denorm.shape[2])   # longitude index

# To xarray.DataArray
y_test_da = xr.DataArray(
    y_test_denorm.squeeze(),  # Removing channel dim if necessary
    dims=["time", "lat", "lon"],
    coords={"time": time, "lat": lat, "lon": lon}
)

predictions_da = xr.DataArray(
    predictions_denorm.squeeze(),  # Removing channel dim if necessary
    dims=["time", "lat", "lon"],
    coords={"time": time, "lat": lat, "lon": lon}
)

# Fixing the shape of 'y_test' and 'predictions' to (time, lat, lon)
lat_correct = ds_cnn['lat']
lon_correct = ds_cnn['lon']
y_test_da = y_test_da.assign_coords(lat=lat_correct, lon=lon_correct)
predictions_da = predictions_da.assign_coords(lat=lat_correct, lon=lon_correct)

# Testing the difference between the mean of y_test_da and predictions_da

print(predictions_da)

dif_mean = y_test_da.mean(dim="time") - predictions_da.mean(dim="time")

dif_percent_mean = ((y_test_da.mean(dim="time") - predictions_da.mean(dim="time")) / np.abs(y_test_da.mean(dim="time"))) * 100

# Plot the results
#dif_percent_mean.plot(vmin = -20, vmax = 20, cmap = "coolwarm")

#plt.show()

fa.plot_maps(y_test_da.mean(dim="time"), predictions_da.mean(dim="time"), dif_percent_mean)