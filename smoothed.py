import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix

filename = "EHDD_old.csv"
df = pd.read_csv(filename)
df["datetime"] = pd.to_datetime(df[["year", "month", "day", "hour", "minute", "second"]])
timespan_years = (df.datetime.max() - df.datetime.min()) / np.timedelta64(1, "Y")

# JUST FOR THE TIME BEING
# CLIP TO EVENTS LARGER THAN 5 FOR SPEED
df = df[df.magnitude>5.0]

# Bins from Helmstetter et al. 2006
grid_spacing = 0.05
lon_edges = np.arange(360.0 - 121.55, 360.0 - 114.45 + grid_spacing, grid_spacing)
lat_edges = np.arange(32.45, 36.65 + grid_spacing, grid_spacing)
H, _, _ = np.histogram2d(df.longitude.values, df.latitude.values, bins=(lon_edges, lat_edges))
H = H.T
H = H / timespan_years
H = np.log10(H)
H[np.isinf(H)] = np.nan

plt.close("all")
plt.figure()
plt.imshow(H, interpolation="nearest", origin="lower", extent=[lon_edges[0], lon_edges[-1], lat_edges[0], lat_edges[-1]])
plt.show(block=False)
plt.xlabel("longitude")
plt.ylabel("latitude")
plt.colorbar(label="earthquakes per year");


# Helmstetter et al. 2006 calcuations
# Do we have to loop over all earthquake pairs?
# Do we do distance calculations with grid cell centroids or actual earthquake locations?
N = len(df) # number of earthquakes (H2006, eq 1)
d0 = grid_spacing # differential degrees for (H2006, eq 3)