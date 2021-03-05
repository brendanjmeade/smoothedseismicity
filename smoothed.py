import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.close("all")


def quick_plot(field, colorbar_string, lon_edges, lat_edges):
    plt.figure()
    plt.imshow(
        field,
        interpolation="nearest",
        origin="lower",
        extent=[lon_edges[0], lon_edges[-1], lat_edges[0], lat_edges[-1]],
    )
    plt.show(block=False)
    plt.xlabel("longitude")
    plt.ylabel("latitude")
    plt.colorbar(label=colorbar_string)


def calc_kernel_influence(src_lon, src_lat, target_lon, target_lat, d):
    """ Calculate the influence of an individual earthquake at a target location """
    # Cd = 1
    Cd = 1 / np.sqrt(d)  # not sure this is right
    kernel_exponent = 1.5
    delta_lon = src_lon - target_lon
    delta_lat = src_lat - target_lat
    distance = np.sqrt(np.power(delta_lon, 2) + np.power(delta_lat, 2))
    Kr = Cd / np.power(distance + d, kernel_exponent)
    return Kr


def read_data():
    filename = "EHDD_old.csv"
    df = pd.read_csv(filename)
    df["datetime"] = pd.to_datetime(
        df[["year", "month", "day", "hour", "minute", "second"]]
    )
    return df


def grid_seismicity(df):
    grid_spacing = 0.05
    lon_edges = np.arange(360.0 - 121.55, 360.0 - 114.45 + grid_spacing, grid_spacing)
    lat_edges = np.arange(32.45, 36.65 + grid_spacing, grid_spacing)
    H, _, _ = np.histogram2d(
        df.longitude.values, df.latitude.values, bins=(lon_edges, lat_edges)
    )
    H = H.T
    timespan_years = (df.datetime.max() - df.datetime.min()) / np.timedelta64(1, "Y")
    H = H / timespan_years
    H = np.log10(H)
    H[np.isinf(H)] = np.nan
    quick_plot(H, "earthquakes per year", lon_edges, lat_edges)
    return grid_spacing, lon_edges, lat_edges


def grid_centers(grid_spacing, lon_edges, lat_edges):
    lon_centers = lon_edges[1:-2] + grid_spacing  # longitudes of bin centers
    lat_centers = lat_edges[1:-2] + grid_spacing  # latitudes of bin centers
    lon_centers_mat, lat_centers_mat = np.meshgrid(
        lon_centers, lat_centers
    )  # Plaid gridding of bin centers
    return lon_centers, lat_centers, lon_centers_mat, lat_centers_mat


def example_kernel(
    d0, lon_centers_mat, lat_centers_mat, lon_centers, lat_centers, lon_edges, lat_edges
):
    Kr = calc_kernel_influence(
        242, 34, lon_centers_mat.flatten(), lat_centers_mat.flatten(), d0
    )
    Kr = Kr.reshape((len(lat_centers), len(lon_centers)))
    quick_plot(Kr, "Example kernel", lon_edges, lat_edges)


def calc_mu_star(df, d, lon_centers_mat, lat_centers_mat, lon_centers, lat_centers):
    N = len(df)  # number of earthquakes
    mu_star = np.zeros(lon_centers_mat.flatten().shape)
    for i in range(0, mu_star.size):
        mu_star[i] = (
            sum(
                calc_kernel_influence(
                    lon_centers_mat.flatten()[i],
                    lat_centers_mat.flatten()[i],
                    df.longitude.values,
                    df.latitude.values,
                    d,
                )
            )
            / N
        )
        print(i)
    mu_star = mu_star.reshape((len(lat_centers), len(lon_centers)))
    return mu_star


def main():
    df = read_data()
    df = df[df.magnitude > 5.0]  # CLIP TO EVENTS LARGER THAN 5 FOR SPEED
    grid_spacing, lon_edges, lat_edges = grid_seismicity(df)
    lon_centers, lat_centers, lon_centers_mat, lat_centers_mat = grid_centers(
        grid_spacing, lon_edges, lat_edges
    )

    # Calculate mu_star
    d0 = 0.05  # differential degrees
    mu_star = calc_mu_star(
        df, d0, lon_centers_mat, lat_centers_mat, lon_centers, lat_centers
    )
    quick_plot(mu_star, "mu_star", lon_edges, lat_edges)


if __name__ == "__main__":
    main()
