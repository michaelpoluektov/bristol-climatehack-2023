import datetime
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from astral import Observer
from astral.sun import sun
import numpy as np
import pandas as pd
import torch
import xarray as xr
from ocf_blosc2 import Blosc2
from tqdm import tqdm

from utils import const_prep, time_prep


HRV_IN_PATH = Path("/root/dataset/google/new/")


# location for sunrise calculations
sunrise_location = Observer(latitude=52.48, longitude=1.77)

# location for sunset calculations
sunset_location = Observer(latitude=54.36, longitude=-8.68)


def get_sunrise(date_np):
    # convert numpy datetime64 to datetime
    date = pd.to_datetime(date_np).to_pydatetime()
    date = date.replace(tzinfo=datetime.timezone.utc)  # make utc

    # calculate sunrise time
    s = sun(observer=sunrise_location, date=date)
    sunrise_time = s["sunrise"]

    # convert sunrise time back to numpy.datetime64
    sunrise_np = np.datetime64(sunrise_time)
    return sunrise_np - np.timedelta64(10, "m")


def get_sunset(date_np):
    # convert numpy datetime64 to datetime
    date = pd.to_datetime(date_np).to_pydatetime()
    date = date.replace(tzinfo=datetime.timezone.utc)  # make utc

    # calculate sunset time
    s = sun(observer=sunset_location, date=date)
    sunset_time = s["sunset"]

    # convert sunset time back to numpy.datetime64
    sunset_np = np.datetime64(sunset_time)
    return sunset_np + np.timedelta64(10, "m")


def split_by_day(datetime_array: np.ndarray):
    """Splits a numpy array of datetimes by and returns split indicies."""

    dates = datetime_array.astype("datetime64[D]")
    unique_days, indices = np.unique(dates, return_index=True)
    sorted_indices = np.sort(indices)

    return unique_days, sorted_indices[1:]


def get_pv_meta():
    file_name = Path("dataset", "pv", "metadata.csv")
    return pd.read_csv(file_name).set_index("ss_id")


class PVPrep:
    """Preps PV data for that month."""

    def __init__(self, year, month):
        file_name = f"dataset/pv/{year}/{month}.parquet"
        self.pv_data = (
            pd.read_parquet(file_name)
            .drop("generation_wh", axis=1)
            .reset_index()
            .assign(timestamp=lambda x: x["timestamp"].dt.tz_localize(None))
        )
        self.pv_meta_data = get_pv_meta()

    def split_days(self):
        day_conv = defaultdict(lambda: {"site2const": {}, "site2f": {}})
        for site, site_df in self.pv_data.groupby("ss_id"):
            site_meta = (
                self.pv_meta_data.loc[site]
                .to_numpy()[[0, 1, 3, 4, 5]]
                .astype(np.double)
            )
            site_times = site_df["timestamp"].to_numpy()
            site_power = site_df["power"].to_numpy()
            time_features = np.array([time_prep(t, site_meta) for t in site_times])

            days, day_indices = split_by_day(site_times)
            split_times = np.split(site_times, day_indices)
            split_power = np.split(site_power, day_indices)
            split_time_features = np.split(time_features, day_indices)

            for day, t, p, tf in zip(
                days, split_times, split_power, split_time_features
            ):
                day_conv[day.item().day]["site2f"][site] = {
                    "times": t.copy(),
                    "power": torch.tensor(p, dtype=torch.bfloat16).clone(),
                    "time_features": torch.tensor(tf, dtype=torch.bfloat16).clone(),
                }
                day_conv[day.item().day]["site2const"][site] = torch.tensor(
                    const_prep(site_meta, t[0]), dtype=torch.bfloat16
                ).clone()

        # keep a list of all times for that day
        all_times = self.pv_data["timestamp"].unique().to_numpy()
        days, day_indices = split_by_day(all_times)
        split_all_times = np.split(all_times, day_indices)
        for day, ts in zip(days, split_all_times):
            day_conv[day.item().day]["all_times"] = ts

        return day_conv


class HRVPrep:
    """Preps HRV data for that month."""

    def __init__(self, year, month):
        first_day = datetime.datetime(year, month, 1)
        next_month = month + 1 if month < 12 else 1
        next_year = year if month < 12 else year + 1
        last_day = datetime.datetime(next_year, next_month, 1) - datetime.timedelta(
            minutes=5
        )

        first_day_str = first_day.strftime("%Y-%m-%d 00:00")
        last_day_str = last_day.strftime("%Y-%m-%d 23:55")

        hrv = xr.open_dataset(
            Path(HRV_IN_PATH, f"subset_data_{year}.zarr"),
            engine="zarr",
            chunks="auto",
        )
        hrv_data = (
            hrv["data"]
            .sel(
                time=slice(
                    first_day_str,
                    last_day_str,
                )
            )
            .isel(
                x_geostationary=slice(46, 46 + 684),
                y_geostationary=slice(53, 53 + 592),
            )
        )
        self.data = torch.tensor(hrv_data.to_numpy(), dtype=torch.bfloat16)
        self.times = (
            hrv["time"]
            .sel(
                time=slice(
                    first_day_str,
                    last_day_str,
                )
            )
            .to_numpy()
        )

        self.prep()

    def prep(self):
        # check times are within sunset
        good = [get_sunrise(t) < t < get_sunset(t) for t in self.times]
        self.data = self.data[good]
        self.times = self.times[good]

        # remove nans and normalise data
        good = [not torch.isnan(row).any() for row in self.data]
        self.data = self.data[good]
        self.times = self.times[good]
        self.data = 2 * self.data - 1

    def split_days(self):
        days, split_indices = split_by_day(self.times)
        times = np.split(self.times, split_indices)
        data = np.split(self.data, split_indices)
        day_conv = {}
        for day, ts, ds in zip(days, times, data):
            day_conv[day.item().day] = {
                "data": ds.squeeze().clone(),
                "times": ts.copy(),
            }

        return day_conv


def prep_month(ym):
    """Preps all data for this month."""

    year, month = ym
    hrv = HRVPrep(year, month)
    pv = PVPrep(year, month)

    print("Splitting HRV")
    hrv_days = hrv.split_days()
    print("Splitting PV")
    pv_days = pv.split_days()

    # create parent folder
    out_folder = Path("dataset/prepped", str(year))
    out_folder.mkdir(exist_ok=True, parents=True)

    # create data files
    days = set(hrv_days.keys()).intersection(set(pv_days.keys()))
    print("we have", len(days))
    for day in days:
        out_file = Path(out_folder, f"m{month:02}_d{day:02}.pt_data")
        print(f"Saving {out_file} ...")
        torch.save({"hrv": hrv_days[day], "pv": pv_days[day]}, out_file)


def main():
    # prep multiple months at once
    years_months = [(year, month) for year in [2020, 2021] for month in range(1, 13)]
    with ProcessPoolExecutor(max_workers=24) as executor:
        list(tqdm(executor.map(prep_month, years_months), total=len(years_months)))


if __name__ == "__main__":
    main()
