from datetime import datetime
import json
import glob
import random

from tqdm import tqdm
import numpy as np
from torch.utils.data import IterableDataset, DataLoader, get_worker_info
import torch


def get_site_pos() -> dict[str, dict[int, tuple[int, int]]]:
    # site_locations: a dictionary of PV ids to tuples representing the coordinates
    # of that PV in the HRV array
    with open("dataset/indices.json") as f:
        return {
            data_source: {
                int(site): (int(location[0]), int(location[1]))
                for site, location in locations.items()
            }
            for data_source, locations in json.load(f).items()
        }


def infinite_iter(thing):
    while True:
        yield from thing


def get_int(filename):
    """Get int from (prepped) day dataset file."""
    fs = filename.split("_")
    a = int(fs[0][-2:])
    return a * 100 + int(fs[1][1:3])


class DayDataset(IterableDataset):
    def __init__(self, step, is_train=True, num_drop=2):
        self.site_pos = get_site_pos()
        self.data_files = glob.glob("./dataset/prepped/20*/*")
        print(f"Training with {len(self.data_files)} files")
        self.is_train = is_train
        self.num_drop = num_drop

    def calc_valid_start_times(self, times_pv, times_hrv):
        """Combine PV and HRV valid start times."""
        times = np.intersect1d(times_pv[self.num_drop : -self.num_drop], times_hrv)
        for idx in np.where(np.diff(times) != np.timedelta64(300, "s"))[0]:
            times[idx - 60 + 1 : idx + 1] = 0
        times = times[times != np.datetime64(0, "ns")]
        return times

    def __iter__(self):
        info = get_worker_info()

        if info is not None:
            self.data_files = self.data_files[info.id :: info.num_workers]
        random.shuffle(self.data_files)

        for day_file in self.data_files:
            data = torch.load(day_file)
            # shuffle sites
            valid_sites = list(data["pv"]["site2const"].keys())
            random.shuffle(valid_sites)
            site_iter = infinite_iter(valid_sites)

            hrv_t = data["hrv"]["times"]

            start_times = self.calc_valid_start_times(data["pv"]["all_times"], hrv_t)
            np.random.shuffle(start_times)

            for start_time in start_times:
                small_end_time = start_time + np.timedelta64("1", "h")
                end_time = start_time + np.timedelta64("5", "h")
                for _ in range(40):
                    site = next(site_iter)

                    # try loading site at start_time
                    # if error, try next site
                    try:
                        # pv
                        site_f = data["pv"]["site2f"][site]
                        site_cst = data["pv"]["site2const"][site]
                        s_t_idx = site_f["times"].searchsorted(start_time)
                        site_time_index_end = site_f["times"].searchsorted(end_time)
                        assert site_time_index_end - s_t_idx == 60, "PV index mismatch"
                        s_pow = site_f["power"][s_t_idx : s_t_idx + 60]
                        assert s_pow.shape == (60,)
                        s_t_ftrs = site_f["time_features"][s_t_idx : s_t_idx + 60]
                        assert s_t_ftrs.shape == (60, 8)
                        s_cst_ftrs = site_cst
                        assert s_cst_ftrs.shape == (6,)

                        # hrv
                        h_x, h_y = self.site_pos["hrv"][site]
                        hrv_index = hrv_t.searchsorted(start_time)
                        hrv_index_end = hrv_t.searchsorted(small_end_time)
                        assert hrv_index_end - hrv_index == 12, "HRV index mismatch"
                        hrv = data["hrv"]["data"][
                            hrv_index : hrv_index + 12,
                            h_y - 64 : h_y + 64,
                            h_x - 64 : h_x + 64,
                        ]
                        assert hrv.shape == (12, 128, 128), "HRV shape mismatch"

                    except KeyError:
                        # print(f"Unknown site coords {site}")
                        continue
                    except AssertionError:
                        # print(f'bad {site} @ {start_time} with {e}')
                        continue

                    # make sure data is correct shape
                    hrv = hrv.unsqueeze(0)
                    s_pow = s_pow.unsqueeze(-1)
                    s_cst_ftrs = s_cst_ftrs.unsqueeze(0)
                    yield hrv, s_pow[:12], s_t_ftrs, s_cst_ftrs, s_pow[12:]


def benchmark():
    ds = DayDataset(5, True)
    data_loader = DataLoader(ds, batch_size=16, num_workers=3, prefetch_factor=2)

    t = datetime.now()
    for batch in tqdm(data_loader):
        pass
    print(datetime.now() - t)


if __name__ == "__main__":
    benchmark()
