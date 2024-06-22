from datetime import datetime

import numpy as np


def npdate2date(dt: np.datetime64):
    return dt.astype("datetime64[ms]").astype(datetime)


def calc_angles(
    latitude: float, longitude: float, dt: np.datetime64
) -> tuple[float, float]:
    """Based on https://github.com/mperezcorrales/SunPositionCalculator.

    Args:
        latitude (float): latitude in degrees
        longitude (float): longitude in degrees
        dt (datetime): time of sun position

    Returns:
        (float, float): elevation, azimuth of sun position in degrees
    """
    date = npdate2date(dt).timetuple()
    hour = date[3]
    minute = date[4]
    # Check your timezone to add the offset
    hour_minute = hour + minute / 60
    day_of_year = date[7]
    g = (360 / 365.25) * (day_of_year + hour_minute / 24)
    g_radians = np.deg2rad(g)
    declination = (
        0.396372
        - 22.91327 * np.cos(g_radians)
        + 4.02543 * np.sin(g_radians)
        - 0.387205 * np.cos(2 * g_radians)
        + 0.051967 * np.sin(2 * g_radians)
        - 0.154527 * np.cos(3 * g_radians)
        + 0.084798 * np.sin(3 * g_radians)
    )
    time_correction = (
        0.004297
        + 0.107029 * np.cos(g_radians)
        - 1.837877 * np.sin(g_radians)
        - 0.837378 * np.cos(2 * g_radians)
        - 2.340475 * np.sin(2 * g_radians)
    )
    SHA = (hour_minute - 12) * 15 + longitude + time_correction
    lat_radians = np.deg2rad(latitude)
    d_radians = np.deg2rad(declination)
    SHA_radians = np.deg2rad(SHA)
    SZA_radians = np.arccos(
        np.sin(lat_radians) * np.sin(d_radians)
        + np.cos(lat_radians) * np.cos(d_radians) * np.cos(SHA_radians)
    )
    SZA = np.rad2deg(SZA_radians)
    SEA = 90 - SZA

    if SEA < 0:
        return 0, 0

    cos_AZ = (np.sin(d_radians) - np.sin(lat_radians) * np.cos(SZA_radians)) / (
        np.cos(lat_radians) * np.sin(SZA_radians)
    )
    AZ_rad = np.arccos(np.clip(cos_AZ, -1, 1))  # need due to precision errors
    if hour_minute > 12:
        AZ = 360 - np.rad2deg(AZ_rad)
    else:
        AZ = np.rad2deg(AZ_rad)
    return SEA, AZ


def calc_air_mass(elevation, pickering=True):
    elevation = elevation % 360
    if pickering:
        return -1 / np.sin(np.deg2rad(elevation + 244 / (165 + 47 * elevation**1.1)))
    else:
        zenith = (90 - elevation) % 360
        return 1 / (
            np.cos(np.deg2rad(zenith)) + 0.50572 * (96.07995 - zenith) ** -1.6364
        )


def calc_vector(elevation, azimuth):
    # no clue how the vector coords relate to the real world
    theta = np.deg2rad(azimuth)
    phi = np.deg2rad(elevation)
    return np.array(
        [
            np.cos(phi) * np.sin(theta),
            np.cos(phi) * np.cos(theta),
            np.sin(phi),
        ]
    )


def cosine_similarity(v1, v2):
    # v1 and v2 should already be normed from calc_vector
    norm_v1 = v1 / np.linalg.norm(v1)
    norm_v2 = v2 / np.linalg.norm(v2)

    return np.dot(norm_v1, norm_v2)


def const_prep(pv_meta, time):
    _, _, orien, tilt, _ = pv_meta
    yday = npdate2date(time).timetuple().tm_yday
    day_angle = 360 * yday / 365

    angles = np.array([np.deg2rad(t) for t in [day_angle, orien, tilt]])
    return np.concatenate((np.sin(angles), np.cos(angles)))


def time_prep(time, pv_meta):
    lat, long, orien, tilt, _ = pv_meta
    sun_elev, sun_az = calc_angles(lat, long, time)
    dt: datetime = npdate2date(time)
    minute_angle = 360 * (dt.hour + dt.minute / 60) / 24
    v1 = calc_vector(sun_elev, sun_az)
    v2 = calc_vector(tilt, orien)

    angles = np.array([np.deg2rad(t) for t in [sun_elev, sun_az, minute_angle]])
    sim = cosine_similarity(v1, v2)
    air_mass = 1 / calc_air_mass(sun_elev)
    return np.concatenate(
        (
            np.sin(angles),
            np.cos(angles),
            np.array((sim, air_mass)),
        )
    )
