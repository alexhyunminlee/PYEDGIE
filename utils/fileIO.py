import csv
import os
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd


def csv_to_dict(file_path: str) -> list[dict]:
    """
    Reads a CSV file and returns a list of dictionaries
     where each dictionary represents a non-empty row.

    Parameters:
    -file_path: Path to the CSV file

    Returns:
    - List of dictionaries with column titles as keys

    Authors: Alex Lee (alexlee5124@gmail.com)
    Date: 12/13/2024
    """
    result = []

    try:
        with open(file_path, encoding="utf-8-sig") as csv_file:
            reader = csv.DictReader(csv_file)

            for row in reader:
                # Only add rows where all values are not empty
                if any(row.values()):
                    result.append({key: value for key, value in row.items() if value})

    except FileNotFoundError:
        print(f"Error: The file at {file_path} was not found.")
        return []
    except Exception as e:
        print(f"An error occurred: {e}")
        return []
    else:
        return result


def save_dict_to_csv(data_dict: dict, file_name: str) -> None:
    """
    Export a dictionary to a CSV file where each key is a column header
     and the values are rows of data.

    :param data_dict: The dictionary to export (keys are column headers,
     values are lists of column data).
    :param output_file: The path to the output CSV file.
    """
    with open(file_name, mode="w", newline="") as csvfile:
        writer = csv.writer(csvfile)

        # Write headers
        headers = list(data_dict.keys())
        writer.writerow(headers)

        # Determine the maximum number of rows (excluding "Coordinates")
        max_rows = max(len(values) for key, values in data_dict.items() if key != "Coordinates")

        # Write data rows
        for i in range(max_rows):
            row = []
            for key in headers:
                if key == "coordinates_lat_long":
                    # Write all coordinates in one cell
                    if i == 0:  # Only write on the first row
                        row.append("".join(map(str, data_dict[key])))
                    else:
                        row.append("")
                else:
                    # Write the corresponding value or
                    #  an empty string if out of range
                    row.append(data_dict[key][i] if i < len(data_dict[key]) else "")
            writer.writerow(row)


def _process_weather_data_types(weatherDF: Any) -> Any:
    """Process and convert weather data types."""
    dataTypes = list(weatherDF.columns)
    for dataType in dataTypes:
        # Convert to float
        if dataType == "temperature (degC)":
            weatherDF[dataType] = [float(temperature_C) for temperature_C in weatherDF[dataType]]
        elif dataType == "wind_speed (m/s)" or dataType == "relative_humidity (0-1)":
            weatherDF[dataType] = [float(windSpeed_M_S) for windSpeed_M_S in weatherDF[dataType]]
        # Convert from W to kW
        elif dataType == "surface_solar_radiation (W/m^2)":
            weatherDF["surface_solar_radiation_kW_per_m2"] = [
                float(suface_solar_W_m2) / 1000
                for suface_solar_W_m2 in weatherDF.pop("surface_solar_radiation (W/m^2)")
            ]
        elif dataType == "direct_normal_solar_radiation (W/m^2)":
            weatherDF["direct_normal_solar_kW_per_m2"] = [
                float(direct_normal_solar_W_m2) / 1000
                for direct_normal_solar_W_m2 in weatherDF.pop("direct_normal_solar_radiation (W/m^2)")
            ]
        elif dataType == "surface_diffuse_solar_radiation (W/m^2)":
            weatherDF["surface_diffuse_solar_kW_per_m2"] = [
                float(surface_diffuse_solar_W_m2) / 1000
                for surface_diffuse_solar_W_m2 in weatherDF.pop("surface_diffuse_solar_radiation (W/m^2)")
            ]
    return weatherDF


def _validate_datetime_range(weatherDF: Any, start: datetime | None, end: datetime | None) -> None:
    """Validate start and end datetime ranges."""
    if start is None or end is None:
        raise ValueError()
    if start >= weatherDF.index[-1]:
        raise ValueError()
    if start < weatherDF.index[0]:
        raise ValueError()
    if end <= weatherDF.index[0]:
        raise ValueError()
    if end > weatherDF.index[-1]:
        raise ValueError()
    if start >= end:
        raise ValueError()


def _resample_weather_data(weatherDF: Any, start: datetime | None, end: datetime | None, timeWindow: timedelta) -> Any:
    """Resample weather data to specified time window."""
    # Resample the weather data to start at start date and end at end date
    if start not in weatherDF.index:
        weatherDF.loc[start] = np.nan
        weatherDF = weatherDF.sort_index()
        weatherDF = weatherDF.interpolate(method="time")
    weatherDF = weatherDF[weatherDF.index >= start]

    if end not in weatherDF.index:
        weatherDF.loc[end] = np.nan
        weatherDF = weatherDF.sort_index()
        weatherDF = weatherDF.interpolate(method="time")
    weatherDF = weatherDF[weatherDF.index <= end]

    # Resample the weather data to the new timewindow
    newTimesteps = pd.date_range(start=weatherDF.index[0], end=weatherDF.index[-1], freq=timeWindow)
    weatherDF_lastRow = weatherDF.tail(1).copy()
    additionalRows = newTimesteps.difference(weatherDF.index)
    additionalRows = pd.DataFrame(index=additionalRows, columns=weatherDF.columns)
    weatherDF = pd.concat([weatherDF, additionalRows])
    weatherDF.sort_index(inplace=True)
    weatherDF.interpolate(method="linear", inplace=True)
    weatherDF = weatherDF[weatherDF.index.isin(newTimesteps)]
    if weatherDF.index[-1] != weatherDF_lastRow.index[-1]:
        weatherDF = pd.concat([weatherDF, weatherDF_lastRow])

    return weatherDF


def importWeather(
    file_path: str, start: datetime | None = None, end: datetime | None = None, timeWindow: timedelta | None = None
) -> Any:
    """
    Reads the weather file. The weather file is to be downloaded from Oikolab
    as a default. This decides the column header names.

    Parameters:
    - file_path: Path to the CSV file

    Returns:
    - A dictionary containing weather information. Each key represents the
    type of data and the corresponding value is an array of data

    Authors: Priyadarshan (priyada@purdue.edu),
             Alex Lee (alexlee5124@gmail.com)
    Date: 12/13/2024
    """
    if timeWindow is None:
        timeWindow = timedelta(hours=1)

    # Check that the file exists
    if not os.path.exists(file_path):
        raise ValueError()

    # Read the CSV file into a DataFrame
    weatherDF = pd.read_csv(
        file_path,
        index_col=0,  # Use the first column as the index
        parse_dates=True,  # Parse the index as datetime objects
    )

    # Timezone correction
    try:
        timeZoneCorrection = int(weatherDF.pop("utc_offset (hrs)")[0])
    except Exception:
        print("ERROR [importWeather()]: Error occurred in finding the timezone offset")
    weatherDF.index = weatherDF.index + pd.Timedelta(hours=timeZoneCorrection)

    # Convert to appropriate data types
    weatherDF.drop("coordinates (lat,lon)", axis=1, inplace=True)
    weatherDF.drop("model (name)", axis=1, inplace=True)
    weatherDF = _process_weather_data_types(weatherDF)

    # Define start/end datetimes
    if start is None:
        start = weatherDF.index[0]
    if end is None:
        end = weatherDF.index[-1]

    _validate_datetime_range(weatherDF, start, end)
    weatherDF = _resample_weather_data(weatherDF, start, end, timeWindow)

    # Export to CSV
    weatherDF.to_csv("data.csv", index=True)

    return weatherDF
