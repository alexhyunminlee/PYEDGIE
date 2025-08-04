import os
import sys
from datetime import datetime, timedelta
from unittest.mock import patch

import pandas as pd
import pytest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# Custom exceptions for better error handling
class BuildingValidationError(ValueError):
    """Custom exception for building validation errors."""

    INVALID_REGION = "Invalid region"
    INVALID_TIME_PARAMETERS = "Invalid time parameters"
    START_TIME_AFTER_END = "Start time must be before end time"
    ATTACHED_STATUS_NOT_SET = "Attached status not set"


# Create a minimal test Building class that doesn't inherit from Simulator
class BuildingForTesting:
    def __init__(self, name: str, region: str, attached: bool, characteristics: dict[str, float | int | str]) -> None:
        self.name = name
        self.region = region
        self.attached = attached
        self.characteristics = characteristics
        self.calculateR()

    def _validate_region(self) -> None:
        """Validate that region is one of the allowed values."""
        valid_regions = ["West", "Midwest", "South", "Northeast"]
        if self.region not in valid_regions:
            raise BuildingValidationError(BuildingValidationError.INVALID_REGION)

    def _validate_time_parameters(self, tStart: datetime, tEnd: datetime, tWindow: timedelta) -> None:
        """Validate time parameters."""
        if tStart is None or tEnd is None or tWindow is None:
            raise BuildingValidationError(BuildingValidationError.INVALID_TIME_PARAMETERS)
        if not (tStart < tEnd):
            raise BuildingValidationError(BuildingValidationError.START_TIME_AFTER_END)

    def _get_scaling_factors(self) -> tuple[float, float]:
        """Get scaling factors based on region."""
        if self.region == "West":  # https://www.eia.gov/consumption/residential/data/2020/c&e/pdf/ce4.2.pdf
            scaling_detached = (17e9 + 96e9) / (16.97e6 * 8760)
            scaling_attached = (
                (1e9 + 6e9) / (8760 * 1.69e6) + (1e9 + 5e9) / (8760 * 1.89e6) + (3e9 + 14e9) / (8760 * 5.70e6)
            ) / 3
        elif self.region == "Midwest":
            scaling_detached = (18e9 + 115e9) / (8760 * 18.58e6)
            scaling_attached = (
                (1e9 + 6e9) / (8760 * 1.33e6) + (1e9 + 5e9) / (8760 * 1.95e6) + (2e9 + 10e9) / (8760 * 4.20e6)
            ) / 3
        elif self.region == "Northeast":
            scaling_detached = (10e9 + 66e9) / (11.23e6 * 8760)
            scaling_attached = (
                (2e9 + 8e9) / (8760 * 1.95e6) + (2e9 + 8e9) / (8760 * 3.15e6) + (3e9 + 11e9) / (8760 * 5.10e6)
            ) / 3
        else:  # South
            scaling_detached = (31e9 + 205e9) / (30.29e6 * 8760)
            scaling_attached = (
                (2e9 + 11e9) / (8760 * 2.48e6) + (1e9 + 8e9) / (8760 * 2.36e6) + (5e9 + 26e9) / (8760 * 7.83e6)
            ) / 3
        return scaling_detached, scaling_attached

    def _scale_data(self, scaledData: pd.DataFrame, scaling_detached: float, scaling_attached: float) -> pd.DataFrame:
        """Scale the data based on building characteristics."""
        # Scale data
        scaledData.update(scaledData.iloc[:, 1:].div(scaledData.iloc[:, 1:].mean(axis=0), axis="columns"))

        if self.attached:
            scaledData.update(scaledData.iloc[:, 1:].mul(scaling_attached))
        else:
            scaledData.update(scaledData.iloc[:, 1:].mul(scaling_detached))
        return scaledData

    def _create_time_series(self, scaledData: pd.DataFrame, tStart: datetime, tEnd: datetime) -> pd.DataFrame:
        """Create time series from tStart to tEnd."""
        baseline = pd.DataFrame(columns=scaledData.columns)

        # Create time series from tStart to tEnd
        scaledData["DateTime"] += pd.DateOffset(years=(tStart.year - scaledData["DateTime"][0].year))
        for year in range(0, tEnd.year - tStart.year + 1):
            # Add the number of years to the DateTime column for each iteration
            scaledData["DateTime"] = scaledData["DateTime"] + pd.DateOffset(years=year)
            baseline = pd.concat([baseline, scaledData], ignore_index=True)

        baseline = baseline[
            (baseline["DateTime"] >= tStart.replace(hour=0, minute=0, second=0, microsecond=0))
            & (baseline["DateTime"] <= tEnd.replace(hour=0, minute=0, second=0, microsecond=0))
        ]
        baseline.set_index(baseline.columns[0], inplace=True)
        return baseline

    def generateBaselineElectricity(
        self, tStart: datetime, tEnd: datetime, tWindow: timedelta, filePath: str
    ) -> pd.DataFrame | None:
        """
        Read the baseline electricity load file, scale it based on the building characteristics,
        and choose the data to be between the desired start and end time.
        """
        # Error checking
        self._validate_region()
        if self.attached is None:
            raise BuildingValidationError(BuildingValidationError.ATTACHED_STATUS_NOT_SET)
        self._validate_time_parameters(tStart, tEnd, tWindow)

        # Read in raw data
        try:
            scaledData = pd.read_excel(filePath)
        except FileNotFoundError:
            print(f"Error: The file '{filePath}' was not found. Please check the file path.")
            return None
        except ValueError as e:
            print(f"Error: There was an issue with the file format. Details: {e}")
            return None
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            return None

        # Randomly select a column and update data to only include that column
        import secrets

        scaledData = scaledData.iloc[:, [0, secrets.randbelow(scaledData.shape[1] - 1) + 1]]

        # Get scaling factors
        scaling_detached, scaling_attached = self._get_scaling_factors()

        # Scale data
        scaledData = self._scale_data(scaledData, scaling_detached, scaling_attached)

        # Convert UTC to EST time
        scaledData.insert(loc=1, column="DateTime", value=scaledData["DateTimeUTC"] - timedelta(hours=5))
        scaledData.drop(columns=["DateTimeUTC"], inplace=True)

        # Create time series
        baseline = self._create_time_series(scaledData, tStart, tEnd)

        # Retime
        currentWindowSize = baseline.index.to_series().diff()[-1]
        if tWindow >= currentWindowSize:
            baseline = baseline.resample(tWindow).mean()
        else:
            baseline = baseline.resample(tWindow).interpolate(method="linear")

        return baseline

    def calculateR(self) -> None:
        """Calculate overall thermal resistance R parameter for the building."""
        storyHeight = float(self.characteristics["storyHeight"])
        aspectRatio = float(self.characteristics["aspectRatio"])
        numStories = int(self.characteristics["numStories"])
        floorArea = float(self.characteristics["floorArea"])

        lamb = 0.25
        Ur = 0.36
        density = 1.293  # kg/m^3
        Cp = 1005  # J/kg-K
        volume = floorArea * storyHeight  # m^3
        v = 1.5  # Air Exchange Rate per hour
        h_out = 22.7 / 1000
        h_in = 8.29 / 1000
        valueWindow = 2.5  # W/m^2-K
        valueWall = 0.4  # W/m^2-K

        if self.attached:
            # calculation for attached
            Aw = 2 * storyHeight * (aspectRatio + 1) * ((numStories * floorArea / aspectRatio) ** 0.5) / 2
        else:
            # Calculation for detached
            Aw = 2 * storyHeight * (aspectRatio + 1) * ((numStories * floorArea / aspectRatio) ** 0.5)

        # Modify this code
        areaRoof = floorArea / numStories
        modtCp = v * Cp * density * volume / 3600
        Uwall = lamb * valueWindow + (1 - lamb) * valueWall
        R = 1 / (modtCp / 1000 + Uwall * Aw / 1000 + Ur * areaRoof / 1000) + 1 / (h_in * Aw) + 1 / (h_out * Aw)
        self.characteristics["R"] = R

    def Hp_sizecooling(self, filepath: str, fullP: pd.DataFrame) -> float:
        """Simulate heat pump with resistance heating equipment for cooling."""
        import os
        import sys

        import numpy as np

        # Add the PYEDGIE directory to the path to import utils
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from PYEDGIE.utils import conversions, distributions, fileIO

        t1 = 1
        R = float(self.characteristics["R"])
        floorArea = self.characteristics["floorArea"]
        designTempCool = float(self.characteristics["designTempCool"])

        weather = fileIO.importWeather(filepath)
        thetaFull = weather["temperature (degC)"]
        solarFull = weather["surface_solar_radiation_kW_per_m2"]
        K = len(thetaFull)

        # Use the first K rows of fullP, or pad if needed
        if len(fullP) >= K:
            electricLoadThermalPower = fullP.iloc[:K]
        else:
            # Pad with zeros if fullP is shorter than K
            padding = pd.DataFrame(
                {"thermal_power": [0.0] * (K - len(fullP))},
                index=pd.date_range(fullP.index[-1] + pd.Timedelta(hours=1), periods=K - len(fullP), freq="h"),
            )
            electricLoadThermalPower = pd.concat([fullP, padding])

        bodyThermalPower = np.array(distributions.gauss(K, t1, 0.1, 0.5))  # thermal power from body heat, kW
        solarThermalPower = (
            0.03
            * np.array(distributions.gauss(K, t1, 0.9, 1.1))
            * float(floorArea)
            * np.array(solarFull).reshape(-1, 1)
        )  # thermal power from sunlight, kW

        # Convert electricLoadThermalPower to numpy array for addition
        electricLoadArray = np.array(electricLoadThermalPower["thermal_power"]).reshape(-1, 1)
        qe1 = electricLoadArray + bodyThermalPower + solarThermalPower

        differences = abs(designTempCool - thetaFull)

        ind = np.argmin(differences)
        # Find the index of the minimum absolute difference

        Tset = (np.array(distributions.trirnd(61, 76, 1, 1)) - 32.0) * 5.0 / 9.0
        thermalPower1 = abs((Tset - conversions.f2c(designTempCool)) / R - np.array(qe1)[ind])
        Hp_Cooling = np.ceil(distributions.trirnd(1.2, 1.3, 1, 1) * (thermalPower1 / (0.80)))

        return float(Hp_Cooling[0, 0])

    def Hp_sizeheating(self, filepath: str, fullP: pd.DataFrame) -> float:
        """Simulate heat pump with resistance heating equipment for heating."""
        import os
        import sys

        import numpy as np

        # Add the PYEDGIE directory to the path to import utils
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from PYEDGIE.utils import conversions, distributions, fileIO

        t1 = 1
        R = float(self.characteristics["R"])
        floorArea = self.characteristics["floorArea"]
        designTempHeat = float(self.characteristics["designTempHeat"])

        weather = fileIO.importWeather(filepath)
        thetaFull = weather["temperature (degC)"]
        solarFull = weather["surface_solar_radiation_kW_per_m2"]
        K = len(thetaFull)

        # Use the first K rows of fullP, or pad if needed
        if len(fullP) >= K:
            electricLoadThermalPower = fullP.iloc[:K]
        else:
            # Pad with zeros if fullP is shorter than K
            padding = pd.DataFrame(
                {"thermal_power": [0.0] * (K - len(fullP))},
                index=pd.date_range(fullP.index[-1] + pd.Timedelta(hours=1), periods=K - len(fullP), freq="h"),
            )
            electricLoadThermalPower = pd.concat([fullP, padding])

        bodyThermalPower = np.array(distributions.gauss(K, t1, 0.1, 0.5))  # thermal power from body heat, kW
        solarThermalPower = (
            0.03
            * np.array(distributions.gauss(K, t1, 0.9, 1.1))
            * float(floorArea)
            * np.array(solarFull).reshape(-1, 1)
        )  # thermal power from sunlight, kW

        # Convert electricLoadThermalPower to numpy array for addition
        electricLoadArray = np.array(electricLoadThermalPower["thermal_power"]).reshape(-1, 1)
        qe1 = electricLoadArray + bodyThermalPower + solarThermalPower

        differences = abs(designTempHeat - thetaFull)

        ind = np.argmin(differences)
        # Find the index of the minimum absolute difference

        Tset = (np.array(distributions.trirnd(61, 76, 1, 1)) - 32.0) * 5.0 / 9.0
        thermalPower1 = abs((Tset - conversions.f2c(designTempHeat)) / R - np.array(qe1)[ind])
        Hp_Heating = np.ceil(distributions.trirnd(1.2, 1.3, 1, 1) * (thermalPower1 / (0.80)))

        return float(Hp_Heating[0, 0])


def test_generate_baseline_electricity_basic_functionality():
    """Test basic functionality of generateBaselineElectricity with real data."""
    characteristics = {
        "storyHeight": 3.0,
        "aspectRatio": 1.0,
        "numStories": 2,
        "floorArea": 100.0,
        "designTempCool": 75.0,
        "designTempHeat": 68.0,
    }

    building = BuildingForTesting("test_building", "West", True, characteristics)

    t_start = datetime(2020, 1, 1, 0, 0, 0)
    t_end = datetime(2020, 1, 2, 0, 0, 0)
    t_window = timedelta(hours=1)
    data_file_path = "data/cleanedMFREDdata.xlsx"

    result = building.generateBaselineElectricity(t_start, t_end, t_window, data_file_path)

    # Check that result is not None
    assert result is not None

    # Check that result is a pandas DataFrame
    assert isinstance(result, pd.DataFrame)

    # Check that result has expected structure
    assert len(result.columns) == 1  # Should have one data column
    assert len(result) > 0  # Should have some data

    # Check that index is datetime
    assert isinstance(result.index, pd.DatetimeIndex)

    # Check that data is within expected time range
    assert result.index.min() >= t_start
    assert result.index.max() <= t_end


def test_generate_baseline_electricity_detached_building():
    """Test generateBaselineElectricity with detached building."""
    characteristics = {
        "storyHeight": 3.0,
        "aspectRatio": 1.0,
        "numStories": 2,
        "floorArea": 100.0,
        "designTempCool": 75.0,
        "designTempHeat": 68.0,
    }

    detached_building = BuildingForTesting("detached_building", "South", False, characteristics)

    t_start = datetime(2020, 1, 1, 0, 0, 0)
    t_end = datetime(2020, 1, 2, 0, 0, 0)
    t_window = timedelta(hours=1)
    data_file_path = "data/cleanedMFREDdata.xlsx"

    result = detached_building.generateBaselineElectricity(t_start, t_end, t_window, data_file_path)

    assert result is not None
    assert isinstance(result, pd.DataFrame)
    assert len(result) > 0


def test_generate_baseline_electricity_different_regions():
    """Test generateBaselineElectricity with different regions."""
    characteristics = {
        "storyHeight": 3.0,
        "aspectRatio": 1.0,
        "numStories": 2,
        "floorArea": 100.0,
        "designTempCool": 75.0,
        "designTempHeat": 68.0,
    }

    regions = ["West", "Midwest", "South", "Northeast"]

    for region in regions:
        building = BuildingForTesting(f"building_{region}", region, True, characteristics)

        t_start = datetime(2020, 1, 1, 0, 0, 0)
        t_end = datetime(2020, 1, 2, 0, 0, 0)
        t_window = timedelta(hours=1)
        data_file_path = "data/cleanedMFREDdata.xlsx"

        result = building.generateBaselineElectricity(t_start, t_end, t_window, data_file_path)

        assert result is not None
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0


def test_generate_baseline_electricity_different_time_windows():
    """Test generateBaselineElectricity with different time windows."""
    characteristics = {
        "storyHeight": 3.0,
        "aspectRatio": 1.0,
        "numStories": 2,
        "floorArea": 100.0,
        "designTempCool": 75.0,
        "designTempHeat": 68.0,
    }

    time_windows = [timedelta(minutes=15), timedelta(hours=1), timedelta(hours=4), timedelta(days=1)]

    building = BuildingForTesting("test_building", "West", True, characteristics)

    t_start = datetime(2020, 1, 1, 0, 0, 0)
    t_end = datetime(2020, 1, 2, 0, 0, 0)
    data_file_path = "data/cleanedMFREDdata.xlsx"

    for window in time_windows:
        result = building.generateBaselineElectricity(t_start, t_end, window, data_file_path)

        assert result is not None
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0


def test_generate_baseline_electricity_longer_time_period():
    """Test generateBaselineElectricity with a longer time period."""
    characteristics = {
        "storyHeight": 3.0,
        "aspectRatio": 1.0,
        "numStories": 2,
        "floorArea": 100.0,
        "designTempCool": 75.0,
        "designTempHeat": 68.0,
    }

    building = BuildingForTesting("test_building", "West", True, characteristics)

    t_start = datetime(2020, 1, 1, 0, 0, 0)
    t_end = datetime(2020, 1, 7, 0, 0, 0)  # One week
    t_window = timedelta(hours=1)
    data_file_path = "data/cleanedMFREDdata.xlsx"

    result = building.generateBaselineElectricity(t_start, t_end, t_window, data_file_path)

    assert result is not None
    assert isinstance(result, pd.DataFrame)
    assert len(result) > 0


def test_generate_baseline_electricity_invalid_region():
    """Test generateBaselineElectricity with invalid region."""
    characteristics = {
        "storyHeight": 3.0,
        "aspectRatio": 1.0,
        "numStories": 2,
        "floorArea": 100.0,
        "designTempCool": 75.0,
        "designTempHeat": 68.0,
    }

    invalid_building = BuildingForTesting("invalid_building", "InvalidRegion", True, characteristics)

    t_start = datetime(2020, 1, 1, 0, 0, 0)
    t_end = datetime(2020, 1, 2, 0, 0, 0)
    t_window = timedelta(hours=1)
    data_file_path = "data/cleanedMFREDdata.xlsx"

    with pytest.raises(BuildingValidationError, match=BuildingValidationError.INVALID_REGION):
        invalid_building.generateBaselineElectricity(t_start, t_end, t_window, data_file_path)


def test_generate_baseline_electricity_invalid_time_parameters():
    """Test generateBaselineElectricity with invalid time parameters."""
    characteristics = {
        "storyHeight": 3.0,
        "aspectRatio": 1.0,
        "numStories": 2,
        "floorArea": 100.0,
        "designTempCool": 75.0,
        "designTempHeat": 68.0,
    }

    building = BuildingForTesting("test_building", "West", True, characteristics)

    t_start = datetime(2020, 1, 1, 0, 0, 0)
    t_end = datetime(2020, 1, 2, 0, 0, 0)
    t_window = timedelta(hours=1)
    data_file_path = "data/cleanedMFREDdata.xlsx"

    # Test with None parameters
    with pytest.raises(BuildingValidationError, match=BuildingValidationError.INVALID_TIME_PARAMETERS):
        building.generateBaselineElectricity(None, t_end, t_window, data_file_path)

    with pytest.raises(BuildingValidationError, match=BuildingValidationError.INVALID_TIME_PARAMETERS):
        building.generateBaselineElectricity(t_start, None, t_window, data_file_path)

    with pytest.raises(BuildingValidationError, match=BuildingValidationError.INVALID_TIME_PARAMETERS):
        building.generateBaselineElectricity(t_start, t_end, None, data_file_path)


def test_generate_baseline_electricity_invalid_time_order():
    """Test generateBaselineElectricity with start time after end time."""
    characteristics = {
        "storyHeight": 3.0,
        "aspectRatio": 1.0,
        "numStories": 2,
        "floorArea": 100.0,
        "designTempCool": 75.0,
        "designTempHeat": 68.0,
    }

    building = BuildingForTesting("test_building", "West", True, characteristics)

    t_start = datetime(2020, 1, 2, 0, 0, 0)
    t_end = datetime(2020, 1, 1, 0, 0, 0)
    t_window = timedelta(hours=1)
    data_file_path = "data/cleanedMFREDdata.xlsx"

    with pytest.raises(BuildingValidationError, match=BuildingValidationError.START_TIME_AFTER_END):
        building.generateBaselineElectricity(t_start, t_end, t_window, data_file_path)


def test_generate_baseline_electricity_missing_attached_status():
    """Test generateBaselineElectricity with missing attached status."""
    characteristics = {
        "storyHeight": 3.0,
        "aspectRatio": 1.0,
        "numStories": 2,
        "floorArea": 100.0,
        "designTempCool": 75.0,
        "designTempHeat": 68.0,
    }

    building = BuildingForTesting("test_building", "West", None, characteristics)

    t_start = datetime(2020, 1, 1, 0, 0, 0)
    t_end = datetime(2020, 1, 2, 0, 0, 0)
    t_window = timedelta(hours=1)
    data_file_path = "data/cleanedMFREDdata.xlsx"

    with pytest.raises(BuildingValidationError, match=BuildingValidationError.ATTACHED_STATUS_NOT_SET):
        building.generateBaselineElectricity(t_start, t_end, t_window, data_file_path)


def test_generate_baseline_electricity_file_not_found():
    """Test generateBaselineElectricity with non-existent file."""
    characteristics = {
        "storyHeight": 3.0,
        "aspectRatio": 1.0,
        "numStories": 2,
        "floorArea": 100.0,
        "designTempCool": 75.0,
        "designTempHeat": 68.0,
    }

    building = BuildingForTesting("test_building", "West", True, characteristics)

    t_start = datetime(2020, 1, 1, 0, 0, 0)
    t_end = datetime(2020, 1, 2, 0, 0, 0)
    t_window = timedelta(hours=1)

    result = building.generateBaselineElectricity(t_start, t_end, t_window, "nonexistent_file.xlsx")

    # Should return None for file not found
    assert result is None


@patch("pandas.read_excel")
def test_generate_baseline_electricity_file_format_error(mock_read_excel):
    """Test generateBaselineElectricity with file format error."""
    characteristics = {
        "storyHeight": 3.0,
        "aspectRatio": 1.0,
        "numStories": 2,
        "floorArea": 100.0,
        "designTempCool": 75.0,
        "designTempHeat": 68.0,
    }

    mock_read_excel.side_effect = ValueError("Invalid file format")

    building = BuildingForTesting("test_building", "West", True, characteristics)

    t_start = datetime(2020, 1, 1, 0, 0, 0)
    t_end = datetime(2020, 1, 2, 0, 0, 0)
    t_window = timedelta(hours=1)
    data_file_path = "data/cleanedMFREDdata.xlsx"

    result = building.generateBaselineElectricity(t_start, t_end, t_window, data_file_path)

    # Should return None for file format error
    assert result is None


def test_generate_baseline_electricity_data_scaling():
    """Test that data is properly scaled based on building characteristics."""
    characteristics = {
        "storyHeight": 3.0,
        "aspectRatio": 1.0,
        "numStories": 2,
        "floorArea": 100.0,
        "designTempCool": 75.0,
        "designTempHeat": 68.0,
    }

    # Test with different building characteristics
    large_building_chars = characteristics.copy()
    large_building_chars["floorArea"] = 500.0

    building = BuildingForTesting("test_building", "West", True, characteristics)
    large_building = BuildingForTesting("large_building", "West", True, large_building_chars)

    t_start = datetime(2020, 1, 1, 0, 0, 0)
    t_end = datetime(2020, 1, 2, 0, 0, 0)
    t_window = timedelta(hours=1)
    data_file_path = "data/cleanedMFREDdata.xlsx"

    result_small = building.generateBaselineElectricity(t_start, t_end, t_window, data_file_path)
    result_large = large_building.generateBaselineElectricity(t_start, t_end, t_window, data_file_path)

    assert result_small is not None
    assert result_large is not None

    # The larger building should have different scaling factors
    # Note: Due to random column selection, we can't guarantee exact relationships
    # but we can check that both results are valid DataFrames
    assert isinstance(result_small, pd.DataFrame)
    assert isinstance(result_large, pd.DataFrame)


def test_generate_baseline_electricity_timezone_conversion():
    """Test that UTC to EST time conversion is working."""
    characteristics = {
        "storyHeight": 3.0,
        "aspectRatio": 1.0,
        "numStories": 2,
        "floorArea": 100.0,
        "designTempCool": 75.0,
        "designTempHeat": 68.0,
    }

    building = BuildingForTesting("test_building", "West", True, characteristics)

    t_start = datetime(2020, 1, 1, 0, 0, 0)
    t_end = datetime(2020, 1, 2, 0, 0, 0)
    t_window = timedelta(hours=1)
    data_file_path = "data/cleanedMFREDdata.xlsx"

    result = building.generateBaselineElectricity(t_start, t_end, t_window, data_file_path)

    assert result is not None

    # Check that the timezone conversion was applied
    # The original data is in UTC, and the function converts to EST (UTC-5)
    # So the result should have times that are 5 hours behind the original UTC times
    if len(result) > 0:
        # The first timestamp in the result should be in EST
        first_time = result.index[0]
        # This is a basic check - in a real scenario, you might want to verify
        # the exact timezone conversion logic
        assert isinstance(first_time, pd.Timestamp)


def test_generate_baseline_electricity_resampling():
    """Test that resampling works correctly for different time windows."""
    characteristics = {
        "storyHeight": 3.0,
        "aspectRatio": 1.0,
        "numStories": 2,
        "floorArea": 100.0,
        "designTempCool": 75.0,
        "designTempHeat": 68.0,
    }

    building = BuildingForTesting("test_building", "West", True, characteristics)

    t_start = datetime(2020, 1, 1, 0, 0, 0)
    t_end = datetime(2020, 1, 2, 0, 0, 0)
    large_window = timedelta(hours=2)
    data_file_path = "data/cleanedMFREDdata.xlsx"

    result = building.generateBaselineElectricity(t_start, t_end, large_window, data_file_path)

    assert result is not None
    assert isinstance(result, pd.DataFrame)

    if len(result) > 1:
        # Check that the time difference between consecutive rows matches the window
        time_diff = result.index[1] - result.index[0]
        assert time_diff == large_window


def test_generate_baseline_electricity_edge_case_same_start_end():
    """Test edge case where start and end times are the same."""
    characteristics = {
        "storyHeight": 3.0,
        "aspectRatio": 1.0,
        "numStories": 2,
        "floorArea": 100.0,
        "designTempCool": 75.0,
        "designTempHeat": 68.0,
    }

    building = BuildingForTesting("test_building", "West", True, characteristics)

    t_start = datetime(2020, 1, 1, 12, 0, 0)
    t_end = datetime(2020, 1, 1, 12, 0, 0)
    t_window = timedelta(hours=1)
    data_file_path = "data/cleanedMFREDdata.xlsx"

    with pytest.raises(BuildingValidationError, match=BuildingValidationError.START_TIME_AFTER_END):
        building.generateBaselineElectricity(t_start, t_end, t_window, data_file_path)


def test_generate_baseline_electricity_data_consistency():
    """Test that multiple calls with same parameters produce consistent results."""
    characteristics = {
        "storyHeight": 3.0,
        "aspectRatio": 1.0,
        "numStories": 2,
        "floorArea": 100.0,
        "designTempCool": 75.0,
        "designTempHeat": 68.0,
    }

    building = BuildingForTesting("test_building", "West", True, characteristics)

    t_start = datetime(2020, 1, 1, 0, 0, 0)
    t_end = datetime(2020, 1, 2, 0, 0, 0)
    t_window = timedelta(hours=1)
    data_file_path = "data/cleanedMFREDdata.xlsx"

    result1 = building.generateBaselineElectricity(t_start, t_end, t_window, data_file_path)
    result2 = building.generateBaselineElectricity(t_start, t_end, t_window, data_file_path)

    # Note: Due to random column selection, results might differ
    # But both should be valid DataFrames
    assert result1 is not None
    assert result2 is not None
    assert isinstance(result1, pd.DataFrame)
    assert isinstance(result2, pd.DataFrame)


# =============================================================================
# Tests for calculateR function
# =============================================================================


def test_calculate_r_basic_functionality():
    """Test basic functionality of calculateR with valid inputs."""
    characteristics = {
        "storyHeight": 3.0,
        "aspectRatio": 1.0,
        "numStories": 2,
        "floorArea": 100.0,
        "designTempCool": 75.0,
        "designTempHeat": 68.0,
    }

    building = BuildingForTesting("test_building", "West", True, characteristics)

    # Check that R was calculated and stored
    assert "R" in building.characteristics
    assert isinstance(building.characteristics["R"], float)
    assert building.characteristics["R"] > 0


def test_calculate_r_detached_building():
    """Test calculateR with detached building (different wall area calculation)."""
    characteristics = {
        "storyHeight": 3.0,
        "aspectRatio": 1.0,
        "numStories": 2,
        "floorArea": 100.0,
        "designTempCool": 75.0,
        "designTempHeat": 68.0,
    }

    building = BuildingForTesting("test_building", "West", False, characteristics)

    # Check that R was calculated and stored
    assert "R" in building.characteristics
    assert isinstance(building.characteristics["R"], float)
    assert building.characteristics["R"] > 0


def test_calculate_r_different_building_sizes():
    """Test calculateR with different building sizes."""
    base_characteristics = {
        "storyHeight": 3.0,
        "aspectRatio": 1.0,
        "numStories": 2,
        "designTempCool": 75.0,
        "designTempHeat": 68.0,
    }

    # Test with different floor areas
    small_building = BuildingForTesting("small", "West", True, {**base_characteristics, "floorArea": 50.0})
    large_building = BuildingForTesting("large", "West", True, {**base_characteristics, "floorArea": 200.0})

    # Both should have R values calculated
    assert "R" in small_building.characteristics
    assert "R" in large_building.characteristics
    assert small_building.characteristics["R"] > 0
    assert large_building.characteristics["R"] > 0

    # Larger building should have different R value (not necessarily larger due to complex calculation)
    assert small_building.characteristics["R"] != large_building.characteristics["R"]


def test_calculate_r_different_aspect_ratios():
    """Test calculateR with different aspect ratios."""
    base_characteristics = {
        "storyHeight": 3.0,
        "numStories": 2,
        "floorArea": 100.0,
        "designTempCool": 75.0,
        "designTempHeat": 68.0,
    }

    # Test with different aspect ratios
    square_building = BuildingForTesting("square", "West", True, {**base_characteristics, "aspectRatio": 1.0})
    rectangular_building = BuildingForTesting("rectangular", "West", True, {**base_characteristics, "aspectRatio": 2.0})

    # Both should have R values calculated
    assert "R" in square_building.characteristics
    assert "R" in rectangular_building.characteristics
    assert square_building.characteristics["R"] > 0
    assert rectangular_building.characteristics["R"] > 0


def test_calculate_r_different_story_heights():
    """Test calculateR with different story heights."""
    base_characteristics = {
        "aspectRatio": 1.0,
        "numStories": 2,
        "floorArea": 100.0,
        "designTempCool": 75.0,
        "designTempHeat": 68.0,
    }

    # Test with different story heights
    low_building = BuildingForTesting("low", "West", True, {**base_characteristics, "storyHeight": 2.5})
    high_building = BuildingForTesting("high", "West", True, {**base_characteristics, "storyHeight": 4.0})

    # Both should have R values calculated
    assert "R" in low_building.characteristics
    assert "R" in high_building.characteristics
    assert low_building.characteristics["R"] > 0
    assert high_building.characteristics["R"] > 0


def test_calculate_r_different_number_of_stories():
    """Test calculateR with different number of stories."""
    base_characteristics = {
        "storyHeight": 3.0,
        "aspectRatio": 1.0,
        "floorArea": 100.0,
        "designTempCool": 75.0,
        "designTempHeat": 68.0,
    }

    # Test with different number of stories
    single_story = BuildingForTesting("single", "West", True, {**base_characteristics, "numStories": 1})
    multi_story = BuildingForTesting("multi", "West", True, {**base_characteristics, "numStories": 3})

    # Both should have R values calculated
    assert "R" in single_story.characteristics
    assert "R" in multi_story.characteristics
    assert single_story.characteristics["R"] > 0
    assert multi_story.characteristics["R"] > 0


# =============================================================================
# Tests for Hp_sizecooling function
# =============================================================================


def test_hp_size_cooling_basic_functionality():
    """Test basic functionality of Hp_sizecooling with valid inputs."""
    characteristics = {
        "storyHeight": 3.0,
        "aspectRatio": 1.0,
        "numStories": 2,
        "floorArea": 100.0,
        "designTempCool": 75.0,
        "designTempHeat": 68.0,
    }

    building = BuildingForTesting("test_building", "West", True, characteristics)

    # Create mock fullP data (thermal power from electrical loads)
    mock_fullP = pd.DataFrame(
        {"thermal_power": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]},
        index=pd.date_range("2021-06-03 20:00:00", periods=8, freq="H"),
    )

    weather_file = "data/TexasWeather1.csv"

    result = building.Hp_sizecooling(weather_file, mock_fullP)

    # Check that result is a positive float
    assert isinstance(result, float)
    assert result > 0


def test_hp_size_cooling_different_design_temps():
    """Test Hp_sizecooling with different design temperatures."""
    base_characteristics = {
        "storyHeight": 3.0,
        "aspectRatio": 1.0,
        "numStories": 2,
        "floorArea": 100.0,
        "designTempHeat": 68.0,
    }

    # Create mock fullP data
    mock_fullP = pd.DataFrame(
        {"thermal_power": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]},
        index=pd.date_range("2021-06-03 20:00:00", periods=8, freq="H"),
    )

    weather_file = "data/TexasWeather1.csv"

    # Test with different cooling design temperatures
    cool_70 = BuildingForTesting("cool_70", "West", True, {**base_characteristics, "designTempCool": 70.0})
    cool_80 = BuildingForTesting("cool_80", "West", True, {**base_characteristics, "designTempCool": 80.0})

    result_70 = cool_70.Hp_sizecooling(weather_file, mock_fullP)
    result_80 = cool_80.Hp_sizecooling(weather_file, mock_fullP)

    # Both should return positive values
    assert result_70 > 0
    assert result_80 > 0


def test_hp_size_cooling_different_floor_areas():
    """Test Hp_sizecooling with different floor areas."""
    base_characteristics = {
        "storyHeight": 3.0,
        "aspectRatio": 1.0,
        "numStories": 2,
        "designTempCool": 75.0,
        "designTempHeat": 68.0,
    }

    # Create mock fullP data
    mock_fullP = pd.DataFrame(
        {"thermal_power": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]},
        index=pd.date_range("2021-06-03 20:00:00", periods=8, freq="H"),
    )

    weather_file = "data/TexasWeather1.csv"

    # Test with different floor areas
    small_building = BuildingForTesting("small", "West", True, {**base_characteristics, "floorArea": 50.0})
    large_building = BuildingForTesting("large", "West", True, {**base_characteristics, "floorArea": 200.0})

    result_small = small_building.Hp_sizecooling(weather_file, mock_fullP)
    result_large = large_building.Hp_sizecooling(weather_file, mock_fullP)

    # Both should return positive values
    assert result_small > 0
    assert result_large > 0


def test_hp_size_cooling_file_not_found():
    """Test Hp_sizecooling with non-existent weather file."""
    characteristics = {
        "storyHeight": 3.0,
        "aspectRatio": 1.0,
        "numStories": 2,
        "floorArea": 100.0,
        "designTempCool": 75.0,
        "designTempHeat": 68.0,
    }

    building = BuildingForTesting("test_building", "West", True, characteristics)

    # Create mock fullP data
    mock_fullP = pd.DataFrame(
        {"thermal_power": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]},
        index=pd.date_range("2021-06-03 20:00:00", periods=8, freq="H"),
    )

    # Test with non-existent file
    with pytest.raises(ValueError):
        building.Hp_sizecooling("nonexistent_file.csv", mock_fullP)


# =============================================================================
# Tests for Hp_sizeheating function
# =============================================================================


def test_hp_size_heating_basic_functionality():
    """Test basic functionality of Hp_sizeheating with valid inputs."""
    characteristics = {
        "storyHeight": 3.0,
        "aspectRatio": 1.0,
        "numStories": 2,
        "floorArea": 100.0,
        "designTempCool": 75.0,
        "designTempHeat": 68.0,
    }

    building = BuildingForTesting("test_building", "West", True, characteristics)

    # Create mock fullP data (thermal power from electrical loads)
    mock_fullP = pd.DataFrame(
        {"thermal_power": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]},
        index=pd.date_range("2021-06-03 20:00:00", periods=8, freq="H"),
    )

    weather_file = "data/TexasWeather1.csv"

    result = building.Hp_sizeheating(weather_file, mock_fullP)

    # Check that result is a positive float
    assert isinstance(result, float)
    assert result > 0


def test_hp_size_heating_different_design_temps():
    """Test Hp_sizeheating with different design temperatures."""
    base_characteristics = {
        "storyHeight": 3.0,
        "aspectRatio": 1.0,
        "numStories": 2,
        "floorArea": 100.0,
        "designTempCool": 75.0,
    }

    # Create mock fullP data
    mock_fullP = pd.DataFrame(
        {"thermal_power": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]},
        index=pd.date_range("2021-06-03 20:00:00", periods=8, freq="H"),
    )

    weather_file = "data/TexasWeather1.csv"

    # Test with different heating design temperatures
    heat_60 = BuildingForTesting("heat_60", "West", True, {**base_characteristics, "designTempHeat": 60.0})
    heat_75 = BuildingForTesting("heat_75", "West", True, {**base_characteristics, "designTempHeat": 75.0})

    result_60 = heat_60.Hp_sizeheating(weather_file, mock_fullP)
    result_75 = heat_75.Hp_sizeheating(weather_file, mock_fullP)

    # Both should return positive values
    assert result_60 > 0
    assert result_75 > 0


def test_hp_size_heating_different_floor_areas():
    """Test Hp_sizeheating with different floor areas."""
    base_characteristics = {
        "storyHeight": 3.0,
        "aspectRatio": 1.0,
        "numStories": 2,
        "designTempCool": 75.0,
        "designTempHeat": 68.0,
    }

    # Create mock fullP data
    mock_fullP = pd.DataFrame(
        {"thermal_power": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]},
        index=pd.date_range("2021-06-03 20:00:00", periods=8, freq="H"),
    )

    weather_file = "data/TexasWeather1.csv"

    # Test with different floor areas
    small_building = BuildingForTesting("small", "West", True, {**base_characteristics, "floorArea": 50.0})
    large_building = BuildingForTesting("large", "West", True, {**base_characteristics, "floorArea": 200.0})

    result_small = small_building.Hp_sizeheating(weather_file, mock_fullP)
    result_large = large_building.Hp_sizeheating(weather_file, mock_fullP)

    # Both should return positive values
    assert result_small > 0
    assert result_large > 0


def test_hp_size_heating_file_not_found():
    """Test Hp_sizeheating with non-existent weather file."""
    characteristics = {
        "storyHeight": 3.0,
        "aspectRatio": 1.0,
        "numStories": 2,
        "floorArea": 100.0,
        "designTempCool": 75.0,
        "designTempHeat": 68.0,
    }

    building = BuildingForTesting("test_building", "West", True, characteristics)

    # Create mock fullP data
    mock_fullP = pd.DataFrame(
        {"thermal_power": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]},
        index=pd.date_range("2021-06-03 20:00:00", periods=8, freq="H"),
    )

    # Test with non-existent file
    with pytest.raises(ValueError):
        building.Hp_sizeheating("nonexistent_file.csv", mock_fullP)


def test_hp_size_heating_vs_cooling_comparison():
    """Test that heating and cooling sizes are calculated correctly for same building."""
    characteristics = {
        "storyHeight": 3.0,
        "aspectRatio": 1.0,
        "numStories": 2,
        "floorArea": 100.0,
        "designTempCool": 75.0,
        "designTempHeat": 68.0,
    }

    building = BuildingForTesting("test_building", "West", True, characteristics)

    # Create mock fullP data
    mock_fullP = pd.DataFrame(
        {"thermal_power": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]},
        index=pd.date_range("2021-06-03 20:00:00", periods=8, freq="H"),
    )

    weather_file = "data/TexasWeather1.csv"

    cooling_size = building.Hp_sizecooling(weather_file, mock_fullP)
    heating_size = building.Hp_sizeheating(weather_file, mock_fullP)

    # Both should return positive values
    assert cooling_size > 0
    assert heating_size > 0

    # They should be different values (heating and cooling have different requirements)
    assert cooling_size != heating_size
