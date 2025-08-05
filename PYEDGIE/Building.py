from datetime import datetime, timedelta
from random import randrange
from typing import Any

import numpy as np
import pandas as pd
from ochre.Simulator import Simulator

from .utils import conversions, distributions, fileIO


class BuildingValidationError(ValueError):
    """Custom exception for building validation errors."""

    pass


# Error message constants
INVALID_REGION_MSG = "Please have an appropriate region for your building"
INVALID_TIME_PARAMS_MSG = "Please provide appropriate time parameters"
INVALID_TIME_ORDER_MSG = "The start time should be before the end time"
INVALID_ATTACHED_STATUS_MSG = "Please set the attached/detached status for your building"


class Building(Simulator):  # type: ignore[no-any-unimported]
    def __init__(self, name: str, region: str, attached: bool, characteristics: dict[str, float | int | str]) -> None:
        super().__init__()
        self.name = name
        self.region = region
        self.attached = attached
        self.characteristics = characteristics
        self.calculateR()

    # Getters
    def getRegion(self) -> str:
        return self.region

    def getAttached(self) -> bool:
        return self.attached

    def getStoryHeight(self) -> float:
        return float(self.characteristics["storyHeight"])

    def getAspectRation(self) -> float:
        return float(self.characteristics["aspectRatio"])

    def getNumStories(self) -> int:
        return int(self.characteristics["numStories"])

    def getFloorArea(self) -> float:
        return float(self.characteristics["floorArea"])

    def getR(self) -> float:
        return float(self.characteristics["R"])

    def getDesignTempCool(self) -> float:
        return float(self.characteristics["designTempCool"])

    def getDesignTempHeat(self) -> float:
        return float(self.characteristics["designTempHeat"])

    # Setters
    def setRegion(self, region: str) -> None:
        self.region = region

    def setAttached(self, attached: bool) -> None:
        self.attached = attached

    def setStoryHeight(self, storyHeight: float) -> None:
        self.characteristics["storyHeight"] = storyHeight

    def setAspectRation(self, aspectRatio: float) -> None:
        self.characteristics["aspectRatio"] = aspectRatio

    def setNumStories(self, numStories: int) -> None:
        self.characteristics["numStories"] = numStories

    def setFloorArea(self, floorArea: float) -> None:
        self.characteristics["floorArea"] = floorArea

    def setR(self, R: float) -> None:
        self.characteristics["R"] = R

    def setDesignTempCool(self, designTempCool: float) -> None:
        self.characteristics["designTempCool"] = designTempCool

    def setDesignTempHeat(self, designTempHeat: float) -> None:
        self.characteristics["designTempHeat"] = designTempHeat

    def _validate_region(self) -> None:
        """Validate that region is one of the allowed values."""
        valid_regions = ["West", "Midwest", "South", "Northeast"]
        if self.region not in valid_regions:
            raise BuildingValidationError(INVALID_REGION_MSG)

    def _validate_time_parameters(self, tStart: datetime, tEnd: datetime, tWindow: timedelta) -> None:
        """Validate time parameters."""
        if tStart is None or tEnd is None or tWindow is None:
            raise BuildingValidationError(INVALID_TIME_PARAMS_MSG)
        if not (tStart < tEnd):
            raise BuildingValidationError(INVALID_TIME_ORDER_MSG)

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

    def _scale_data(self, scaledData: Any, scaling_detached: float, scaling_attached: float) -> Any:
        """Scale the data based on building characteristics."""
        # Scale data
        scaledData.update(scaledData.iloc[:, 1:].div(scaledData.iloc[:, 1:].mean(axis=0), axis="columns"))

        if self.attached:
            scaledData.update(scaledData.iloc[:, 1:].mul(scaling_attached))
        else:
            scaledData.update(scaledData.iloc[:, 1:].mul(scaling_detached))
        return scaledData

    def _create_time_series(self, scaledData: Any, tStart: datetime, tEnd: datetime) -> Any:
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
    ) -> Any | None:
        """
        Read the baseline electricity load file, scale it based on the building characteristics,
        and choose the data to be between the desired start and end time.

        Authors: Priyadarshan (priyada@purdue.edu), Alex Lee (alexlee5124@gmail.com), Zachary Tan
        Date: 2/4/2025
        """
        # Error checking
        self._validate_region()
        if self.attached is None:
            raise BuildingValidationError(INVALID_ATTACHED_STATUS_MSG)
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
        # Using randrange for data selection, not cryptographic purposes
        scaledData = scaledData.iloc[:, [0, randrange(1, scaledData.shape[1])]]  # noqa: S311

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
        """
        Calculate overall thermal resistance R parameter for the building.

        Authors: Priyadarshan (priyada@purdue.edu), Alex Lee (alexlee5124@gmail.com), Zachary Tan
        Date: 2/20/2025
        """
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

    def Hp_sizecooling(self, filepath: str, fullP: Any) -> float:
        """
        Simulate heat pump with resistance heating equiments
        """

        t1 = 1
        R = float(self.characteristics["R"])
        floorArea = self.characteristics["floorArea"]
        designTempCool = float(self.characteristics["designTempCool"])

        weather = fileIO.importWeather(filepath)
        thetaFull = weather["temperature (degC)"]
        solarFull = weather["surface_solar_radiation_kW_per_m2"]
        K = len(thetaFull)

        start_idx = solarFull.index[0].hour
        end_idx = solarFull.index[0].hour + K
        electricLoadThermalPower = fullP.iloc[
            start_idx:end_idx
        ]  # thermal power from electrical loads other than heat pump, kW
        bodyThermalPower = np.array(distributions.gauss(K, t1, 0.1, 0.5))  # thermal power from body heat, kW
        solarThermalPower = (
            0.03
            * np.array(distributions.gauss(K, t1, 0.9, 1.1))
            * float(floorArea)
            * np.array(solarFull).reshape(-1, 1)
        )  # thermal power from sunlight, kW

        qe1 = electricLoadThermalPower + bodyThermalPower + solarThermalPower

        differences = abs(designTempCool - thetaFull)

        ind = np.argmin(differences)
        # Find the index of the minimum absolute difference

        Tset = (np.array(distributions.trirnd(61, 76, 1, 1)) - 32.0) * 5.0 / 9.0
        thermalPower1 = abs((Tset - conversions.f2c(designTempCool)) / R - np.array(qe1)[ind])
        Hp_Cooling = np.ceil(distributions.trirnd(1.2, 1.3, 1, 1) * (thermalPower1 / (0.80)))

        return float(Hp_Cooling[0, 0])

    def Hp_sizeheating(self, filepath: str, fullP: Any) -> float:
        """
        Simulate heat pump with resistance heating equipment
        """

        t1 = 1
        R = float(self.characteristics["R"])
        floorArea = self.characteristics["floorArea"]
        designTempHeat = float(self.characteristics["designTempHeat"])

        weather = fileIO.importWeather(filepath)
        thetaFull = weather["temperature (degC)"]
        solarFull = weather["surface_solar_radiation_kW_per_m2"]
        K = len(thetaFull)

        start_idx = solarFull.index[0].hour
        end_idx = solarFull.index[0].hour + K
        electricLoadThermalPower = fullP.iloc[
            start_idx:end_idx
        ]  # thermal power from electrical loads other than heat pump, kW
        bodyThermalPower = np.array(distributions.gauss(K, t1, 0.1, 0.5))  # thermal power from body heat, kW
        solarThermalPower = (
            0.03
            * np.array(distributions.gauss(K, t1, 0.9, 1.1))
            * float(floorArea)
            * np.array(solarFull).reshape(-1, 1)
        )  # thermal power from sunlight, kW

        qe1 = electricLoadThermalPower + bodyThermalPower + solarThermalPower

        differences = abs(designTempHeat - thetaFull)

        ind = np.argmin(differences)
        # Find the index of the minimum absolute difference

        Tset = (np.array(distributions.trirnd(61, 76, 1, 1)) - 32.0) * 5.0 / 9.0
        thermalPower1 = abs((Tset - conversions.f2c(designTempHeat)) / R - np.array(qe1)[ind])
        Hp_Heating = np.ceil(distributions.trirnd(1.2, 1.3, 1, 1) * (thermalPower1 / (0.80)))

        return float(Hp_Heating[0, 0])

    def select_best_hvac_heating(
        self, y_target: float
    ) -> tuple[float | None, float | None, float | None, float | None]:
        """
        Select the best HVAC heating unit based on target heating capacity
        """
        x_target = float(self.characteristics["designTempHeat"])
        # import hvac data
        heating_files = [
            "2ton_A_heating.csv",
            "2ton_B_heating.csv",
            "3ton_C_heating.csv",
            "3ton_A_heating.csv",
            "3ton_B_heating.csv",
            "4ton_C_heating.csv",
            "4ton_A_heating.csv",
            "4ton_B_heating.csv",
            "5ton_C_heating.csv",
            "5ton_A_heating.csv",
            "5ton_B_heating.csv",
            "6ton_C_heating.csv",
            "6ton_A_heating.csv",
            "6ton_B_heating.csv",
            "7ton_C_heating.csv",
            "7ton_A_heating.csv",
            "7ton_B_heating.csv",
            "8ton_C_heating.csv",
            "8ton_A_heating.csv",
            "8ton_B_heating.csv",
        ]

        # Temperature range for heating
        temp_min_heating = conversions.f2c(5)
        temp_max_heating = conversions.f2c(48)

        # Initialize best heating unit selection
        best_curve_error = float("inf")
        best_heating_file = ""
        best_heating_data: np.ndarray | None = None
        heating_val_5C: float | None = None
        heating_val_45C: float | None = None

        # --- HEATING SELECTION ---
        for filename in heating_files:
            try:
                tbl = pd.read_csv(filename)
                if tbl.shape[1] >= 2:
                    x = np.vectorize(conversions.f2c)(tbl.iloc[:, 0].values)
                    y = np.vectorize(conversions.btu2wh)(tbl.iloc[:, 1].values)
                    valid_idx = (x >= temp_min_heating) & (x <= temp_max_heating)
                    x = x[valid_idx]
                    y = y[valid_idx]

                    # Linear fit
                    if len(x) > 1:
                        coeffs = np.polyfit(x, y, 1)
                        y_pred = float(np.polyval(coeffs, x_target))
                        error = abs(y_pred - y_target)

                        # Check if this is the best heating unit
                        if error < best_curve_error:
                            best_curve_error = error
                            best_heating_file = filename
                            best_heating_data = coeffs

            except Exception:
                print("Error loading %s\n", filename)

        # Display selected heating unit
        if best_heating_data is not None:
            heating_val_5C = float(np.polyval(best_heating_data, conversions.f2c(5)))
            heating_val_45C = float(np.polyval(best_heating_data, conversions.f2c(45)))
        else:
            print("No valid heating data found.\n")
            return None, None, None, None

        # --- COOLING SELECTION ---
        best_cooling_file = best_heating_file.replace("heating", "cooling")
        try:
            tbl = pd.read_csv(best_cooling_file)
            if tbl.shape[1] >= 2:
                x = np.vectorize(conversions.f2c)(tbl.iloc[:, 0].values)
                y = np.vectorize(conversions.btu2wh)(tbl.iloc[:, 1].values)
                cooling_val_82F = float(np.interp(conversions.f2c(82), x, y))
                cooling_val_95F = float(np.interp(conversions.f2c(95), x, y))
            else:
                print("No valid cooling data found.\n")
                return None, None, heating_val_5C, heating_val_45C
        except Exception:
            print("Error loading %s\n", best_cooling_file)
            return None, None, heating_val_5C, heating_val_45C
        return cooling_val_95F, cooling_val_82F, heating_val_5C, heating_val_45C

    def select_best_hvac_cooling(
        self, y_target: float
    ) -> tuple[float | None, float | None, float | None, float | None]:
        """
        Select the best HVAC cooling unit based on target cooling capacity
        """
        x_target = float(self.characteristics["designTempCool"])
        # import hvac data
        cooling_files = [
            "2ton_A_cooling.csv",
            "2ton_B_cooling.csv",
            "3ton_C_cooling.csv",
            "3ton_A_cooling.csv",
            "3ton_B_cooling.csv",
            "4ton_C_cooling.csv",
            "4ton_A_cooling.csv",
            "4ton_B_cooling.csv",
            "5ton_C_cooling.csv",
            "5ton_A_cooling.csv",
            "5ton_B_cooling.csv",
            "6ton_C_cooling.csv",
            "6ton_A_cooling.csv",
            "6ton_B_cooling.csv",
            "7ton_C_cooling.csv",
            "7ton_A_cooling.csv",
            "7ton_B_cooling.csv",
            "8ton_C_cooling.csv",
            "8ton_A_cooling.csv",
            "8ton_B_cooling.csv",
        ]
        best_curve_error = float("inf")
        best_cooling_file = ""
        best_cooling_data: np.ndarray | None = None
        cooling_val_82F: float | None = None
        cooling_val_95F: float | None = None

        # --- COOLING SELECTION ---
        for filename in cooling_files:
            try:
                tbl = pd.read_csv(filename)
                if tbl.shape[1] >= 2:
                    x = np.vectorize(conversions.f2c)(tbl.iloc[:, 0].values)
                    y = np.vectorize(conversions.btu2wh)(tbl.iloc[:, 1].values)

                    # Linear fit
                    coeffs = np.polyfit(x, y, 1)
                    y_pred = float(np.polyval(coeffs, x_target))
                    error = abs(y_pred - y_target)

                    # Check if this is the best cooling unit
                    if error < best_curve_error:
                        best_curve_error = error
                        best_cooling_file = filename
                        best_cooling_data = coeffs

            except Exception:
                print("Error loading %s\n", filename)

        # Display selected cooling unit
        if best_cooling_data is not None:
            cooling_val_82F = float(np.polyval(best_cooling_data, conversions.f2c(82)))
            cooling_val_95F = float(np.polyval(best_cooling_data, conversions.f2c(95)))
        else:
            print("No valid cooling data found.\n")
            return None, None, None, None

        # --- HEATING SELECTION ---
        best_heating_file = best_cooling_file.replace("cooling", "heating")
        try:
            tbl = pd.read_csv(best_heating_file)
            if tbl.shape[1] >= 2:
                x = np.vectorize(conversions.f2c)(tbl.iloc[:, 0].values)
                y = np.vectorize(conversions.btu2wh)(tbl.iloc[:, 1].values)
                heating_val_5C = float(np.interp(conversions.f2c(5), x, y))
                heating_val_45C = float(np.interp(conversions.f2c(45), x, y))
            else:
                print("No valid heating data found.\n")
                return cooling_val_82F, cooling_val_95F, None, None
        except Exception:
            print("Error loading %s\n", best_heating_file)
            return cooling_val_82F, cooling_val_95F, None, None
        return cooling_val_95F, cooling_val_82F, heating_val_5C, heating_val_45C


if __name__ == "__main__":
    # characteristics: dict[str, float | int | str] = {
    #     "storyHeight": 3.0,
    #     "aspectRatio": 1.0,
    #     "numStories": 2,
    #     "floorArea": 100.0,
    #     "designTempCool": 75.0,
    #     "designTempHeat": 68.0,
    # }
    # try:
    #     building = Building("test_building", "West", True, characteristics)
    # except Exception as e:
    #     print(f"Error creating building: {e}")
    pass
