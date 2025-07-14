import pandas as pd
import pytest

from utils import fileIO


def test_csv_to_dict():
    file_path = "data/TexasWeather1.csv"
    result = fileIO.csv_to_dict(file_path)
    assert len(result) == 8
    assert result[0]["coordinates (lat,lon)"] == "(31.9973, -102.0779)"
    assert result[0]["model (name)"] == "era5"
    assert result[0]["temperature (degC)"] == "22.95"
    assert result[0]["wind_speed (m/s)"] == "4.19"
    assert result[0]["relative_humidity (0-1)"] == "0.66"


def test_save_dict_to_csv(tmp_path):
    # Test data dictionary
    test_data = {"name": ["John", "Jane", "Bob"], "age": [25, 30, 35], "city": ["Austin", "Dallas", "Houston"]}

    # Create output file path
    output_file = tmp_path / "test_output.csv"

    # Call the function
    fileIO.save_dict_to_csv(test_data, str(output_file))

    # Verify file was created
    assert output_file.exists()

    # Read the file back and verify contents
    with open(output_file) as f:
        content = f.read()

    # Check that headers are present
    assert "name,age,city" in content

    # Check that data rows are present
    assert "John,25,Austin" in content
    assert "Jane,30,Dallas" in content
    assert "Bob,35,Houston" in content


def test_process_weather_data_types():
    # Create a sample DataFrame similar to expected input
    df = pd.DataFrame({
        "temperature (degC)": ["20.0", "21.5"],
        "wind_speed (m/s)": ["3.0", "4.0"],
        "relative_humidity (0-1)": ["0.5", "0.6"],
        "surface_solar_radiation (W/m^2)": ["1000", "2000"],
        "direct_normal_solar_radiation (W/m^2)": ["500", "1000"],
        "surface_diffuse_solar_radiation (W/m^2)": ["200", "400"],
    })
    result = fileIO._process_weather_data_types(df)
    # Check type conversions
    assert all(isinstance(x, float) for x in result["temperature (degC)"])
    assert all(isinstance(x, float) for x in result["wind_speed (m/s)"])
    assert all(isinstance(x, float) for x in result["relative_humidity (0-1)"])
    # Check new columns for kW/m^2
    assert "surface_solar_radiation_kW_per_m2" in result
    assert result["surface_solar_radiation_kW_per_m2"].tolist() == [1.0, 2.0]
    assert "direct_normal_solar_kW_per_m2" in result
    assert result["direct_normal_solar_kW_per_m2"].tolist() == [0.5, 1.0]
    assert "surface_diffuse_solar_kW_per_m2" in result
    assert result["surface_diffuse_solar_kW_per_m2"].tolist() == [0.2, 0.4]


def test_validate_datetime_range():
    from datetime import datetime

    import pandas as pd

    # Create a sample DataFrame with datetime index
    dates = pd.date_range("2023-01-01", periods=5, freq="H")
    df = pd.DataFrame({"temp": [20, 21, 22, 23, 24]}, index=dates)

    # Test valid range
    start = datetime(2023, 1, 1, 1, 0)  # 1 hour after start
    end = datetime(2023, 1, 1, 3, 0)  # 3 hours after start
    fileIO._validate_datetime_range(df, start, end)  # Should not raise error

    # Test invalid range - start >= end
    start = datetime(2023, 1, 1, 3, 0)
    end = datetime(2023, 1, 1, 1, 0)
    with pytest.raises(ValueError):
        fileIO._validate_datetime_range(df, start, end)

    # Test invalid range - start before data range
    start = datetime(2022, 12, 31, 23, 0)
    end = datetime(2023, 1, 1, 1, 0)
    with pytest.raises(ValueError):
        fileIO._validate_datetime_range(df, start, end)
