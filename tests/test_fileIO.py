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
