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
    # You can add more asserts for other columns or rows as needed


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
