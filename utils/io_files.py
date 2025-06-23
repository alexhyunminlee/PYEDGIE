import csv


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
                    # Write the corresponding value or an empty string if out of
                    # range
                    row.append(data_dict[key][i] if i < len(data_dict[key]) else "")
            writer.writerow(row)
