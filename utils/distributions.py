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
