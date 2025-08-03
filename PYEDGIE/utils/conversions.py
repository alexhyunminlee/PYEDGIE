def f2c(Fahrenheit: float) -> float:
    """
    Takes a temperature measurement in Fahrenheit and converts it to Celsius

    Parameters:
    - Fahrenheit: Temperature measurement in Fahrenheit

    Returns:
    - Temperature measurement in Celsius

    Authors: Priyadarshan (priyada@purdue.edu),
             Alex Lee (alexlee5124@gmail.com)
    Date: 12/12/2024
    """
    return (Fahrenheit - 32) * 5.0 / 9.0


def c2f(Celisus: float) -> float:
    """
    Takes a temperature measurement in Celsius and converts it to Fahrenheit

    Parameters:
    - Celisus: Temperature measurement in Celisus

    Returns:
    - Temperature measurement in Fahrenheit

    Authors: Priyadarshan (priyada@purdue.edu),
             Alex Lee (alexlee5124@gmail.com)
    Date: 12/12/2024
    """
    return (Celisus * 9.0 / 5.0) + 32


def btu2wh(btu: float) -> float:
    return btu * 0.293071
