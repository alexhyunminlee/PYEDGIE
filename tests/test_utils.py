import os
import sys

from PYEDGIE.utils.conversions import btu2wh, c2f, f2c

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_f2c() -> None:
    assert f2c(32) == 0
    assert f2c(212) == 100
    assert f2c(98.6) == 37.0


def test_c2f() -> None:
    assert c2f(0) == 32
    assert c2f(100) == 212
    assert c2f(37.77777777777778) == 100


def test_btu2wh() -> None:
    assert btu2wh(1000) == 293.071
    assert btu2wh(10000) == 2930.71
