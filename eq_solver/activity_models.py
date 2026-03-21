from __future__ import annotations
from typing import Any
import numpy as np

from eq_solver._types import Float, Int


def epsilon_r(temp: Float) -> Float:
    """relative permittivity of water.

    Args:
        temp: temperature in Kelvin
    """
    t_deg = temp - 273.15
    return 88.15 - 0.414 * t_deg + 0.131 * 1e-2 * t_deg**2 - 0.046 * 1e-4 * t_deg**3

def A(temp: Float) -> Float:
    """prefactor A in (mol/L)^(-0.5)"""
    return 1.82e6 * (epsilon_r(temp) * temp)**(-1.5)

def B(temp: Float) -> Float:
    """prefactor B in nm^-1 (mol/L)^(-0.5)"""
    return 503.0 * (epsilon_r(temp) * temp)**(-0.5)

def debye_huckel(I: Float, z: Int, temp: Float, a=None, b=None) -> Float:
    return 10.**(-A(temp) * z**2 * np.sqrt(I))

def davies(I: Float, z: Int, temp: Float, a=None, b=None) -> Float:
    lgG = -A(temp) * z**2 * (np.sqrt(I) / (1 + np.sqrt(I)) - 0.3 * I)
    return 10.**lgG

def truesdell_jones(I: Float, z: Int, temp: Float, a: Float, b: Float) -> Float:
    lgG = -A(temp) * z**2 * np.sqrt(I) / (1 + B(temp) * a * np.sqrt(I)) + b * I
    return 10.**lgG

def ex_debye_huckel(I: Float, z: Int, temp: Float, a: Float, b=None) -> Float:
    lgG = -A(temp) * z**2 * np.sqrt(I) / (1 + B(temp) * a * np.sqrt(I))
    return 10.**lgG

functions = {
    'debye_huckel': debye_huckel,
    'ex_debye_huckel': ex_debye_huckel,
    'davies': davies,
    'truesdell_jones': truesdell_jones,
    'none': lambda I, z, temp, a, b: np.ones_like(z),
}
