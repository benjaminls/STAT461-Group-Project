"""
Common utility functions used across multiple modules.

This module provides general-purpose utility functions that are used by other modules
in the codebase. It contains helper functions and shared functionality that doesn't
fit specifically into other specialized modules.

The utilities here are designed to be generic and reusable across different parts
of the particle tracking pipeline, avoiding code duplication and promoting consistent
implementations.
"""

import logging
import os
import numpy as np


def _init_logger() -> logging.Logger:
    """Set up logger for a script.

    Returns:
        logging.Logger: Configured logger instance
    """
    console_handler = logging.StreamHandler()
    level = getattr(logging, os.environ.get("PYTHON_GNN_LOG_LEVEL", "INFO").upper(), None)
    if level is None:
        raise ValueError(f"Invalid log level: {os.environ.get('PYTHON_GNN_LOG_LEVEL')}")
    console_handler.setLevel(level)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    new_logger = logging.getLogger(__name__)
    new_logger.addHandler(console_handler)
    new_logger.setLevel(level)
    return new_logger

logger = _init_logger()


def cartesian_to_eta_phi_r(x: float, y: float, z: float) -> tuple[float, float, float]:
    """Convert Cartesian coordinates to pseudorapidity (eta) and azimuthal angle (phi).

    Args:
        x (float): x-coordinate
        y (float): y-coordinate
        z (float): z-coordinate

    Returns:
        tuple[float, float, float]: Pseudorapidity (eta), azimuthal angle (phi), and radius (r) from interaction point
    """
    r = np.sqrt(x**2 + y**2 + z**2)  # radius from interaction point
    r_xy = np.sqrt(x**2 + y**2)  # radius in xy plane
    theta = np.abs(np.arctan(r_xy/z))  # polar angle between direction and positive z-axis
    eta = -np.log(np.tan(theta/2))  # pseudorapidity
    phi = np.arctan2(y, x)  # azimuthal angle in [-pi,pi]
    phi = phi if phi >= 0 else phi + 2*np.pi  # convert to [0,2pi] range
    return eta, phi, r

def cartesian_to_eta_phi(x: float, y: float, z: float) -> tuple[float, float]:
    """Convert Cartesian coordinates to pseudorapidity (eta) and azimuthal angle (phi).

    Args:
        x (float): x-coordinate
        y (float): y-coordinate
        z (float): z-coordinate

    Returns:
        tuple[float, float]: Pseudorapidity (eta), azimuthal angle (phi)
    """
    r_xy = np.sqrt(x**2 + y**2)  # radius in xy plane
    theta = np.abs(np.arctan(r_xy/z))  # polar angle between direction and positive z-axis
    eta = -np.log(np.tan(theta/2))  # pseudorapidity
    phi = np.arctan2(y, x)  # azimuthal angle in [-pi,pi]
    phi = phi if phi >= 0 else phi + 2*np.pi  # convert to [0,2pi] range
    return eta, phi

def eta_phi_to_cartesian(eta: float, phi: float, r: float = 1) -> tuple[float, float, float]:
    """Convert pseudorapidity (eta) and azimuthal angle (phi) to Cartesian coordinates.

    Args:
        eta (float): Pseudorapidity
        phi (float): Azimuthal angle
        r (float): Distance from interaction point

    Returns:
        tuple[float, float, float]: Cartesian coordinates (x, y, z)
    """
    x = r * np.cos(phi) / np.cosh(eta)
    y = r * np.sin(phi) / np.cosh(eta) 
    z = r * np.tanh(eta)
    return x, y, z


if __name__ == "__main__":
    example_xyz = [
        (1, 0, 0.1), # x-axis
        (1, 0.25, 0.1), # a bit up from x-axis in y direction (positive phi)
        (1, -0.25, 0.1), # a bit up from x-axis in y direction (negative phi)
        (0, 1, 0.1), # y-axis
        (0, 1, 5), # endcap event
        ]
    for x, y, z in example_xyz:
        eta, phi, r = cartesian_to_eta_phi_r(x, y, z)
        x_new, y_new, z_new = eta_phi_to_cartesian(eta, phi, r=r)
        print(f"({x}, {y}, {z}) -> eta: {eta:.5f}, phi: {phi:.5f} -> ({x_new:.5f}, {y_new:.5f}, {z_new:.5f})")
        # print(f"({eta:.5f}, {phi:.5f}) -> ({x_new:.5f}, {y_new:.5f}, {z_new:.5f})")
        print()
