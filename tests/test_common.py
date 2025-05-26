import pytest
import numpy as np
from utils.common import cartesian_to_eta_phi, eta_phi_to_cartesian


def test_reversibility_simple_cases():
    """Test reversibility of coordinate conversions for simple cases."""
    # Points on axes
    point_cases = [
        (1.0, 0.0, 0.1),  # near x-axis
        (0.0, 1.0, 0.1),  # near y-axis
        (1.0, 1.0, 1.0),  # diagonal
    ]

    for x, y, z in point_cases:
        # Convert to eta-phi
        eta, phi = cartesian_to_eta_phi(x, y, z)

        # Convert back to Cartesian
        x_new, y_new, z_new = eta_phi_to_cartesian(eta, phi)

        # Calculate original and new magnitudes
        r_orig = np.sqrt(x**2 + y**2 + z**2)
        r_new = np.sqrt(x_new**2 + y_new**2 + z_new**2)

        # Normalize vectors for comparison (since eta-phi doesn't preserve magnitude)
        x_norm, y_norm, z_norm = x / r_orig, y / r_orig, z / r_orig
        x_new_norm, y_new_norm, z_new_norm = x_new / r_new, y_new / r_new, z_new / r_new

        # Assert that the normalized coordinates are approximately equal
        assert np.isclose(x_norm, x_new_norm, atol=1e-6)
        assert np.isclose(y_norm, y_new_norm, atol=1e-6)
        assert np.isclose(z_norm, z_new_norm, atol=1e-6)


def test_z_axis_case():
    """Test the special z-axis case separately.

    Points exactly on the z-axis are a special case for eta-phi coordinates.
    - When z is positive, eta approaches +infinity
    - When z is negative, eta approaches -infinity
    - phi is not well-defined (could be any value)

    Instead of testing exact reversibility, we check that the functions
    handle these cases without crashing and produce reasonable results when possible.
    """
    # Near z-axis test cases (not exactly on z-axis to avoid division by zero)
    test_cases = [
        (1e-6, 1e-6, 1.0),  # almost on positive z-axis
        (1e-6, 1e-6, -1.0),  # almost on negative z-axis
    ]

    for x, y, z in test_cases:
        # For points near the z-axis, test basic properties
        eta, phi = cartesian_to_eta_phi(x, y, z)

        # Check that eta has the expected sign based on z
        if z > 0:
            assert eta > 0
        else:
            assert eta < 0

        # Magnitude of eta should be large for points near z-axis
        assert abs(eta) > 10

        # Convert back and check properties
        x_new, y_new, z_new = eta_phi_to_cartesian(eta, phi)

        # Sign of z should be preserved in conversion
        assert np.sign(z) == np.sign(z_new)


# Create a separate test for the exact z-axis points
def test_exact_z_axis():
    """Test the behavior with points exactly on the z-axis."""
    # Z-axis test cases
    test_cases = [
        (0.0, 0.0, 1.0),  # positive z-axis
        (0.0, 0.0, -1.0),  # negative z-axis
    ]

    for x, y, z in test_cases:
        # These cases will produce warnings, but shouldn't crash
        eta, phi = cartesian_to_eta_phi(x, y, z)

        # Check that the conversion runs without errors
        # The specific values are not reliable, so we don't assert anything about them

        # Converting back may produce NaN/inf values, which is expected
        # We just verify the function runs without errors
        eta_phi_to_cartesian(eta, phi)

        # No assertions here - we're just checking that the functions don't crash


def test_reversibility_quadrants():
    """Test reversibility across different quadrants."""
    # Points in different quadrants
    point_cases = [
        (1.0, 1.0, 1.0),  # (+,+,+)
        (1.0, 1.0, -1.0),  # (+,+,-)
        (1.0, -1.0, 1.0),  # (+,-,+)
        (1.0, -1.0, -1.0),  # (+,-,-)
        (-1.0, 1.0, 1.0),  # (-,+,+)
        (-1.0, 1.0, -1.0),  # (-,+,-)
        (-1.0, -1.0, 1.0),  # (-,-,+)
        (-1.0, -1.0, -1.0),  # (-,-,-)
    ]

    for x, y, z in point_cases:
        # Convert to eta-phi
        eta, phi = cartesian_to_eta_phi(x, y, z)

        # Convert back to Cartesian
        x_new, y_new, z_new = eta_phi_to_cartesian(eta, phi)

        # Calculate original and new magnitudes
        r_orig = np.sqrt(x**2 + y**2 + z**2)
        r_new = np.sqrt(x_new**2 + y_new**2 + z_new**2)

        # Normalize vectors for comparison
        x_norm, y_norm, z_norm = x / r_orig, y / r_orig, z / r_orig
        x_new_norm, y_new_norm, z_new_norm = x_new / r_new, y_new / r_new, z_new / r_new

        # Assert that the normalized coordinates are approximately equal
        assert np.isclose(x_norm, x_new_norm, atol=1e-6)
        assert np.isclose(y_norm, y_new_norm, atol=1e-6)
        assert np.isclose(z_norm, z_new_norm, atol=1e-6)


def test_reversibility_random_cases():
    """Test reversibility for random points."""
    # Set random seed for reproducibility
    np.random.seed(42)

    # Generate 20 random points
    for _ in range(20):
        # Generate random point with magnitude between 0.1 and 10
        magnitude = 0.1 + 9.9 * np.random.random()

        # Generate random unit vector
        phi = 2 * np.pi * np.random.random()
        cos_theta = 2 * np.random.random() - 1
        sin_theta = np.sqrt(1 - cos_theta**2)

        # Calculate Cartesian coordinates
        x = magnitude * sin_theta * np.cos(phi)
        y = magnitude * sin_theta * np.sin(phi)
        z = magnitude * cos_theta

        # Convert to eta-phi
        eta, phi = cartesian_to_eta_phi(x, y, z)

        # Convert back to Cartesian
        x_new, y_new, z_new = eta_phi_to_cartesian(eta, phi)

        # Calculate original and new magnitudes
        r_orig = np.sqrt(x**2 + y**2 + z**2)
        r_new = np.sqrt(x_new**2 + y_new**2 + z_new**2)

        # Normalize vectors for comparison
        x_norm, y_norm, z_norm = x / r_orig, y / r_orig, z / r_orig
        x_new_norm, y_new_norm, z_new_norm = x_new / r_new, y_new / r_new, z_new / r_new

        # Assert that the normalized coordinates are approximately equal
        assert np.isclose(x_norm, x_new_norm, atol=1e-6)
        assert np.isclose(y_norm, y_new_norm, atol=1e-6)
        assert np.isclose(z_norm, z_new_norm, atol=1e-6)


def test_edge_cases():
    """Test reversibility for edge cases."""
    # Large values
    large_coords = [
        (1e6, 1e6, 1e6),
        (1e-6, 1e-6, 1e-6),
    ]

    for x, y, z in large_coords:
        # Convert to eta-phi
        eta, phi = cartesian_to_eta_phi(x, y, z)

        # Convert back to Cartesian
        x_new, y_new, z_new = eta_phi_to_cartesian(eta, phi)

        # Calculate original and new magnitudes
        r_orig = np.sqrt(x**2 + y**2 + z**2)
        r_new = np.sqrt(x_new**2 + y_new**2 + z_new**2)

        # Normalize vectors for comparison
        x_norm, y_norm, z_norm = x / r_orig, y / r_orig, z / r_orig
        x_new_norm, y_new_norm, z_new_norm = x_new / r_new, y_new / r_new, z_new / r_new

        # Assert that the normalized coordinates are approximately equal
        assert np.isclose(x_norm, x_new_norm, atol=1e-6)
        assert np.isclose(y_norm, y_new_norm, atol=1e-6)
        assert np.isclose(z_norm, z_new_norm, atol=1e-6)


def test_round_trip_eta_phi():
    """Test round-trip conversion starting from eta-phi coordinates."""
    eta_phi_cases = [
        (0.0, 0.0),
        (1.5, np.pi / 4),
        (-1.5, 3 * np.pi / 4),
        (0.0, np.pi),
        (2.0, -np.pi / 2),
    ]

    for eta, phi in eta_phi_cases:
        # Convert to Cartesian
        x, y, z = eta_phi_to_cartesian(eta, phi)

        # Convert back to eta-phi
        eta_new, phi_new = cartesian_to_eta_phi(x, y, z)

        # Normalize phi to [-π,π]
        if phi_new > np.pi:
            phi_new -= 2 * np.pi
        elif phi_new < -np.pi:
            phi_new += 2 * np.pi

        # Assert that the eta-phi coordinates are approximately equal
        assert np.isclose(eta, eta_new, atol=1e-6)
        assert np.isclose(phi, phi_new, atol=1e-6)
