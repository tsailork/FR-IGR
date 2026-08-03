"""
High-performance VTK dataset loader and spatial interpolation tools for 2D CFD datasets.
"""

import os
import glob
import re
import numpy as np
from scipy.interpolate import griddata
from vtkmodules.vtkIOXML import vtkXMLUnstructuredGridReader, vtkXMLStructuredGridReader
from vtkmodules.util.numpy_support import vtk_to_numpy


def load_vtk_dataset(file_path):
    """
    Reads a .vtu or .vts XML file and extracts all node coordinates and PointData fields as NumPy arrays.

    Parameters:
        file_path (str): Path to the target VTK file (.vtu or .vts).

    Returns:
        coords (ndarray): Shape (N, 2) array of (x, y) nodal coordinates.
        fields (dict): Dictionary mapping field names (e.g. 'rho', 'u', 'v', 'Pressure', 'Sigma') 
                       to 1D NumPy arrays of length N.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"VTK dataset file not found: {file_path}")

    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".vtu":
        reader = vtkXMLUnstructuredGridReader()
    elif ext == ".vts":
        reader = vtkXMLStructuredGridReader()
    else:
        raise ValueError(f"Unsupported VTK file extension '{ext}'. Expected .vtu or .vts.")

    reader.SetFileName(file_path)
    reader.Update()
    output = reader.GetOutput()

    # Extract nodal coordinates (x, y)
    vtk_points = output.GetPoints()
    if vtk_points is None:
        raise ValueError(f"No point coordinates found in VTK file: {file_path}")

    points_arr = vtk_to_numpy(vtk_points.GetData())
    coords = points_arr[:, :2] # (N, 2) array for 2D

    # Extract PointData scalar and vector fields
    point_data = output.GetPointData()
    n_arrays = point_data.GetNumberOfArrays()

    fields = {}
    for i in range(n_arrays):
        array_vtk = point_data.GetArray(i)
        if array_vtk is not None:
            name = array_vtk.GetName()
            arr_np = vtk_to_numpy(array_vtk)
            fields[name] = arr_np

    return coords, fields


def resample_to_uniform_grid(coords, values, resolution=(256, 256), method='cubic', fill_value=0.0):
    """
    Resamples un-structured or non-uniform solution point data onto a regular 2D Cartesian grid.

    Parameters:
        coords (ndarray): (N, 2) array of physical (x, y) coordinates.
        values (ndarray): (N,) array of field scalar values.
        resolution (tuple): (Nx, Ny) grid dimensions for output matrix.
        method (str): Interpolation mode ('linear', 'cubic', or 'nearest').
        fill_value (float): Value for out-of-bounds evaluation.

    Returns:
        X (ndarray): (Ny, Nx) Meshgrid X matrix.
        Y (ndarray): (Ny, Nx) Meshgrid Y matrix.
        grid_values (ndarray): (Ny, Nx) Interpolated scalar field matrix.
    """
    Nx, Ny = resolution
    x_min, x_max = np.min(coords[:, 0]), np.max(coords[:, 0])
    y_min, y_max = np.min(coords[:, 1]), np.max(coords[:, 1])

    x_lin = np.linspace(x_min, x_max, Nx)
    y_lin = np.linspace(y_min, y_max, Ny)
    X, Y = np.meshgrid(x_lin, y_lin)

    # Use scipy griddata for robust multi-dimensional interpolation
    grid_values = griddata(coords, values, (X, Y), method=method, fill_value=fill_value)

    # Fallback to nearest neighbor interpolation if NaNs exist at boundaries
    if np.isnan(grid_values).any():
        nan_mask = np.isnan(grid_values)
        grid_nearest = griddata(coords, values, (X, Y), method='nearest')
        grid_values[nan_mask] = grid_nearest[nan_mask]

    return X, Y, grid_values


def find_latest_vtk_file(pv_outputs_dir, prefix="plot_"):
    """
    Finds the latest numerical dataset file (.vtu) matching a specific prefix inside pv_outputs/.

    Parameters:
        pv_outputs_dir (str): Directory containing VTK plot outputs.
        prefix (str): File prefix pattern ('plot_' or 'sol_').

    Returns:
        latest_file (str): Full path to the latest file sorted by step index.
    """
    pattern = os.path.join(pv_outputs_dir, f"{prefix}*.vtu")
    files = glob.glob(pattern)
    if not files:
        # Fallback search for any .vtu file
        files = glob.glob(os.path.join(pv_outputs_dir, "*.vtu"))

    if not files:
        raise FileNotFoundError(f"No VTK dataset files found in: {pv_outputs_dir}")

    def extract_index(filepath):
        numbers = re.findall(r'\d+', os.path.basename(filepath))
        return int(numbers[-1]) if numbers else 0

    files.sort(key=extract_index)
    return files[-1]
