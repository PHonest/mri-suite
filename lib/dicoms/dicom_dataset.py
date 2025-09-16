from __future__ import annotations

from io import BytesIO
from typing import List, Tuple
from pathlib import Path

from pydicom import Dataset, dcmread, dcmwrite
from pydicom.filebase import DicomFileLike
import numpy as np


def _check_types(dcm_list: List[Dataset]):
    if not isinstance(dcm_list, List):
        raise TypeError("DicomDataset only support list datatype")

    if len(dcm_list) == 0:
        raise ValueError("Passed empty list for DicomDataset")

    for _file in dcm_list:
        if not isinstance(_file, Dataset):
            raise TypeError(
                "DicomDataset only support pydicom::Dataset as list elements"
            )

    if "ImagePositionPatient" not in dcm_list[0]:
        raise ValueError("The dicom dataset does not contain ImagePositionPatient")

    if any([_dcm.Modality == "SEG" for _dcm in dcm_list]):
        raise TypeError(
            "Found Segmentation in dicom dataset. Please use DicomSegmentation instead"
        )

    if not (
        all(
            [
                _dcm.SeriesInstanceUID == dcm_list[0].SeriesInstanceUID
                for _dcm in dcm_list
            ]
        )
    ):
        raise ValueError(
            "The instances of the datasets do not match in SeriesInstanceUID"
        )
    if not (
        all(
            [_dcm.StudyInstanceUID == dcm_list[0].StudyInstanceUID for _dcm in dcm_list]
        )
    ):
        raise ValueError(
            "The instances of the datasets do not match in StudyInstanceUID"
        )
    if not (
        all(
            [
                _dcm.pixel_array.shape == dcm_list[0].pixel_array.shape
                for _dcm in dcm_list
            ]
        )
    ):
        raise ValueError(
            "The instances of the datasets do not match in pixel_array shape"
        )


def _sort_dicom_series(series: List[Dataset]) -> List[Dataset]:
    """
    Sorts a list of DICOM datasets according to the order they should be viewed.

    Parameters:
    - series: List[pydicom.Dataset]
        A list of DICOM datasets representing the image slices.

    Returns:
    - sorted_series: List[pydicom.Dataset]
        The sorted list of DICOM datasets.
    """
    # Check if the series is empty
    if not series:
        return series

    # Extract ImageOrientationPatient and ImagePositionPatient from the first slice
    ds0 = series[0]
    try:
        iop = ds0.ImageOrientationPatient  # [r1, r2, r3, c1, c2, c3]
        ipp = ds0.ImagePositionPatient  # [x0, y0, z0]
    except AttributeError as e:
        raise ValueError(f"Missing required DICOM tags: {e}")

    # Convert to numpy arrays
    iop = np.array(iop, dtype=np.float64)
    ipp = np.array(ipp, dtype=np.float64)

    # Get the direction cosines
    row_cosines = iop[:3]
    col_cosines = iop[3:]
    # Compute the normal vector (slice normal)
    slice_normal = np.cross(row_cosines, col_cosines)

    # For each dataset, compute the position along the slice normal
    positions = []
    for idx, ds in enumerate(series):
        # Ensure that ImagePositionPatient is present
        if not hasattr(ds, "ImagePositionPatient"):
            raise ValueError(f"Dataset at index {idx} is missing ImagePositionPatient")

        # Get ImagePositionPatient
        ds_ipp = np.array(ds.ImagePositionPatient, dtype=np.float64)
        # Compute the distance along the slice normal
        position = np.dot(ds_ipp - ipp, slice_normal)
        positions.append(position)

    # Create a list of tuples (position, dataset)
    position_dataset_tuples = list(zip(positions, series))

    # Sort the list based on the positions
    sorted_tuples = sorted(position_dataset_tuples, key=lambda x: x[0])

    # Extract the sorted datasets
    sorted_series = [ds for _, ds in sorted_tuples]

    return sorted_series


def _is_sorted(series: List[Dataset], tol: float = 1e-5) -> bool:
    """
    Checks if a list of DICOM datasets is sorted according to the viewing order.

    Parameters:
    - series: List[pydicom.Dataset]
        A list of DICOM datasets representing the image slices.
    - tol: float
        A tolerance value for floating-point comparisons (default: 1e-5).

    Returns:
    - is_sorted: bool
        True if the series is sorted in viewing order, False otherwise.
    """
    # Check if the series is empty or has only one element
    if len(series) <= 1:
        return True

    # Extract ImageOrientationPatient and ImagePositionPatient from the first slice
    ds0 = series[0]
    try:
        iop = ds0.ImageOrientationPatient  # [r1, r2, r3, c1, c2, c3]
        ipp = ds0.ImagePositionPatient  # [x0, y0, z0]
    except AttributeError as e:
        raise ValueError(f"Missing required DICOM tags: {e}")

    # Convert to numpy arrays
    iop = np.array(iop, dtype=np.float64)
    ipp = np.array(ipp, dtype=np.float64)

    # Get the direction cosines
    row_cosines = iop[:3]
    col_cosines = iop[3:]
    # Compute the normal vector (slice normal)
    slice_normal = np.cross(row_cosines, col_cosines)

    # For each dataset, compute the position along the slice normal
    positions = []
    for idx, ds in enumerate(series):
        # Ensure that ImagePositionPatient is present
        if not hasattr(ds, "ImagePositionPatient"):
            raise ValueError(f"Dataset at index {idx} is missing ImagePositionPatient")

        # Get ImagePositionPatient
        ds_ipp = np.array(ds.ImagePositionPatient, dtype=np.float64)
        # Compute the distance along the slice normal
        position = np.dot(ds_ipp - ipp, slice_normal)
        positions.append(position)

    # Check if positions are monotonically increasing or decreasing with tolerance
    is_increasing = all((y - x) >= -tol for x, y in zip(positions, positions[1:]))
    is_decreasing = all((x - y) >= -tol for x, y in zip(positions, positions[1:]))

    is_sorted = is_increasing

    return is_sorted


class DicomDataset:
    """This class acts as a wrapper for the pydicom:Dataset

    It is build on the idea to add functionionality on top of the pydicom:Dataset to work with multiple dicom files, are cloesly related (e.g. same series)

    Args:
        force (bool, optional): If true, the internal consistency checks will be overwritten
    """

    def __init__(self, dcm_list: List[Dataset], force: bool = False):
        if force:
            self.dataset = dcm_list
            return

        _check_types(dcm_list)
        self.dataset: List[Dataset] = dcm_list

        if not self.is_sorted:
            self.sort()

    @property
    def shape(self) -> Tuple:
        """Returns the shape of the pixel_array (the first dimesion will be the number of datasets)"""
        return self.get_pixel_array().shape

    @property
    def is_sorted(self) -> bool:
        """Checks whether a the interal dataset is sorted with respect to the z-coordinate
        of the tag ImagePositionPatient"""

        return _is_sorted(self.dataset)

    @property
    def patient_id(self) -> str:
        """Returns the study instance uid"""
        return self.dataset[0].PatientID

    @property
    def study_uid(self) -> str:
        """Returns the study instance uid"""
        return self.dataset[0].StudyInstanceUID

    @property
    def series_uid(self) -> str:
        """Returns the series instance uid"""
        return self.dataset[0].SeriesInstanceUID

    @property
    def image_position_patient(self) -> str:
        """Returns the image position patient for the first slice

        x,y,z of 0,0,0 is the top left corner of the first slice in the series
        """
        return self.dataset[0].ImagePositionPatient

    @property
    def image_orientation_patient(self) -> str:
        """Returns the series instance uid"""
        return self.dataset[0].ImageOrientationPatient

    @property
    def slice_spacing(self) -> str:
        """Returns the series instance uid"""
        return self.dataset[0].SpacingBetweenSlices

    @property
    def slice_thickness(self) -> str:
        """Returns the series instance uid"""
        return self.dataset[0].SliceThickness

    @property
    def pixel_spacing(self) -> str:
        """Returns the series instance uid"""
        return self.dataset[0].PixelSpacing

    def infer_slice_normal(self) -> np.ndarray:
        """Infers the slice normal from the ImageOrientationPatient tag

        Uses ImagePositionPatient from first and last slice to determine
        the correct direction of the slice normal vector.
        """
        if not hasattr(self.dataset[0], "ImageOrientationPatient"):
            raise ValueError("ImageOrientationPatient tag is missing in the dataset")

        if len(self.dataset) < 2:
            # If only one slice, return the cross product without direction correction
            iop = np.array(self.dataset[0].ImageOrientationPatient, dtype=np.float64)
            row_cosines = iop[:3]
            col_cosines = iop[3:]
            slice_normal = np.cross(row_cosines, col_cosines)
            return slice_normal / np.linalg.norm(slice_normal)

        # Get orientation from first slice
        iop = np.array(self.dataset[0].ImageOrientationPatient, dtype=np.float64)
        row_cosines = iop[:3]
        col_cosines = iop[3:]

        # Compute cross product (two possible directions)
        slice_normal = np.cross(row_cosines, col_cosines)
        slice_normal = slice_normal / np.linalg.norm(slice_normal)

        # Get positions of first and last slice to determine correct direction
        first_position = np.array(
            self.dataset[0].ImagePositionPatient, dtype=np.float64
        )
        last_position = np.array(
            self.dataset[-1].ImagePositionPatient, dtype=np.float64
        )

        # Vector from first to last slice
        slice_direction = last_position - first_position
        slice_direction = slice_direction / np.linalg.norm(slice_direction)

        # Check if computed normal aligns with actual slice progression
        # If dot product is negative, we need to flip the normal
        if np.dot(slice_normal, slice_direction) < 0:
            slice_normal = -slice_normal

        return slice_normal

    def sort(self) -> None:
        """Sorts the internal dataset with respect to the z-coordinate
        of the tag ImagePositionPatient"""

        self.dataset = _sort_dicom_series(self.dataset)

    def get_pixel_array(self, axis: int = 0) -> np.ndarray:
        return np.stack([_file.pixel_array for _file in self.dataset], axis=axis)

    def set_pixel_array(self, pixel_array: np.ndarray) -> None:
        """Sets the pixel_array of the internal dataset"""
        assert (
            pixel_array.shape == self.shape
        ), "The shape of the pixel_array does not match the shape of the internal dataset"

        for _slice in range(pixel_array.shape[0]):
            self.dataset[_slice].PixelData = pixel_array[_slice].tobytes()

    def as_bytes(self) -> List[bytes]:
        """Interprets the dicom dataset as sitk List of bytes"""
        dataset_bytes = []

        for _ds in self.dataset:
            with BytesIO() as buffer:
                memory_dataset = DicomFileLike(buffer)
                dcmwrite(memory_dataset, _ds)
                memory_dataset.seek(0)
                dataset_bytes.append(memory_dataset.read())

        return dataset_bytes

    @staticmethod
    def read_from_files(path: Path) -> DicomDataset:
        """Returns a DicomDataset by reading dcm files from disk"""
        if not path.is_dir():
            raise FileNotFoundError("The given file is not a directory")

        dataset = [dcmread(_file) for _file in path.iterdir()]
        return DicomDataset(dataset)

    def display_dicom_slice(self, slice: int) -> None:
        """
        Displays the slice at the specified index from a list of pydicom.Dataset objects.

        Parameters:
            slice (int): Index of the slice to display.
        """
        import matplotlib.pyplot as plt

        if slice < 0 or slice >= len(self.dataset):
            print(
                f"Index {slice} is out of range. The series contains {len(self.dataset)} slices."
            )
            return

        # Get the dataset at the specified index
        ds = self.dataset[slice]

        # Ensure the dataset has pixel data
        if not hasattr(ds, "pixel_array"):
            print(f"The DICOM dataset at index {slice} does not contain pixel data.")
            return

        # Extract pixel array and display it
        pixel_array = ds.pixel_array
        plt.figure(figsize=(6, 6))
        plt.imshow(pixel_array, cmap="gray")
        plt.title(f"Slice {slice} - {str(self.shape)}")
        plt.axis("off")
        plt.show()

        return
