from __future__ import annotations

import json
import nrrd
from typing import Optional
from dataclasses import dataclass, field
from pathlib import Path
from enum import Enum
import ast

import numpy as np
import pandas as pd

from lib.dicoms.dicom_dataset import DicomDataset
from lib.metadata.metadata import StudyMetadata


EMPTY_SPACING_STRING = "{'voxel_spacing': (None, None), 'slice_thickness': None, 'spacing_between_slices': None}"
EMPTY_SPACING = {
    "voxel_spacing": (None, None),
    "slice_thickness": None,
    "spacing_between_slices": None,
}


class ModalityType(Enum):
    COR_PDF = "cor_pdf"
    COR_T1 = "cor_t1"
    TRA_PDF = "tra_pdf"
    SAG_T2 = "sag_t2"


@dataclass
class Spacing:
    voxel_spacing: tuple[float, float]
    slice_thickness: float
    spacing_between_slices: float = field(default=None)

    def __init__(
        self,
        voxel_spacing: tuple[float, float],
        slice_thickness: float,
        spacing_between_slices: float = None,
    ) -> None:
        self.voxel_spacing = voxel_spacing
        self.slice_thickness = slice_thickness
        self.spacing_between_slices = spacing_between_slices

    def to_dict(self):
        return {
            "voxel_spacing": self.voxel_spacing,
            "slice_thickness": self.slice_thickness,
            "spacing_between_slices": self.spacing_between_slices,
        }


@dataclass
class ShoulderMetadataStruct:
    """This class represents the annotations for one study located in the image/sequence domain
    During this project we switched from patient-domain based annotations to image-domain based.

    """

    study_uid: str = field(default=None)
    patient_id: str = field(default=None)
    manufacturer: str = field(default=None)

    cor_pdf_suid: str = field(default=None)
    cor_pdf_spacing: Optional[Spacing] = field(default=None)
    cor_pdf_slices: Optional[int] = field(default=None)

    cor_t1_suid: str = field(default=None)
    cor_t1_spacing: Optional[Spacing] = field(default=None)
    cor_t1_slices: Optional[int] = field(default=None)

    tra_pdf_suid: str = field(default=None)
    tra_pdf_spacing: Optional[Spacing] = field(default=None)
    tra_pdf_slices: Optional[int] = field(default=None)

    sag_t2_suid: str = field(default=None)
    sag_t2_spacing: Optional[Spacing] = field(default=None)
    sag_t2_slices: Optional[int] = field(default=None)

    def __init__(self):
        self.annotations = []

    def get_series_uid(self, modality: ModalityType) -> str:
        return getattr(self, f"{modality.value}_suid")

    def get_slice_count(self, modality: ModalityType) -> int:
        return getattr(self, f"{modality.value}_slices")

    def get_spacing(self, modality: ModalityType) -> Spacing:
        return getattr(self, f"{modality.value}_spacing")

    @staticmethod
    def load(data: dict, stringified: bool = True) -> ShoulderMetadataStruct:
        try:
            item = ShoulderMetadataStruct()
            item.study_uid = data["study_uid"]
            item.patient_id = data["patient_id"]
            item.manufacturer = data["manufacturer"]

            item.cor_pdf_suid = data[f"{ModalityType.COR_PDF.value}_suid"]
            item.cor_t1_suid = data[f"{ModalityType.COR_T1.value}_suid"]
            item.tra_pdf_suid = data[f"{ModalityType.TRA_PDF.value}_suid"]
            item.sag_t2_suid = data[f"{ModalityType.SAG_T2.value}_suid"]

            item.cor_pdf_slices = int(
                data.get(f"{ModalityType.COR_PDF.value}_slices", 0)
            )
            item.cor_t1_slices = int(data.get(f"{ModalityType.COR_T1.value}_slices", 0))
            item.tra_pdf_slices = int(
                data.get(f"{ModalityType.TRA_PDF.value}_slices", 0)
            )
            item.sag_t2_slices = int(data.get(f"{ModalityType.SAG_T2.value}_slices", 0))

            if stringified:
                item.cor_pdf_spacing = Spacing(
                    **ast.literal_eval(
                        data.get(
                            f"{ModalityType.COR_PDF.value}_spacing",
                            EMPTY_SPACING_STRING,
                        )
                    )
                )
                item.cor_t1_spacing = Spacing(
                    **ast.literal_eval(
                        data.get(
                            f"{ModalityType.COR_T1.value}_spacing", EMPTY_SPACING_STRING
                        )
                    )
                )
                item.tra_pdf_spacing = Spacing(
                    **ast.literal_eval(
                        data.get(
                            f"{ModalityType.TRA_PDF.value}_spacing",
                            EMPTY_SPACING_STRING,
                        )
                    )
                )
                item.sag_t2_spacing = Spacing(
                    **ast.literal_eval(
                        data.get(
                            f"{ModalityType.SAG_T2.value}_spacing", EMPTY_SPACING_STRING
                        )
                    )
                )

            else:
                item.cor_pdf_spacing = Spacing(
                    **data.get(f"{ModalityType.COR_PDF.value}_spacing", EMPTY_SPACING)
                )
                item.cor_t1_spacing = Spacing(
                    **data.get(f"{ModalityType.COR_T1.value}_spacing", EMPTY_SPACING)
                )
                item.tra_pdf_spacing = Spacing(
                    **data.get(f"{ModalityType.TRA_PDF.value}_spacing", EMPTY_SPACING)
                )
                item.sag_t2_spacing = Spacing(
                    **data.get(f"{ModalityType.SAG_T2.value}_spacing", EMPTY_SPACING)
                )

            return item

        except KeyError as e:
            print(f"Error reading candidate: {e}")
            return

    @staticmethod
    def read_list_from_csv(path: Path, *args, **kwargs) -> list[ShoulderMetadataStruct]:
        df = pd.read_csv(
            path,
            na_values=None,
            keep_default_na=False,
            dtype=object,
            *args,
            **kwargs,
        )

        studies = []

        for _, study in df.iterrows():
            studies.append(ShoulderMetadataStruct.load(study))

        return studies


@dataclass
class Segmentation:
    pixels: np.ndarray
    orientation: tuple[float, float, float, float, float, float]


@dataclass
class ReferencePoints:
    lat_acr_border: tuple[float, float, float]
    lat_clav_border: tuple[float, float, float]
    apical_humerus: tuple[float, float, float]
    lat_glen_sup: tuple[float, float, float]
    lat_glen_inf: tuple[float, float, float]
    proc_cora_tip: tuple[float, float, float]

    @staticmethod
    def load_from_json_obj(json_obj: dict) -> ReferencePoints:
        # Extract control points from the JSON structure
        markups = json_obj.get("markups", [])
        if not markups:
            raise ValueError("No markups found in JSON object")

        control_points = markups[0].get("controlPoints", [])
        if not control_points:
            raise ValueError("No control points found in JSON object")

        # Create a dictionary mapping labels to positions
        points_dict = {}
        for point in control_points:
            label = point.get("label")
            position = point.get("position")
            if label and position:
                points_dict[label] = tuple(position)

        # Required labels
        required_labels = [
            "lat_acr_border",
            "lat_clav_border",
            "apical_humerus",
            "lat_glen_sup",
            "lat_glen_inf",
            "proc_cora_tip",
        ]

        # Check if all required control points exist
        missing_labels = [
            label for label in required_labels if label not in points_dict
        ]
        if missing_labels:
            raise ValueError(
                f"Missing required control points: {', '.join(missing_labels)}"
            )

        # Create and return ReferencePoints instance
        return ReferencePoints(
            lat_acr_border=points_dict["lat_acr_border"],
            lat_clav_border=points_dict["lat_clav_border"],
            apical_humerus=points_dict["apical_humerus"],
            lat_glen_sup=points_dict["lat_glen_sup"],
            lat_glen_inf=points_dict["lat_glen_inf"],
            proc_cora_tip=points_dict["proc_cora_tip"],
        )


@dataclass
class Study:
    cor_t1_reference_points: ReferencePoints
    sag_t2_reference_points: ReferencePoints
    tra_pdf_reference_points: ReferencePoints

    cor_pdf_images_path: Path
    cor_t1_images_path: Path
    sag_t2_images_path: Path
    tra_pdf_images_path: Path

    cor_pdf_images: Optional[DicomDataset]
    cor_t2_images: Optional[DicomDataset]
    sag_t2_images: Optional[DicomDataset]
    tra_pdf_images: Optional[DicomDataset]

    segmentation: Segmentation

    def read_image(self, mod: ModalityType) -> DicomDataset:
        """Read and cache DICOM images for the specified modality.

        Args:
            mod: The modality type to read (cor_pdf, cor_t1, sag_t2, tra_pdf)

        Returns:
            DicomDataset: The loaded DICOM dataset

        Raises:
            ValueError: If the modality path is not set or doesn't exist
        """
        # Map modality to corresponding attributes
        modality_map = {
            ModalityType.COR_PDF: ("cor_pdf_images", "cor_pdf_images_path"),
            ModalityType.COR_T1: ("cor_t2_images", "cor_t1_images_path"),
            ModalityType.SAG_T2: ("sag_t2_images", "sag_t2_images_path"),
            ModalityType.TRA_PDF: ("tra_pdf_images", "tra_pdf_images_path"),
        }

        if mod not in modality_map:
            raise ValueError(f"Unknown modality: {mod}")

        cache_attr, path_attr = modality_map[mod]

        # Check if already cached
        cached_dataset = getattr(self, cache_attr)
        if cached_dataset is not None:
            return cached_dataset

        # Get the path
        path = getattr(self, path_attr)
        if path is None:
            raise ValueError(f"Path for modality {mod} is not set")
        if not path.exists():
            raise ValueError(f"Path for modality {mod} does not exist: {path}")

        # Load the dataset
        dataset = DicomDataset.read_from_files(path)

        # Cache it
        setattr(self, cache_attr, dataset)

        return dataset

    @staticmethod
    def load_from_folder(folder_path: Path, metadata: ShoulderMetadataStruct) -> Study:
        # Load reference points from JSON files
        with open(folder_path / "cort1_annotations.json", "r") as f:
            cor_t1_reference_points = ReferencePoints.load_from_json_obj(json.load(f))

        with open(folder_path / "sagt2_annotations.json", "r") as f:
            sag_t2_reference_points = ReferencePoints.load_from_json_obj(json.load(f))

        with open(folder_path / "trapdf_annotations.json", "r") as f:
            tra_pdf_reference_points = ReferencePoints.load_from_json_obj(json.load(f))

        # Find modality folders using series UIDs from metadata
        cor_pdf_path = (
            folder_path / metadata.cor_pdf_suid if metadata.cor_pdf_suid else None
        )
        cor_t1_path = (
            folder_path / metadata.cor_t1_suid if metadata.cor_t1_suid else None
        )
        sag_t2_path = (
            folder_path / metadata.sag_t2_suid if metadata.sag_t2_suid else None
        )
        tra_pdf_path = (
            folder_path / metadata.tra_pdf_suid if metadata.tra_pdf_suid else None
        )

        # Verify that the paths exist
        if cor_pdf_path and not cor_pdf_path.exists():
            raise ValueError(f"cor_pdf series folder not found: {cor_pdf_path}")
        if cor_t1_path and not cor_t1_path.exists():
            raise ValueError(f"cor_t1 series folder not found: {cor_t1_path}")
        if sag_t2_path and not sag_t2_path.exists():
            raise ValueError(f"sag_t2 series folder not found: {sag_t2_path}")
        if tra_pdf_path and not tra_pdf_path.exists():
            raise ValueError(f"tra_pdf series folder not found: {tra_pdf_path}")

        # Load segmentation
        seg_files = list(folder_path.glob("seg*.nrrd"))
        if not seg_files:
            raise ValueError(f"No segmentation file found in {folder_path}")
        segmentation_pixels, header = nrrd.read(str(seg_files[0]))

        # Extract orientation from NRRD header (space directions)
        space_directions = header.get("space directions", None)
        if space_directions is None:
            raise ValueError(
                f"No 'space directions' found in NRRD header for {seg_files[0]}"
            )

        # Flatten the first two direction vectors into a 6-tuple
        orientation = tuple(space_directions[0][:3]) + tuple(space_directions[1][:3])

        segmentation = Segmentation(
            pixels=segmentation_pixels.transpose(
                1, 0, 2
            ),  # we have to change the orientation here
            orientation=orientation,
        )

        return Study(
            cor_t1_reference_points=cor_t1_reference_points,
            sag_t2_reference_points=sag_t2_reference_points,
            tra_pdf_reference_points=tra_pdf_reference_points,
            cor_pdf_images_path=cor_pdf_path,
            cor_t1_images_path=cor_t1_path,
            sag_t2_images_path=sag_t2_path,
            tra_pdf_images_path=tra_pdf_path,
            cor_pdf_images=None,
            cor_t2_images=None,
            sag_t2_images=None,
            tra_pdf_images=None,
            segmentation=segmentation,
        )


if __name__ == "__main__":
    metadata = ShoulderMetadataStruct.read_list_from_csv(Path("shoulders/small.csv"))
    metadata_map = {s.study_uid: s for s in metadata}

    study = Study.load_from_folder(
        Path(
            "/Users/admin/Projekte/Schultern/Segmentierungen/Manual/1.2.840.113619.6.95.31.0.3.4.1.3096.13.214767"
        ),
        metadata=metadata_map["1.2.840.113619.6.95.31.0.3.4.1.3096.13.214767"],
    )

    dcm = study.read_image(ModalityType.COR_PDF)
    print("Done")
