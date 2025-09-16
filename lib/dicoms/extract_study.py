"""This is not usable at the moment since it takes years to scroll through large tar files"""

from typing import Optional
from pathlib import Path
import zipfile 

from tqdm import tqdm
import numpy as np
import pydicom

from lib.dicom.dicom_dataset import DicomDataset
from lib.metadata import StudyMetadata

def extract_study(study_map: StudyMetadata, export_path: Path, series_to_export: Optional[list[str]] = None) -> None:
    with zipfile.ZipFile(study_map["file_path"], "r") as zip_file:
        study_path = export_path / study_map["study_instance_uid"]
        study_path.mkdir(exist_ok=True, parents=True)


        for series in study_map["series"].values():
            if series_to_export and series["series_instance_uid"] not in series_to_export: continue

            series_path = study_path / series["series_instance_uid"]
            series_path.mkdir(exist_ok=True, parents=True)

            for instance in series["instances"]:
                with zip_file.open(instance["file_path"]) as f: 
                    instance_path = series_path / instance["instance_uid"]
                    with open(instance_path, "wb") as output_file:
                        output_file.write(f.read())


def get_study(study_map: StudyMetadata, series_to_export: Optional[list[str]] = None, stop_before_pixels: bool = True, force:bool = False) -> list[list[pydicom.Dataset]]:
    """Read a study"""
    study_files = []

    with zipfile.ZipFile(study_map["file_path"], "r") as zip_file:
        for series in study_map["series"].values():
            if series_to_export and series["series_instance_uid"] not in series_to_export: continue

            series_files = []

            for instance in series["instances"]:
                with zip_file.open(instance["file_path"]) as f:
                    series_files.append(pydicom.dcmread(f, stop_before_pixels=stop_before_pixels))

            
            study_files.append(
                DicomDataset(dcm_list=series_files, force=force)
            )

    return study_files

                    

if __name__ == "__main__":
    export_path = Path("/home/homesOnMaster/pehrlich/dataset_preparation/.export/samples/huefte/studies")
    dicom_path = Path("/mnt/ocean_storage/data/koeln/huefte")
    study_metadata: dict[str, StudyMetadata] = np.load(
        "/home/homesOnMaster/pehrlich/dataset_preparation/huefte/study_map_huefte.npy",
        allow_pickle=True
    ).tolist()

    study_uids = ['1.2.840.113619.6.95.31.0.3.4.1.3096.13.280092', '1.2.840.113619.6.95.31.0.3.4.1.3096.13.287084', '1.2.840.113619.6.95.31.0.3.4.1.3096.13.357542', '1.2.840.113619.6.95.31.0.3.4.1.3096.13.280580', '1.2.840.113619.6.95.31.0.3.4.1.3096.13.291010', '1.2.840.113619.6.95.31.0.3.4.1.3096.13.269437', '1.2.840.113619.6.95.31.0.3.4.1.3096.13.275851', '1.2.840.113619.6.95.31.0.3.4.1.3096.13.298013', '1.2.840.113619.6.95.31.0.3.4.1.3096.13.282135', '1.2.840.113619.6.95.31.0.3.4.1.3096.13.360112', '1.2.840.113619.6.95.31.0.3.4.1.3096.13.357608', '1.2.840.113619.6.95.31.0.3.4.1.3096.13.294863']

    # for study_uid, series_uids in tqdm(zip(study_uids, series)):
    #     extract_study(study_map=study_metadata[study_uid], export_path=export_path, series_to_export=series_uids) 
    
    for study_uid in tqdm(study_uids):
        extract_study(study_map=study_metadata[study_uid], export_path=export_path) 