import re
import tarfile
from pathlib import Path
from io import BytesIO
import concurrent.futures
import threading

from tqdm import tqdm
from pydicom import dcmread, Dataset

from lib.metadata import StudyMetadata, Series, Instance


def _infer_if_km(dataset: Dataset) -> bool:
    return bool(re.search(r"(?i).*KM.*", dataset.get("SeriesDescription")))

def build_study_map(dicom_path: Path, ignore: list[str], debug: bool):
    study_map: dict[str, StudyMetadata] = dict()

    with tarfile.open(dicom_path, "r:gz") as outer_tar:
        for file_id, member in tqdm(enumerate(outer_tar)):
            if debug and file_id > 100:
                print("Specified debug mode - only read 100 files")
                break

            if member.name in ignore: continue

            file_data = BytesIO(outer_tar.extractfile(member).read())
            dataset = dcmread(file_data, stop_before_pixels=True)

            study_uid = dataset.StudyInstanceUID
            series_uid = dataset.SeriesInstanceUID

            # create study object
            if study_uid not in study_map.keys(): 
                study_map[study_uid] = {
                    "file_path": str(dicom_path),
                    "study_date": dataset.get("StudyDate"),
                    "study_description": dataset.get("StudyDescription"),
                    "institution_name": dataset.get("InstitutionName"),
                    "laterality": dataset.get("Laterality"),
                    "magnetic_field_strength": dataset.get("MagneticFieldStrength"),
                    "manufacturer": dataset.get("Manufacturer"),
                    "manufacturer_model_name": dataset.get("ManufacturerModelName"),
                    "patient_age": dataset.get("PatientAge"),
                    "patient_id": dataset.get("PatientID"),
                    "patient_sex": dataset.get("PatientSex"),
                    "patient_weight": dataset.get("PatientWeight"),
                    "study_instance_uid": dataset.get("StudyInstanceUID"),
                    "with_km": _infer_if_km(dataset),
                    "series": dict(),
                }
            else:
                study_map[study_uid]["study_date"] = study_map[study_uid]["study_date"] or dataset.get("StudyDate")
                study_map[study_uid]["study_description"] = study_map[study_uid]["study_description"] or dataset.get("StudyDescription")
                study_map[study_uid]["institution_name"] = study_map[study_uid]["institution_name"] or dataset.get("InstitutionName")
                study_map[study_uid]["laterality"] = study_map[study_uid]["laterality"] or dataset.get("Laterality")
                study_map[study_uid]["magnetic_field_strength"] = study_map[study_uid]["magnetic_field_strength"] or dataset.get("MagneticFieldStrength")
                study_map[study_uid]["manufacturer"] = study_map[study_uid]["manufacturer"] or dataset.get("Manufacturer")
                study_map[study_uid]["manufacturer_model_name"] = study_map[study_uid]["manufacturer_model_name"] or dataset.get("ManufacturerModelName")
                study_map[study_uid]["patient_age"] = study_map[study_uid]["patient_age"] or dataset.get("PatientAge")
                study_map[study_uid]["patient_id"] = study_map[study_uid]["patient_id"] or dataset.get("PatientID")
                study_map[study_uid]["patient_sex"] = study_map[study_uid]["patient_sex"] or dataset.get("PatientSex")
                study_map[study_uid]["patient_weight"] = study_map[study_uid]["patient_weight"] or dataset.get("PatientWeight")
                study_map[study_uid]["study_instance_uid"] = study_map[study_uid]["study_instance_uid"] or dataset.get("StudyInstanceUID")
                study_map[study_uid]["with_km"] = study_map[study_uid]["with_km"] or _infer_if_km(dataset)

            # create series object
            if series_uid not in study_map[study_uid]["series"].keys(): 
                series: Series = {
                    "series_instance_uid": series_uid,
                    "series_description": dataset.get("SeriesDescription"),
                    "orientation": tuple(dataset.get("ImageOrientationPatient")),
                    "instances": []
                }
                study_map[study_uid]["series"][series_uid] = series
            else:
                study_map[study_uid]["series"][series_uid]["series_instance_uid"] = study_map[study_uid]["series"][series_uid]["series_instance_uid"] or series_uid
                study_map[study_uid]["series"][series_uid]["orientation"] = study_map[study_uid]["series"][series_uid]["orientation"] or tuple(dataset.get("ImageOrientationPatient"))
                study_map[study_uid]["series"][series_uid]["series_description"] = study_map[study_uid]["series"][series_uid]["series_description"] or dataset.get("SeriesDescription")
            


            # create instance object
            instance: Instance = {
                "file_path": member.name,
                "instance_uid": dataset.SOPInstanceUID
            }

            study_map[study_uid]["series"][series_uid]["instances"].append(instance)

    return study_map

def build_study_dict_concurrent(paths: list[Path], ignore: list[str] = [], debug: bool = False) -> dict[str, StudyMetadata]:
    # Thread-safe dictionary to collect results
    study_map: dict[str, StudyMetadata] = {}
    lock = threading.Lock()
    
    def process_single_file(dicom_path: Path):
        """Process a single tar.gz file and return its study map"""
        local_study_map = build_study_map(dicom_path, ignore=ignore, debug=debug)
        
        # Merge results into the shared dictionary thread-safely
        with lock:
            study_map.update(local_study_map)
    
    # Use ThreadPoolExecutor to process files concurrently
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(paths)) as executor:
        # Submit each file to be processed in a separate thread
        futures = [executor.submit(process_single_file, path) for path in paths]
        
        # Wait for all threads to complete
        concurrent.futures.wait(futures)
        
        # Check for any exceptions
        for future in futures:
            try:
                future.result()  # This will raise any exception that occurred
            except Exception as e:
                print(f"Error processing file: {e}")
    
    return study_map
