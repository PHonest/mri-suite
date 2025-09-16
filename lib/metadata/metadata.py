from typing import TypedDict, Optional

class Instance(TypedDict):
    file_path: str
    instance_uid: str

class Series(TypedDict):
    series_instance_uid: str
    series_folder_name: str
    series_description: str
    instances: list[Instance]
    modality: Optional[str]
    orientation: Optional[tuple[float,float,float,float,float,float]]

class StudyMetadata(TypedDict): 
    file_path: str 
    study_date: str
    study_description: str
    institution_name: str
    laterality: str
    magnetic_field_strength: str
    manufacturer: str
    manufacturer_model_name: str
    patient_age: str
    patient_id: str
    patient_sex: str
    patient_weight: str
    study_instance_uid: str
    series: dict[str, Series]
    with_km: bool
    
