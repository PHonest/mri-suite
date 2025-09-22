import re
import zipfile 
from pathlib import Path

import numpy as np
import pydicom

from lib.metadata import StudyMetadata

def extract_text_from_dicom_report(report: pydicom.Dataset) -> str:
    if "ContentSequence" in report:
        def extract_text(seq):
            texts = []
            for item in seq:
                if "TextValue" in item:
                    texts.append(item.TextValue)
                if "ContentSequence" in item:
                    texts.extend(extract_text(item.ContentSequence))
            return texts
        all_text = extract_text(report.ContentSequence)
        return "\n".join(all_text)
    else:
        print("No text or PDF found in this DICOM report.")

def extract_report(study_metadata: StudyMetadata) -> None:
    save_path = Path("/home/homesOnMaster/pehrlich/dataset_preparation/.export/osg/reports")

    pattern = re.compile(r"befund", re.IGNORECASE)

    report_series = list(filter(lambda x: pattern.search(x["series_description"]), study_metadata["series"].values()))

    # return gracefully if not report was found 
    if not report_series: return

    report_series = report_series[0]

    with zipfile.ZipFile(study_metadata["file_path"], "r") as zip_file:
        for instance in report_series["instances"]:
            with zip_file.open(instance["file_path"]) as f: 
                report = pydicom.dcmread(f)
                report_text = extract_text_from_dicom_report(report)

                report_path = save_path / f"{study_metadata["study_instance_uid"]}.txt"

                with open(report_path, "w", encoding="utf-8") as f:
                    f.write(report_text)


                return report_path


if __name__ == "__main__":
    import json
    from tqdm import tqdm

    study_metadata: dict[str, StudyMetadata] = np.load(
        "/home/homesOnMaster/pehrlich/dataset_preparation/osg/study_map_osg.npy",
        allow_pickle=True
    ).tolist()


    paths: list[Path] = []

    for study in tqdm(study_metadata.values()):
        path = extract_report(study_metadata=study)

        if path: 
            paths.append(path)


    mapping = [{"filename": p.name, "externalID": p.name.split(".txt")[0]} for p in paths]

    with open("mapping.json", "w", encoding="utf-8") as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2)

        
