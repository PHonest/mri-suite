from pathlib import Path
import numpy as np

from lib.metadata import build_study_dict_concurrent

IGNORE = [
    "Hand_2021",
    "Hand_2022",
    "Hand_2023",
    "Hand_2024",
    "Hand_2025",
]

if __name__ == "__main__":
    dicom_paths = [
        Path("/mnt/ocean_storage/data/koeln/handgelenk/Hand_2021.zip"),
        Path("/mnt/ocean_storage/data/koeln/handgelenk/Hand_2022.zip"),
        Path("/mnt/ocean_storage/data/koeln/handgelenk/Hand_2023.zip"),
        Path("/mnt/ocean_storage/data/koeln/handgelenk/Hand_2024.zip"),
        Path("/mnt/ocean_storage/data/koeln/handgelenk/Hand_2025.zip"),
    ]

    study_map = build_study_dict_concurrent(paths=dicom_paths, ignore=IGNORE, debug=False)
    
    # Save the study map to a .npy file
    output_path = Path("study_map_hands.npy")
    np.save(output_path, study_map)
    print(f"Study map saved to {output_path}")
    print(f"Total studies found: {len(study_map)}")