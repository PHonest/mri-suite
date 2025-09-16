from pathlib import Path
import numpy as np

from lib.metadata import build_study_dict_concurrent

IGNORE = [
    "Huefte_2022",
    "Huefte_2023",
    "Huefte_2024",
    "Huefte_2025",
]

if __name__ == "__main__":
    dicom_paths = [
        Path("/mnt/ocean_storage/data/koeln/huefte/Huefte_2022.tar.gz"),
        Path("/mnt/ocean_storage/data/koeln/huefte/Huefte_2023.tar.gz"),
        Path("/mnt/ocean_storage/data/koeln/huefte/Huefte_2024.tar.gz"),
        Path("/mnt/ocean_storage/data/koeln/huefte/Huefte_2025.tar.gz"),
    ]

    study_map = build_study_dict_concurrent(paths=dicom_paths, ignore=IGNORE, debug=False)
    
    # Save the study map to a .npy file
    output_path = Path("study_map_huefte.npy")
    np.save(output_path, study_map)
    print(f"Study map saved to {output_path}")
    print(f"Total studies found: {len(study_map)}")