import numpy as np


from lib.segmentations.structs import Study, ModalityType

from lib.dicoms.dicom_dataset import DicomDataset


def patient_to_image_coordinate(
    patient_coordinate: tuple[float, float, float], series: DicomDataset
) -> tuple[float, float, float]:
    """
    Transforms a coordinate from the patient coordinate system to the image coordinate system.

    Parameters:
    - patient_coordinate: Tuple[float, float, float]
        The (x, y, z) coordinate in the patient coordinate system (in mm).
    - series: DicomDataset

    Returns:
    - image_coordinate: Tuple[float, float, float]
        The (i, j, k) coordinate in the image coordinate system (pixel indices).
    """
    datasets = series.dataset

    # Extract ImageOrientationPatient and PixelSpacing from the first slice
    ds0 = datasets[0]
    image_orientation = ds0.ImageOrientationPatient  # [r1, r2, r3, c1, c2, c3]
    image_position = ds0.ImagePositionPatient  # [x0, y0, z0]
    pixel_spacing = ds0.PixelSpacing  # [row_spacing, col_spacing]

    # Convert lists to numpy arrays
    image_orientation = np.array(image_orientation, dtype=np.float64)
    image_position = np.array(image_position, dtype=np.float64)
    pixel_spacing = np.array(pixel_spacing, dtype=np.float64)

    # Get row and column direction cosines
    row_cosines = image_orientation[0:3]
    col_cosines = image_orientation[3:6]
    # Compute the normal (slice direction) as the cross product of row and column cosines
    slice_normal = np.cross(row_cosines, col_cosines)

    # Collect ImagePositionPatient for all slices and compute slice positions along the normal
    slice_positions = []
    for ds in datasets:
        ipp = np.array(ds.ImagePositionPatient, dtype=np.float64)
        distance_along_normal = np.dot(slice_normal, ipp - image_position)
        slice_positions.append(distance_along_normal)
    slice_positions = np.array(slice_positions)

    # Sort the slices based on their position along the normal
    sorted_indices = np.argsort(slice_positions)
    slice_positions = slice_positions[sorted_indices]
    datasets = [datasets[i] for i in sorted_indices]

    # Compute the distance of the patient coordinate along the slice normal
    patient_coordinate = np.array(patient_coordinate, dtype=np.float64)
    distance_patient = np.dot(slice_normal, patient_coordinate - image_position)

    # Find the slice index k closest to the patient coordinate
    k = np.argmin(np.abs(slice_positions - distance_patient))

    # Get the ImagePositionPatient of the selected slice
    selected_slice = datasets[k]
    ipp_k = np.array(selected_slice.ImagePositionPatient, dtype=np.float64)

    # Compute the in-plane displacement from the ImagePositionPatient of the slice
    delta = patient_coordinate - ipp_k

    # Build the orientation matrix
    orientation_matrix = np.vstack(
        (row_cosines * pixel_spacing[1], col_cosines * pixel_spacing[0])
    ).T  # Shape (3,2)

    # Solve for (i, j)
    try:
        ij = np.linalg.lstsq(orientation_matrix, delta, rcond=None)[0]
    except np.linalg.LinAlgError as e:
        raise ValueError(f"Linear algebra error during computation: {e}")

    i, j = ij
    # Return the image coordinate as (i, j, k)

    if i < 0 or i > series.shape[1]:
        print(f"i index {i} is out of bounds (0 to {series.shape[1]})")
    if j < 0 or j > series.shape[2]:
        print("j index {j} is out of bounds (0 to {series.shape[2]})")
    if k < 0 or k > series.shape[0]:
        print(f"k index {k} is out of bounds (0 to {series.shape[0]})")

    return (int(j), int(i), int(k))


def register_segmentation_with_filtered_landmarks(
    study: Study, modality: ModalityType, max_distance_mm: float = 5.0
) -> np.ndarray:
    """
    Register segmentation using only reliable landmarks (those with patient-space distance < threshold).

    Args:
        study: Study object with segmentation and reference points
        modality: Target modality to register to
        max_distance_mm: Maximum allowed distance in patient space for landmark to be used

    Returns:
        registered_seg: (W, H, D) segmentation in target space
    """
    import numpy as np
    from scipy.ndimage import affine_transform

    if modality == ModalityType.COR_T1:
        return study.segmentation.pixels

    print(f"\n{'='*60}")
    print(f"Filtered Landmark Registration: COR_T1 → {modality.value}")
    print(f"Max landmark distance: {max_distance_mm} mm")
    print(f"{'='*60}\n")

    # Load DICOM series
    src_dcm = study.read_image(ModalityType.COR_T1)
    target_dcm = study.read_image(modality)

    # Segmentation in native format
    seg_whd = study.segmentation.pixels  # (W, H, D)
    print(f"Source segmentation shape (W,H,D): {seg_whd.shape}")

    # Target image shape
    target_img_dhw = target_dcm.get_pixel_array()  # (D, W, H)
    print(f"Target image shape (D,W,H): {target_img_dhw.shape}")
    target_shape_whd = (
        target_img_dhw.shape[1],
        target_img_dhw.shape[2],
        target_img_dhw.shape[0],
    )  # (W, H, D)
    print(f"Target shape (W,H,D): {target_shape_whd}")

    # Get all anatomical landmarks in patient space
    src_ref = study.cor_t1_reference_points
    target_ref = getattr(study, f"{modality.value}_reference_points")

    landmark_names = [
        "lat_acr_border",
        "lat_clav_border",
        "apical_humerus",
        "lat_glen_sup",
        "lat_glen_inf",
        "proc_cora_tip",
    ]

    # Filter landmarks by patient-space distance
    filtered_src_patient = []
    filtered_tgt_patient = []
    used_landmarks = []

    print("Filtering landmarks by patient-space distance:")
    for name in landmark_names:
        src_pt = np.array(getattr(src_ref, name))
        tgt_pt = np.array(getattr(target_ref, name))
        dist = np.linalg.norm(src_pt - tgt_pt)

        if dist < max_distance_mm:
            filtered_src_patient.append(src_pt)
            filtered_tgt_patient.append(tgt_pt)
            used_landmarks.append(name)
            print(f"  ✓ {name:20s}: {dist:6.2f} mm - USING")
        else:
            print(f"  ✗ {name:20s}: {dist:6.2f} mm - SKIPPING")

    filtered_src_patient = np.array(filtered_src_patient)
    filtered_tgt_patient = np.array(filtered_tgt_patient)

    print(
        f"\n✓ Using {len(filtered_src_patient)} reliable landmarks: {', '.join(used_landmarks)}\n"
    )

    # Convert filtered landmarks to image coordinates
    src_points_whd = np.array(
        [patient_to_image_coordinate(tuple(pt), src_dcm) for pt in filtered_src_patient]
    )

    tgt_points_whd = np.array(
        [
            patient_to_image_coordinate(tuple(pt), target_dcm)
            for pt in filtered_tgt_patient
        ]
    )

    print(f"Source landmarks in image coords (W,H,D):")
    for i, (name, pt) in enumerate(zip(used_landmarks, src_points_whd)):
        print(f"  {name:20s}: {pt}")

    print(f"\nTarget landmarks in image coords (W,H,D):")
    for i, (name, pt) in enumerate(zip(used_landmarks, tgt_points_whd)):
        print(f"  {name:20s}: {pt}")

    # Compute affine transformation: tgt = src @ M.T + offset
    N = len(src_points_whd)
    src_homogeneous = np.hstack([src_points_whd, np.ones((N, 1))])

    X, residuals, rank, s = np.linalg.lstsq(src_homogeneous, tgt_points_whd, rcond=None)

    M = X[:3, :].T  # (3, 3) transformation matrix
    offset = X[3, :]  # (3,) translation

    print(f"\nAffine transformation matrix M:")
    print(M)
    print(f"\nTranslation offset:")
    print(offset)

    # Verify transformation quality
    transformed_landmarks = (src_points_whd @ M.T) + offset
    landmark_errors = np.linalg.norm(transformed_landmarks - tgt_points_whd, axis=1)

    print(f"\nLandmark alignment errors (voxels):")
    for i, (name, err) in enumerate(zip(used_landmarks, landmark_errors)):
        print(f"  {name:20s}: {err:6.2f} voxels")
    print(f"Mean error: {np.mean(landmark_errors):.2f} voxels")
    print(f"Max error: {np.max(landmark_errors):.2f} voxels")

    # Apply transformation to segmentation
    M_inv = np.linalg.inv(M)
    offset_inv = -offset @ M_inv.T

    print(f"\nApplying affine transformation...")
    registered_whd = affine_transform(
        seg_whd,
        matrix=M_inv,
        offset=offset_inv,
        output_shape=target_shape_whd,
        order=0,  # Nearest neighbor for labels
        mode="constant",
        cval=0,
    )

    print(f"Registered shape (W,H,D): {registered_whd.shape}")
    print(f"Unique labels: {np.unique(registered_whd)}")
    print(
        f"Non-zero voxels: {np.sum(registered_whd > 0)} / {registered_whd.size} ({100*np.sum(registered_whd > 0)/registered_whd.size:.2f}%)"
    )

    print(f"\n{'='*60}")
    print("Registration complete!")
    print(f"{'='*60}\n")

    return registered_whd
