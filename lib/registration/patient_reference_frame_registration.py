import numpy as np


from lib.segmentations.structs import Study, ModalityType


def register_segmentation(study: Study, modality: ModalityType) -> np.ndarray:
    """
    Register segmentation from COR_T1 to target modality using DICOM geometry.
    Work entirely in patient space, then resample to target image space.
    """
    import numpy as np
    from scipy.ndimage import map_coordinates
    from lib.dicoms.dicom_dataset import DicomDataset

    if modality == ModalityType.COR_T1:
        return study.segmentation.pixels

    # Load source and target DICOM series
    src_dcm = study.read_image(ModalityType.COR_T1)
    target_dcm = study.read_image(modality)

    seg = study.segmentation
    seg_data = seg.pixels  # (Y, X, Z)
    print("Source segmentation shape (Y,X,Z):", seg_data.shape)

    target_img = target_dcm.get_pixel_array()  # (Z, Y, X)
    print("Target image shape (Z,Y,X):", target_img.shape)
    target_shape = (
        target_img.shape[1],
        target_img.shape[2],
        target_img.shape[0],
    )  # (Y, X, Z)
    print("Target shape (Y,X,Z):", target_shape)

    # Get DICOM geometry for source
    src_ds0 = src_dcm.dataset[0]
    src_orientation = np.array(src_ds0.ImageOrientationPatient, dtype=np.float64)
    src_row_dir = src_orientation[0:3]
    src_col_dir = src_orientation[3:6]
    src_slice_dir = np.cross(src_row_dir, src_col_dir)
    src_pixel_spacing = np.array(src_ds0.PixelSpacing, dtype=np.float64)
    src_origin = np.array(src_ds0.ImagePositionPatient, dtype=np.float64)

    # Compute slice spacing for source
    if len(src_dcm.dataset) > 1:
        src_ipp_1 = np.array(src_dcm.dataset[1].ImagePositionPatient, dtype=np.float64)
        src_slice_spacing = np.abs(np.dot(src_ipp_1 - src_origin, src_slice_dir))
    else:
        src_slice_spacing = (
            src_ds0.SliceThickness if hasattr(src_ds0, "SliceThickness") else 1.0
        )

    print(
        f"Source spacing: {src_pixel_spacing[0]:.2f} x {src_pixel_spacing[1]:.2f} x {src_slice_spacing:.2f} mm"
    )

    # Get DICOM geometry for target
    tgt_ds0 = target_dcm.dataset[0]
    tgt_orientation = np.array(tgt_ds0.ImageOrientationPatient, dtype=np.float64)
    tgt_row_dir = tgt_orientation[0:3]
    tgt_col_dir = tgt_orientation[3:6]
    tgt_slice_dir = np.cross(tgt_row_dir, tgt_col_dir)
    tgt_pixel_spacing = np.array(tgt_ds0.PixelSpacing, dtype=np.float64)
    tgt_origin = np.array(tgt_ds0.ImagePositionPatient, dtype=np.float64)

    # Compute slice spacing for target
    if len(target_dcm.dataset) > 1:
        tgt_ipp_1 = np.array(
            target_dcm.dataset[1].ImagePositionPatient, dtype=np.float64
        )
        tgt_slice_spacing = np.abs(np.dot(tgt_ipp_1 - tgt_origin, tgt_slice_dir))
    else:
        tgt_slice_spacing = (
            tgt_ds0.SliceThickness if hasattr(tgt_ds0, "SliceThickness") else 1.0
        )

    print(
        f"Target spacing: {tgt_pixel_spacing[0]:.2f} x {tgt_pixel_spacing[1]:.2f} x {tgt_slice_spacing:.2f} mm"
    )

    # For each voxel in target, compute patient coordinate, then find source voxel
    print("Resampling segmentation to target space...")
    registered = np.zeros(target_shape, dtype=seg_data.dtype)

    # Create target coordinate grid (Y, X, Z)
    for z in range(target_shape[2]):
        if z % 5 == 0:
            print(f"  Processing slice {z}/{target_shape[2]}...")

        # Get image position for this slice
        tgt_slice_origin = tgt_origin + tgt_slice_dir * (z * tgt_slice_spacing)

        for y in range(target_shape[0]):
            for x in range(target_shape[1]):
                # Convert target voxel (y,x,z) to patient coordinates
                # patient_pos = origin + y*row_dir*row_spacing + x*col_dir*col_spacing
                patient_pos = (
                    tgt_slice_origin
                    + y * tgt_row_dir * tgt_pixel_spacing[0]
                    + x * tgt_col_dir * tgt_pixel_spacing[1]
                )

                # Convert patient coordinates back to source voxel indices
                delta = patient_pos - src_origin

                # Project onto source axes
                src_y = np.dot(delta, src_row_dir) / src_pixel_spacing[0]
                src_x = np.dot(delta, src_col_dir) / src_pixel_spacing[1]
                src_z = np.dot(delta, src_slice_dir) / src_slice_spacing

                # Round to nearest voxel
                src_y_idx = int(np.round(src_y))
                src_x_idx = int(np.round(src_x))
                src_z_idx = int(np.round(src_z))

                # Check bounds and copy label
                if (
                    0 <= src_y_idx < seg_data.shape[0]
                    and 0 <= src_x_idx < seg_data.shape[1]
                    and 0 <= src_z_idx < seg_data.shape[2]
                ):
                    registered[y, x, z] = seg_data[src_y_idx, src_x_idx, src_z_idx]

    print("Registered shape (Y,X,Z):", registered.shape)
    print("Registered unique values:", np.unique(registered))
    print("Non-zero voxels in registered:", np.sum(registered > 0))

    return registered
