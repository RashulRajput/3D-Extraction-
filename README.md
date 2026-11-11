# EHBA 3D DICOM Radiomics

Windows desktop workflow for selecting one spatially verified CT or ultrasound series, converting it to NRRD, drawing a manual ROI mask in napari, checking image/mask geometry, and extracting 3D PyRadiomics features. Imaging calculations and napari rendering are configured for CPU-only operation.

## Current data status

The ZIP supplied by the user contains 672 DICOM images. It includes a complete 625-slice CT series (`Abdomen 1.0 I30f 3`) with consistent orientation and regular 0.7 mm slice spacing; this is the selected volume. It also includes a separate 44-slice CT series reconstructed at 10 mm and localizer images. The three previously supplied files were sparse examples from the 625-slice series, not the complete volume. The selected series has been converted to `data/images/volume.nrrd` (512 × 512 × 625 voxels; spacing 0.73828125 × 0.73828125 × 0.7 mm). This input is CT, not ultrasound.

The supplied mask ZIP contains three NIfTI masks. Geometry matching selected `INF_0002_mask.nii.gz` for this volume; its label 1 mask was imported to `data/masks/ROI_label.nrrd`. Geometry QC passes. CPU PyRadiomics extraction completed with 107 features (shape, first-order, and texture classes) in `results/radiomics_features.json`.

The Drive links could not be read from this environment because the Google Drive connector is disabled. This project only scans the selected DICOMDIR's parent folder when its references are missing.

## Setup (PowerShell)

```powershell
cd "$env:USERPROFILE\\OneDrive\\AppData\\Desktop\\EHBA_3D_Radiomics_Project"
py -3.10 -m venv .venv
.\\.venv\\Scripts\\Activate.ps1
python -m pip install --upgrade pip
powershell -ExecutionPolicy Bypass -File scripts\\install_core.ps1
python -m pip check
python environment_check.py
```

Python 3.11 was not installed when this project was prepared, so the project uses the installed Python 3.10 interpreter. No packages are installed globally. `requirements-monai.txt` is an optional CPU-only environment for future supervised work; it is not needed for manual masks or radiomics.

## Run

```powershell
python run_app.py
```

From the app: Scan the DICOM source, inspect series, choose one CT or ultrasound series, and convert only if spatial 3D geometry is verified. Then use **Open 3D Mask Annotator**. The napari viewer opens an empty label 1 (`ROI`) layer; choose axial, coronal, or sagittal orientation, paint/erase across slices, and save an intermediate mask with **Save Mask**. Saving a non-empty mask starts 3D feature extraction automatically; an empty intermediate mask is saved without extraction.

## Command line

```powershell
python scripts\\inspect_dicomdir.py --dicomdir "C:\\path\\to\\DICOMDIR"
python scripts\\import_mask_archive.py --archive "C:\\path\\to\\mask-archive.zip"
python scripts\\mask_annotator.py --image data\\images\\volume.nrrd --mask data\\masks\\ROI_label.nrrd
python scripts\\verify_geometry.py --image data\\images\\volume.nrrd --mask data\\masks\\ROI_label.nrrd
python scripts\\extract_radiomics.py
Get-Content results\\radiomics_features.json
explorer results
```

Feature extraction requires a non-empty label 1 ROI, matching image and mask geometry, and enough voxels for 3D texture features. `config/radiomics.yaml` sets Original image type, `force2D: false`, no resampling, no normalization, and provisional `binWidth: 25`; validate CT discretization choices for the research method.

## Automatic segmentation limits

PyRadiomics does not create a mask. A single unlabelled image cannot train a valid supervised segmentation model, and this project will not fabricate an ROI. Automatic extraction can proceed only when a valid corresponding segmentation is supplied. The current pipeline creates a manual reference mask in napari and keeps future model predictions separate. No model training or diagnostic classification is included.

## VS Code

Open `EHBA_3D_Radiomics.code-workspace`, or run `open_in_vscode.bat`. The Python extension should select `.venv\\Scripts\\python.exe`.
