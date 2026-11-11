from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import sys

import numpy as np
import SimpleITK as sitk


def _hash(data):
    return hashlib.sha256(np.ascontiguousarray(data).view(np.uint8)).hexdigest()


def export_mask(labels: np.ndarray, reference: sitk.Image, mask_path: Path):
    if labels.shape != sitk.GetArrayViewFromImage(reference).shape:
        raise ValueError("Label array shape does not match the reference image array shape.")
    if not np.issubdtype(labels.dtype, np.integer):
        raise TypeError("Mask labels must be integer values.")
    result = sitk.GetImageFromArray(np.asarray(labels, dtype=np.uint8))
    result.CopyInformation(reference)
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(result, str(mask_path), True)
    check = sitk.ReadImage(str(mask_path))
    if check.GetSize() != reference.GetSize() or check.GetSpacing() != reference.GetSpacing() or check.GetOrigin() != reference.GetOrigin() or check.GetDirection() != reference.GetDirection():
        mask_path.unlink(missing_ok=True)
        raise RuntimeError("Saved mask geometry verification failed; invalid output removed.")
    return check


def launch(image_path: Path, mask_path: Path, reference_path: Path | None = None):
    os.environ.setdefault("QT_OPENGL", "software")
    import napari
    from qtpy.QtCore import QEvent, QObject, QProcess
    from qtpy.QtWidgets import QComboBox, QHBoxLayout, QLabel, QMessageBox, QPushButton, QSpinBox, QVBoxLayout, QWidget

    image = sitk.ReadImage(str(image_path))
    if image.GetDimension() != 3:
        raise ValueError(f"Annotator requires a verified 3D volume; got {image.GetDimension()}D.")
    array = sitk.GetArrayFromImage(image)
    if mask_path.exists():
        mask_image = sitk.ReadImage(str(mask_path))
        if mask_image.GetSize() != image.GetSize() or mask_image.GetSpacing() != image.GetSpacing() or mask_image.GetOrigin() != image.GetOrigin() or mask_image.GetDirection() != image.GetDirection():
            raise ValueError("Existing mask geometry differs from image. Fix/inspect it before editing; no resampling was performed.")
        labels = sitk.GetArrayFromImage(mask_image).astype(np.uint8, copy=False)
    else:
        labels = np.zeros(array.shape, dtype=np.uint8)
    viewer = napari.Viewer(ndisplay=2, title="3D ROI Mask Annotator — ROI label 1 (CPU rendering)")
    viewer.add_image(array, name="Image volume", scale=image.GetSpacing()[::-1], contrast_limits=(float(np.min(array)), float(np.max(array))))
    label_layer = viewer.add_labels(labels, name="ROI", scale=image.GetSpacing()[::-1], opacity=0.45, color={1: "#ff3158"})
    label_layer.brush_size = 5
    viewer.dims.order = (0, 1, 2)
    saved_hash = [_hash(labels)]
    label_layer.events.data.connect(lambda event: setattr(label_layer, "_roi_dirty", _hash(label_layer.data) != saved_hash[0]))

    panel = QWidget()
    layout = QVBoxLayout(panel)
    layout.addWidget(QLabel("Paint or erase label 1 (ROI) across slices. Ctrl+Z / Ctrl+Shift+Z: undo/redo."))
    plane = QComboBox()
    plane.addItems(["Axial (XY)", "Coronal (XZ)", "Sagittal (YZ)"])
    plane.currentIndexChanged.connect(lambda i: setattr(viewer.dims, "order", [(0, 1, 2), (1, 0, 2), (2, 0, 1)][i]))
    layout.addWidget(plane)
    brush = QSpinBox()
    brush.setRange(1, 128)
    brush.setValue(5)
    brush.setSuffix(" voxel brush")
    brush.valueChanged.connect(lambda n: setattr(label_layer, "brush_size", n))
    layout.addWidget(brush)
    tools = QHBoxLayout()
    for name, mode in (("Paint", "paint"), ("Erase", "erase")):
        button = QPushButton(name)
        button.clicked.connect(lambda checked=False, value=mode: setattr(label_layer, "mode", value))
        tools.addWidget(button)
    layout.addLayout(tools)
    save = QPushButton("Save Mask")
    clear = QPushButton("Clear Mask")
    layout.addWidget(save)
    layout.addWidget(clear)

    def save_mask():
        data = np.asarray(label_layer.data, dtype=np.uint8)
        export_mask(data, image, mask_path)
        saved_hash[0] = _hash(data)
        label_layer._roi_dirty = False
        QMessageBox.information(panel, "Mask saved", f"Mask saved with matching geometry:\n{mask_path}\nROI voxels: {int(np.count_nonzero(data == 1))}")
        if np.any(data == 1):
            project_root = Path(__file__).parents[1]
            extraction_process.setProgram(sys.executable)
            extraction_process.setArguments([
                str(Path(__file__).with_name("extract_radiomics.py")), "--image", str(image_path),
                "--mask", str(mask_path), "--config", str(project_root / "config/radiomics.yaml"),
                "--output-dir", str(project_root / "results"),
            ])
            extraction_process.setWorkingDirectory(str(project_root))
            extraction_process.start()
            if not extraction_process.waitForStarted(3000):
                QMessageBox.warning(panel, "Extraction not started", "The mask was saved, but PyRadiomics could not be started. Use the main app's Extract 3D Radiomics button.")
            else:
                QMessageBox.information(panel, "Extraction started", "A non-empty mask was saved. 3D PyRadiomics extraction is running in the background.")

    def clear_mask():
        if QMessageBox.question(panel, "Clear ROI?", "Erase every ROI label from the volume?", QMessageBox.Yes | QMessageBox.No) == QMessageBox.Yes:
            label_layer.data = np.zeros(array.shape, dtype=np.uint8)

    save.clicked.connect(save_mask)
    clear.clicked.connect(clear_mask)
    viewer.window.add_dock_widget(panel, area="right", name="ROI controls")
    extraction_process = QProcess(panel)

    def extraction_finished(exit_code, exit_status):
        stderr = bytes(extraction_process.readAllStandardError()).decode("utf-8", errors="replace")
        if exit_code == 0:
            QMessageBox.information(panel, "Radiomics complete", "Features and reports were saved in the project results folder.")
        else:
            QMessageBox.critical(panel, "Radiomics failed", stderr[-5000:] or f"Process exited with code {exit_code}.")

    extraction_process.finished.connect(extraction_finished)

    class CloseGuard(QObject):
        def eventFilter(self, obj, event):
            if event.type() == QEvent.Close and _hash(label_layer.data) != saved_hash[0]:
                answer = QMessageBox.question(panel, "Unsaved ROI changes", "Save changes before closing?", QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel)
                if answer == QMessageBox.Save:
                    save_mask()
                elif answer == QMessageBox.Cancel:
                    return True
            return False

    close_guard = CloseGuard(viewer.window._qt_window)
    viewer.window._qt_window.installEventFilter(close_guard)
    if reference_path is not None and reference_path.exists():
        viewer.open(str(reference_path), plugin="napari")
    napari.run()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=Path, default=Path("data/images/volume.nrrd"))
    parser.add_argument("--mask", type=Path, default=Path("data/masks/ROI_label.nrrd"))
    parser.add_argument("--reference", type=Path)
    args = parser.parse_args()
    launch(args.image, args.mask, args.reference)


if __name__ == "__main__":
    main()
