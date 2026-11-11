from __future__ import annotations

import json
import subprocess
import sys
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "scripts"))
from convert_dicom import convert

DEFAULT_DICOMDIR = Path(r"C:\Users\DELL\OneDrive\AppData\Desktop\PREDICT-IPN\DICOMDIR")
IMAGE = ROOT / "data/images/volume.nrrd"
MASK = ROOT / "data/masks/ROI_label.nrrd"
REPORT = ROOT / "data/inspection/dicom_inspection.json"


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("EHBA 3D DICOM Radiomics")
        self.geometry("760x590")
        self.selected_series = None
        self.dicomdir = DEFAULT_DICOMDIR
        ttk.Label(self, text="3D DICOM Radiomics", font=("Segoe UI", 18, "bold")).pack(pady=(18, 8))
        self.status = tk.StringVar(value="Scan a DICOM source to find and verify one spatial 3D CT or ultrasound series.")
        ttk.Label(self, textvariable=self.status, wraplength=710, justify="left").pack(fill="x", padx=22, pady=8)
        panel = ttk.Frame(self)
        panel.pack(fill="both", expand=True, padx=20, pady=12)
        actions = [
            ("Scan DICOMDIR", self.scan), ("Inspect DICOM series", self.inspect),
            ("Select image volume", self.select_series), ("Convert to NRRD", self.convert),
            ("Open 3D Mask Annotator", self.annotate), ("Save ROI Mask", self.annotate),
            ("Verify Geometry", self.verify), ("Extract 3D Radiomics", self.extract),
            ("View Feature Results", self.view_results), ("Open Results Folder", self.open_results),
        ]
        for i, (label, command) in enumerate(actions):
            ttk.Button(panel, text=label, command=command).grid(row=i // 2, column=i % 2, sticky="ew", padx=8, pady=8, ipady=7)
        panel.columnconfigure(0, weight=1)
        panel.columnconfigure(1, weight=1)
        self.log = tk.Text(self, height=9, wrap="word", state="disabled")
        self.log.pack(fill="x", padx=20, pady=(0, 18))
        self.log_message("No patient identifiers are shown in this application or its reports.")

    def log_message(self, text):
        self.log.configure(state="normal")
        self.log.insert("end", text.rstrip() + "\n")
        self.log.see("end")
        self.log.configure(state="disabled")
        self.status.set(text)

    def scan(self):
        selected = filedialog.askopenfilename(title="Select DICOMDIR", initialdir=str(self.dicomdir.parent), initialfile="DICOMDIR")
        if selected:
            self.dicomdir = Path(selected)
        if not self.dicomdir.parent.is_dir():
            messagebox.showerror("Source folder not found", f"No folder at {self.dicomdir.parent}")
            return
        cmd = [sys.executable, str(ROOT / "scripts/inspect_dicomdir.py"), "--dicomdir", str(self.dicomdir), "--output", str(REPORT)]
        result = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True)
        if result.returncode:
            messagebox.showerror("DICOM scan failed", result.stderr or result.stdout)
            self.log_message("DICOM scan failed. See the error dialog.")
            return
        report = json.loads(REPORT.read_text(encoding="utf-8"))
        if not report.get("dicomdir_exists", False):
            self.log_message(f"DICOMDIR is missing; controlled scan of {self.dicomdir.parent} found {report['candidate_image_count']} DICOM images.")
        self.log_message(f"DICOMDIR references: {report['referenced_file_count']}; resolved: {report['resolved_reference_count']}; images: {report['candidate_image_count']}; missing references: {len(report['missing_references'])}.")
        if not report["series"]:
            self.log_message("No DICOM image files were found. Restore the missing referenced files, then scan again.")

    def _series(self):
        if not REPORT.exists():
            raise FileNotFoundError("Scan the DICOMDIR first.")
        return json.loads(REPORT.read_text(encoding="utf-8"))["series"]

    def inspect(self):
        try:
            series = self._series()
        except Exception as exc:
            messagebox.showerror("Series inspection", str(exc)); return
        if not series:
            messagebox.showinfo("Series inspection", "No image series were found in the available DICOM files."); return
        lines = []
        for s in series:
            lines.append(f"{s['modality'] or 'Unknown'} | {s['description'] or 'Undescribed'} | {s['file_count']} files | {s['rows']}×{s['columns']} | {s['number_of_frames']} frames | {s['spatial_status']}\n{s['spatial_reason']}")
        messagebox.showinfo("DICOM series (patient identifiers omitted)", "\n\n".join(lines))

    def select_series(self):
        try:
            series = self._series()
        except Exception as exc:
            messagebox.showerror("Select volume", str(exc)); return
        choices = [s for s in series if s.get("modality") in ("US", "CT") and s.get("spatial_status") in ("true_spatial_3d_candidate", "spatially_ordered_slices_candidate")]
        if not choices:
            messagebox.showinfo("No spatial 3D series", "No spatially verified CT or ultrasound volume was found.")
            return
        if len(choices) == 1:
            self.selected_series = choices[0]
            self.log_message(f"Selected the only ultrasound series: {self.selected_series.get('description') or 'undescribed'} — {self.selected_series['spatial_status']}.")
            return
        dialog = tk.Toplevel(self); dialog.title("Select ultrasound series"); dialog.geometry("700x380")
        box = tk.Listbox(dialog, height=12, exportselection=False)
        for item in choices:
            box.insert("end", f"{item.get('description') or '(no description)'} | {item['file_count']} files | {item['rows']}×{item['columns']} | {item['number_of_frames']} frames | {item['spatial_status']}")
        box.pack(fill="both", expand=True, padx=14, pady=14)
        def choose():
            selection = box.curselection()
            if not selection:
                messagebox.showwarning("Select a series", "Choose an ultrasound series first.", parent=dialog); return
            self.selected_series = choices[selection[0]]
            self.log_message(f"Selected {self.selected_series.get('description') or 'ultrasound series'} — {self.selected_series['spatial_status']}.")
            dialog.destroy()
        ttk.Button(dialog, text="Select", command=choose).pack(pady=(0, 12))
        if len(choices) == 1:
            box.selection_set(0)

    def convert(self):
        if not self.selected_series:
            messagebox.showwarning("Select a volume", "Inspect and select an ultrasound series first."); return
        if self.selected_series["spatial_status"] not in ("true_spatial_3d_candidate", "spatially_ordered_slices_candidate"):
            messagebox.showerror("3D geometry not verified", self.selected_series["spatial_reason"]); return
        try:
            _, report = convert([Path(f["path"]) for f in self.selected_series["files"]], IMAGE)
            self.log_message(f"NRRD created: {report['size']} voxels, spacing {report['spacing']}.")
        except Exception as exc:
            messagebox.showerror("Conversion stopped", str(exc)); self.log_message(f"Conversion stopped: {exc}")

    def annotate(self):
        if not IMAGE.exists():
            messagebox.showwarning("No working volume", "Convert a verified 3D image first. The source files are currently unavailable."); return
        subprocess.Popen([sys.executable, str(ROOT / "scripts/mask_annotator.py"), "--image", str(IMAGE), "--mask", str(MASK)], cwd=ROOT)

    def verify(self):
        if not MASK.exists():
            messagebox.showwarning("Mask missing", "Save a manually drawn ROI mask first."); return
        result = subprocess.run([sys.executable, str(ROOT / "scripts/verify_geometry.py"), "--image", str(IMAGE), "--mask", str(MASK), "--output", str(ROOT / "results/geometry_report.json")], cwd=ROOT, text=True, capture_output=True)
        messagebox.showinfo("Geometry verification", result.stdout if result.returncode == 0 else result.stderr or result.stdout)

    def extract(self):
        if not MASK.exists():
            messagebox.showwarning("Mask missing", "Save an ROI mask before feature extraction."); return
        result = subprocess.run([sys.executable, str(ROOT / "scripts/extract_radiomics.py")], cwd=ROOT, text=True, capture_output=True)
        if result.returncode:
            messagebox.showerror("Radiomics extraction failed", result.stderr or result.stdout); return
        self.log_message("3D radiomics features saved to results/radiomics_features.json")
        messagebox.showinfo("Extraction complete", result.stdout)

    def view_results(self):
        path = ROOT / "results/radiomics_features.json"
        if not path.exists():
            messagebox.showinfo("No results", "No feature results exist yet."); return
        self.open_path(path)

    def open_results(self):
        self.open_path(ROOT / "results")

    @staticmethod
    def open_path(path):
        import os
        os.startfile(str(path))


if __name__ == "__main__":
    App().mainloop()
