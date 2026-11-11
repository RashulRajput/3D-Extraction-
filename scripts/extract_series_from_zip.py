from __future__ import annotations

import argparse
import json
from pathlib import Path, PurePosixPath
from zipfile import ZipFile

import pydicom


def extract_series(archive: Path, series_uid: str, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=False)
    extracted = []
    try:
        with ZipFile(archive) as zf:
            for member in zf.infolist():
                if member.is_dir():
                    continue
                try:
                    with zf.open(member.filename) as stream:
                        ds = pydicom.dcmread(stream, stop_before_pixels=True, force=False)
                except Exception:
                    continue
                if str(getattr(ds, "SeriesInstanceUID", "")) != series_uid or not hasattr(ds, "Rows"):
                    continue
                rel = PurePosixPath(member.filename)
                if rel.is_absolute() or ".." in rel.parts:
                    raise ValueError(f"Unsafe path inside archive: {member.filename}")
                target = output_dir.joinpath(*rel.parts)
                target.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(member.filename) as src, target.open("wb") as dst:
                    while chunk := src.read(1024 * 1024):
                        dst.write(chunk)
                extracted.append(target)
    except Exception:
        # Leave partial output visible for diagnosis; caller can remove the known staging directory.
        raise
    if not extracted:
        raise ValueError(f"No image files matched SeriesInstanceUID {series_uid}")
    return sorted(extracted, key=lambda p: str(p).lower())


def main():
    parser = argparse.ArgumentParser(description="Extract only the requested DICOM series from a ZIP to a staging folder.")
    parser.add_argument("archive", type=Path)
    parser.add_argument("series_uid")
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    files = extract_series(args.archive, args.series_uid, args.output_dir)
    print(json.dumps({"series_uid": args.series_uid, "extracted_file_count": len(files),
                      "output_dir": str(args.output_dir)}, indent=2))


if __name__ == "__main__":
    main()
