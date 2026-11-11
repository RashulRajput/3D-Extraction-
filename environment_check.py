from importlib import import_module, metadata
import json
import platform
import sys
import traceback


PACKAGES = {
    "numpy": "numpy", "scipy": "scipy", "SimpleITK": "SimpleITK", "pydicom": "pydicom",
    "pynrrd": "nrrd", "PyRadiomics": "radiomics", "napari": "napari", "PyQt5": "PyQt5",
    "magicgui": "magicgui", "matplotlib": "matplotlib", "scikit-image": "skimage",
    "PyYAML": "yaml", "tqdm": "tqdm",
}


def main():
    result = {"python": sys.version, "platform": platform.platform(), "execution_mode": "CPU only", "packages": {}, "errors": []}
    for dist, module in PACKAGES.items():
        try:
            import_module(module)
            result["packages"][dist] = metadata.version(dist)
        except Exception as exc:
            result["errors"].append(f"{dist}: {type(exc).__name__}: {exc}\n{traceback.format_exc()}")
    print(json.dumps(result, indent=2))
    if result["errors"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
