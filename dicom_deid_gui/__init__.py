__all__ = [
    "__version__",
]

__version__ = "0.1.0"


# Allow `python -m dicom_deid_gui` to run GUI
def main() -> int:  # pragma: no cover
    from .app import main as gui_main

    return gui_main()
