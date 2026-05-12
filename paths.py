"""Central path resolver for the project.

Set the ``APO_ROOT`` environment variable to point at the project root
(e.g. ``/content/drive/MyDrive/Automatic-Prompt-Optimization`` on Colab,
or ``C:\\Users\\cypri\\Desktop\\Master Thesis`` on the original dev box).
If unset, ``ROOT`` falls back to the directory that contains this file,
which is the repo checkout itself.
"""
import os
import pathlib

ROOT = pathlib.Path(
    os.environ.get("APO_ROOT", pathlib.Path(__file__).resolve().parent)
)

DATA_DIR = ROOT / "data"
REFERENCES_DIR = ROOT / "references"
VECTORSTORES_DIR = ROOT / "vectorstores"
RESULTS_DIR = ROOT / "results"
