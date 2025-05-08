import os
import shutil

from config.settings import TMP_DIR


def clean_temp_dir():
    """
    Clean the temporary directory to ensure a fresh training
    """
    if os.path.exists(TMP_DIR):
        print("🧹 Cleaning temporary directory...")
        shutil.rmtree(TMP_DIR)
        print("✅ Temporary directory cleaned")
