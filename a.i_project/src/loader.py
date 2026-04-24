import os
import pandas as pd
import pdfplumber
import logging

# ----------------------------
# PROJECT ROOT
# ----------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ----------------------------
# CSV LOADER
# ----------------------------
def load_csv(relative_path):
    file_path = os.path.join(BASE_DIR, relative_path)

    if not os.path.exists(file_path):
        logging.error(f"CSV file not found: {file_path}")
        raise FileNotFoundError(f"CSV file not found: {file_path}")

    try:
        df = pd.read_csv(file_path)

        if df.empty:
            logging.warning("CSV file is empty")

        else:
            logging.info(f"CSV loaded successfully: {df.shape}")

        return df

    except Exception as e:
        logging.error(f"Error loading CSV: {str(e)}")
        raise


# ----------------------------
# PDF LOADER
# ----------------------------
def load_pdf(relative_path):
    file_path = os.path.join(BASE_DIR, relative_path)

    if not os.path.exists(file_path):
        logging.error(f"PDF file not found: {file_path}")
        raise FileNotFoundError(f"PDF file not found: {file_path}")

    text = ""

    try:
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"

        if not text.strip():
            logging.warning(f"PDF loaded but empty text: {file_path}")

        else:
            logging.info(f"PDF loaded successfully: {len(text)} characters")

        return text

    except Exception as e:
        logging.error(f"Error loading PDF: {str(e)}")
        raise
