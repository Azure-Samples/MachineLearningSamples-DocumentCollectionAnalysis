from documentAnalysis import *
import os
import pandas as pd
import logging


def run_step1():
    """
    Step 1: data preprocessing
    """
    logger = logging.getLogger(__name__)

    fpath = get_shared_file_path(CLEANED_DATA_FILE_NAME)
    logger.info("=========  Run Step 1: preprocessing text data")

    if not os.path.exists(fpath):
        # Read raw data into a Pandas DataFrame
        textDF = getData()
        # Write frame with preprocessed text out to TSV file
        cleanedDataFrame = CleanAndSplitText(textDF, saveDF=True)
    else:
        logger.info("File already existed, directly read it")
        cleanedDataFrame = pd.read_csv(fpath, sep='\t', encoding="ISO-8859-1")
    
    return cleanedDataFrame



"""
main
"""
if __name__ == "__main__":
    cleanedDataFrame = run_step1()
    print(cleanedDataFrame.head())
