from documentAnalysis import *
import os
import pandas as pd
import logging
from azureml.logging import get_azureml_logger


def run_step1(saveFile=True):
    """
    Step 1: data preprocessing
    """
    aml_logger = get_azureml_logger()   # logger writes to AMLWorkbench runtime view
    aml_logger.log('amlrealworld.document-collection-analysis.step1', 'true')
    
    logger = logging.getLogger(__name__)

    fpath = get_shared_file_path(CLEANED_DATA_FILE_NAME)
    logger.info("=========  Run Step 1: preprocessing text data")

    # Read raw data into a Pandas DataFrame
    textDF = getData()

    # Write frame with preprocessed text out to TSV file
    cleanedDataFrame = CleanAndSplitText(textDF, idColumnName='ID', textColumnName='Text', saveDF=saveFile)
    
    return cleanedDataFrame



"""
main
"""
if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s : %(name)s : %(levelname)s : %(message)s', level=logging.INFO)

    cleanedDataFrame = run_step1(saveFile=True)
    print(cleanedDataFrame.head())
