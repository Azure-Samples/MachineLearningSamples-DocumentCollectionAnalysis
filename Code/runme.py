from documentAnalysis import *
import logging
import pandas as pd
import os

from multiprocessing import cpu_count
from step1 import run_step1
from step2 import run_step2
from step3 import run_step3, copyFigures, visualizeTopic, saveModel
from azureml.logging import get_azureml_logger



if __name__ == '__main__':
    aml_logger = get_azureml_logger()   # logger writes to AMLWorkbench runtime view
    aml_logger.log('amlrealworld.document-collection-analysis.runme', 'true')

    logging.basicConfig(format='%(asctime)s : %(name)s : %(levelname)s : %(message)s', level=logging.INFO)

    """
    By default, this script will use the entire Congressional dataset.
    If you just need to try it on a smaller dataset, choose the right 'DATASET_FILE'
    setting in documentAnalysis/configs.py file.
    """
    # Step 1: Data preprocessing
    cleanedDataFrame = run_step1(saveFile=False)

    # Step 2: Phrase learning
    run_step2(cleanedDataFrame=cleanedDataFrame, numPhrase=MAX_NUM_PHRASE, maxPhrasePerIter=MAX_PHRASE_PER_ITER,
               maxPhraseLength=MAX_PHRASE_LENGTH, minInstanceCount=MIN_INSTANCE_COUNT)

    # Step 3: Topic modeling
    """
    Train multiple LDA models and evaluate their performance
    """
    # run_step3(topicConfig=[10, 20, 30, 40, 50, 60, 70, 80], 
    #             test_ratio=0.005,
    #             saveModel=False, 
    #             coherence_types=['u_mass', 'c_v'])
    # copyFigures()

    """
    Train one topic model
    """
    # Set topicConfig to an empty list to use default number of topics
    lda = run_step3(topicConfig=[], test_ratio=0.005, saveModel=False, 
                        coherence_types=['u_mass', 'c_v', 'c_uci', 'c_npmi'])
    
    # In case topicConfig is not an empty list, which will NOT return a topic model
    if lda:
        visualizeTopic(lda, topicID=0, topn=1000, multiplier=1000)
        visualizeTopic(lda, topicID=10, topn=1000, multiplier=1000)
        saveModel(lda)



