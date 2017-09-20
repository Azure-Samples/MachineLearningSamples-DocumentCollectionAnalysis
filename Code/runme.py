from documentAnalysis import *
import logging
import pandas as pd
import os

from multiprocessing import cpu_count
from step1 import run_step1
from step2 import run_step2
from step3 import run_step3, copyFigures, visualizeTopic, saveModel



if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s : %(name)s : %(levelname)s : %(message)s', level=logging.INFO)

    cleanedDataFrame = run_step1()
    run_step2(cleanedDataFrame)

    """
    Train multiple LDA models and evaluate their performance
    """
    # run_step3(topicConfig=[10, 20, 30, 40, 50, 100, 150, 175, 200, 225], 
    #             test_ratio=0.005,
    #             saveModel=False, 
    #             coherence_types=['u_mass', 'c_v'])
    # copyFigures()

    """
    Train one topic model
    """
    lda = run_step3(saveModel=False, coherence_types=['u_mass', 'c_v', 'c_uci', 'c_npmi'])
    visualizeTopic(lda, topicID=0, topn=1000, multiplier=1000)
    visualizeTopic(lda, topicID=10, topn=1000, multiplier=1000)
    saveModel(lda)



