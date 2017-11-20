from documentAnalysis import *
from multiprocessing import cpu_count
from step1 import run_step1

import os
import pandas as pd
import logging
from azureml.logging import get_azureml_logger


def run_step2(cleanedDataFrame, config=(0, 0, 0), numPhrase=MAX_NUM_PHRASE, maxPhrasePerIter=MAX_PHRASE_PER_ITER,
                maxPhraseLength=MAX_PHRASE_LENGTH, minInstanceCount=MIN_INSTANCE_COUNT):
    """
    Step 2: phrase learning
    """
    aml_logger = get_azureml_logger()   # logger writes to AMLWorkbench runtime view
    aml_logger.log('amlrealworld.document-collection-analysis.step2', 'true')
    
    logger = logging.getLogger(__name__)
    logger.info("=========  Run Step 2: learn phrases from data")

    minPhrase, maxPhrase, step = config

    if minPhrase == 0 and maxPhrase == 0 and step == 0:
        logger.info("Only need to learn %d phrases" % numPhrase)
        # Instantiate a PhraseLearner and run a configuration
        # We need to put this code under '__main__' to run multiprocessing
        phraseLearner = PhraseLearner(cleanedDataFrame, "CleanedText", numPhrase,
                                maxPhrasePerIter, maxPhraseLength, minInstanceCount)

        textData = list(phraseLearner.textFrame['LowercaseText'])
        phraseLearner.RunConfiguration(textData,
                    phraseLearner.learnedPhrases,
                    addSpace=True,
                    writeFile=True,
                    num_workers=cpu_count()-1)

        phraseLearner.textFrame['TextWithPhrases'] = textData
        phraseLearner.MapVocabToSurfaceForms('CleanedText', 'TextWithPhrases', True)
        newDocsFrame = phraseLearner.ReconstituteDocsFromChunks('DocID', 'TextWithPhrases', True)
    else:
        # make sure the inputs are valid and make sense
        minPhrase = max(10, minPhrase)
        maxPhrase = max(10, maxPhrase)
        step = max(1, step)

        # Instance a phrase learner with minPhrase set
        phraseLearner = PhraseLearner(cleanedDataFrame, "CleanedText", minPhrase,
                                maxPhrasePerIter, maxPhraseLength, minInstanceCount)
        # Get the lower case text data
        textData = list(phraseLearner.textFrame['LowercaseText'])

        # Incrementally learn phrases
        for i in range(minPhrase, maxPhrase + 1, step):
            logger.info("Learning %d phrases, based on previous leaned %d phrases" % (i, len(phraseLearner.learnedPhrases)))

            # need to update this number
            phraseLearner.maxNumPhrases = i
            phraseLearner.RunConfiguration(textData,
                        phraseLearner.learnedPhrases,
                        addSpace=True,
                        writeFile=True,
                        num_workers=cpu_count()-1)

            phraseLearner.textFrame['TextWithPhrases'] = textData
            phraseLearner.MapVocabToSurfaceForms('CleanedText', 'TextWithPhrases', True)
            newDocsFrame = phraseLearner.ReconstituteDocsFromChunks('DocID', 'TextWithPhrases', True)



"""
main
"""
if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s : %(name)s : %(levelname)s : %(message)s', level=logging.INFO)

    cleanedDataFrame = run_step1(saveFile=False)
    run_step2(cleanedDataFrame=cleanedDataFrame, config=(2000, 2000, 1))

