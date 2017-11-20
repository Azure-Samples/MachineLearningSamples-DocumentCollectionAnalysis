from documentAnalysis import *
from multiprocessing import cpu_count
from gensim.models import LdaMulticore
from shutil import copyfile
from wordcloud import WordCloud
from datetime import datetime

import os
import pandas as pd
import logging
import random
import glob
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from azureml.logging import get_azureml_logger


def run_step3(topicConfig=[], test_ratio=0.005, saveModel=True, coherence_types=['u_mass', 'c_v', 'c_uci', 'c_npmi']):
    """
    Step 3: LDA topic modeling
    """
    aml_logger = get_azureml_logger()   # logger writes to AMLWorkbench runtime view
    aml_logger.log('amlrealworld.document-collection-analysis.step3', 'true')

    logger = logging.getLogger(__name__)
    logger.info("=========  Run Step 3: LDA topic modeling")

    run_logger = get_azureml_logger()

    df = loadProcessedTextData(numPhrases=MAX_NUM_PHRASE)
    if df is None or len(df) == 0:
        raise ValueError("Failed to load the processed text data")

    docs = list(df['ProcessedText'])
    if test_ratio >= 1.0 or test_ratio < 0.0:
        test_ratio = 0.005

    topicmodeler = TopicModeler(docs, 
            stopWordFile=FUNCTION_WORDS_FILE,
            minWordCount=MIN_WORD_COUNT, 
            minDocCount=MIN_DOC_COUNT, 
            maxDocFreq=MAX_DOC_FREQ, 
            workers=cpu_count()-1, 
            numTopics=NUM_TOPICS, 
            numIterations=NUM_ITERATIONS, 
            passes=NUM_PASSES, 
            chunksize=CHUNK_SIZE, 
            random_state=RANDOM_STATE,
            test_ratio=test_ratio)

    if topicConfig is None or len(topicConfig) == 0:
        logger.info("Only need to learn %d topics" % NUM_TOPICS)
        
        lda = topicmodeler.TrainLDA(saveModel=saveModel)
        coherence = topicmodeler.EvaluateCoherence(lda, coherence_types)
        perplex = topicmodeler.EvaluatePerplexity(lda)
        run_logger.log("Perplexity", perplex['perplexity'])
        run_logger.log("Per Word Bound", perplex['per_word_bound'])
        for type in coherence:
            run_logger.log(type + " Coherence", coherence[type])
        run_logger.log("Topic Number", NUM_TOPICS)

        return lda
    else:
        for i in topicConfig:
            logger.info("Learning %d topics, from list of topic configuration: %s" % (i, str(topicConfig)))

            # IMPORTANT: update the number of topics need to learn
            topicmodeler.numTopics = i
            
            # train an LDA model
            lda = topicmodeler.TrainLDA(saveModel=saveModel)

            topicmodeler.EvaluateCoherence(lda, coherence_types)
            topicmodeler.EvaluatePerplexity(lda)
        topicmodeler.CollectRunLog()
        topicmodeler.PlotRunLog()


def saveModel(lda=None):
    logger = logging.getLogger("__name__")
    
    if lda is None:
        raise ValueError("The LDA model is None")
    
    # save model into output
    name_seps = LDA_FILE.split('.')
    file_name = '_'.join([name_seps[0], datetime.now().strftime("%Y_%m_%d_%H_%M_%S")])
    file_name += '.' + name_seps[1]

    fpath = os.path.join(OUTPUT_PATH, file_name)
    logger.info("Saving LDA model to file: %s" % fpath)
    lda.save(fpath)


def copyFigures():
    logger = logging.getLogger("__name__")

    files = glob.glob(os.path.join(OUTPUT_PATH, '*.png'))
    # Define the path of the shared folder
    if 'AZUREML_NATIVE_SHARE_DIRECTORY' in os.environ:
        shared_path = os.environ['AZUREML_NATIVE_SHARE_DIRECTORY']
        for f in files:
            file_name = os.path.basename(f)
            copyfile(os.path.join(f), os.path.join(shared_path, file_name))
            logger.info("Copied image file: %s" % f)
    else:
        raise ValueError("Cannot find 'AZUREML_NATIVE_SHARE_DIRECTORY' environment variable")


def _terms_to_counts(terms, multiplier=1000):
    return ' '.join([' '.join(int(multiplier * x[1]) * [x[0]]) for x in terms])


def visualizeTopic(lda, topicID=0, topn=500, multiplier=1000):
    terms = []
    tmp = lda.show_topic(topicID, topn)
    for term in tmp:
        terms.append(term)
    
    # If the version of wordcloud is higher than 1.3, then you will need to set 'collocations' to False.
    # Otherwise there will be word duplicates in the figure. 
    try:
        wordcloud = WordCloud(width=800, height=600, max_words=10000, collocations=False).generate(_terms_to_counts(terms, multiplier))
    except:
        wordcloud = WordCloud(width=800, height=600, max_words=10000).generate(_terms_to_counts(terms, multiplier))
    fig = plt.figure()
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.title("Topic %d" % topicID)
    output_path = os.path.join(OUTPUT_PATH, 'topic_' + str(topicID) + '_' + datetime.now().strftime("%Y_%m_%d_%H_%M_%S") + '.png')
    plt.savefig(output_path)



"""
main
"""
if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s : %(name)s : %(levelname)s : %(message)s', level=logging.INFO)

    run_step3(topicConfig=[10, 20, 30, 40, 50], test_ratio=0.005, saveModel=False, 
                coherence_types=['u_mass', 'c_v', 'c_uci', 'c_npmi'])
    copyFigures()
