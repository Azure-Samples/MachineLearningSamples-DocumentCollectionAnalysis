import nltk
import pandas as pd
import re
import sys
import os
import logging
import urllib.request

from nltk import tokenize
from .configs import *
from azureml.logging import get_azureml_logger



def download_file_from_blob(filename):
    logger = logging.getLogger(__name__)

    # Define the path of the shared folder
    if 'AZUREML_NATIVE_SHARE_DIRECTORY' in os.environ:
        shared_path = os.environ['AZUREML_NATIVE_SHARE_DIRECTORY']
    else:
        logger.warning("Cannot find 'AZUREML_NATIVE_SHARE_DIRECTORY' environment variable, using '%s' instead" % OUTPUT_PATH)
        shared_path = OUTPUT_PATH
    save_path = os.path.join(shared_path, filename)

    url = STORAGE_CONTAINER + filename
    logger.info("Downloading file from URL: %s" % url)
    urllib.request.urlretrieve(url, save_path)
    logger.info("Downloaded file '%s' from blob storage to path '%s'" % (filename, save_path))


def get_shared_file_path(filename):
    logger = logging.getLogger(__name__)

    # Define the path of the shared folder
    if 'AZUREML_NATIVE_SHARE_DIRECTORY' in os.environ:
        shared_path = os.environ['AZUREML_NATIVE_SHARE_DIRECTORY']
    else:
        logger.warning("Cannot find 'AZUREML_NATIVE_SHARE_DIRECTORY' environment variable, using '%s' instead" % OUTPUT_PATH)
        shared_path = OUTPUT_PATH
    file_path = os.path.join(shared_path, filename)
    return file_path


# Function for loading list into dictionary hash tables
def LoadListAsHash(filename):
    logger = logging.getLogger(__name__)

    listHash = {}
    with open(filename, 'r', encoding='utf-8') as fp:
        # Read in lines one by one stripping away extra spaces, 
        # leading spaces, and trailing spaces and inserting each
        # cleaned up line into a hash table
        re1 = re.compile(' +')
        re2 = re.compile('^ +| +$')
        for stringIn in fp.readlines():
            term = re2.sub("", re1.sub(" ", stringIn.strip()))
            if term != '':
                listHash[term] = 1
    logger.info("Loaded list into dictionary from file: %s" % filename)
    return listHash


# Get the data from Data Source
# It returns a Pandas DataFrame
def getData():
    logger = logging.getLogger(__name__)
    logger.info("Getting data from Data Source")

    # Define the path of the shared folder
    if 'AZUREML_NATIVE_SHARE_DIRECTORY' in os.environ:
        shared_path = os.environ['AZUREML_NATIVE_SHARE_DIRECTORY']
    else:
        logger.warning("Cannot find 'AZUREML_NATIVE_SHARE_DIRECTORY' environment variable, using '%s' instead" % OUTPUT_PATH)
        shared_path = OUTPUT_PATH

    data_file = os.path.join(shared_path, DATASET_FILE)
    blacklist_file = os.path.join(shared_path, BLACK_LIST_FILE)
    function_words_file = os.path.join(shared_path, FUNCTION_WORDS_FILE)

    if not os.path.exists(data_file):
        download_file_from_blob(DATASET_FILE)
    if not os.path.exists(blacklist_file):
        download_file_from_blob(BLACK_LIST_FILE)
    if not os.path.exists(function_words_file):
        download_file_from_blob(FUNCTION_WORDS_FILE)

    df = pd.read_csv(data_file, sep='\t', encoding='ISO-8859-1')
    logger.info("Retrieved data frame shape: %d, %d" % (df.shape[0], df.shape[1]))
    return df


def CleanAndSplitText(textDataFrame, idColumnName='ID', textColumnName='Text', saveDF=False):
    aml_logger = get_azureml_logger()   # logger writes to AMLWorkbench runtime view
    aml_logger.log('amlrealworld.document-collection-analysis.preprocessText', 'true')

    logger = logging.getLogger(__name__)

    # Need to download the 'punkt' model for breaking text
    # strings into individual sentences
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        logger.debug("Need to download the 'punkt' model from NLTK")
        nltk.download('punkt')
        logger.debug("Downloading 'punkt' model done.")
    
    logger.info("Clean and split the raw text into sentences")

    textDataOut = [] 
    # This regular expression is for section headers in the bill summaries that we wish to ignore
    reHeaders = re.compile(r" *TABLE OF CONTENTS:? *"
                           "| *Title [IVXLC]+:? *"
                           "| *Subtitle [A-Z]+:? *"
                           "| *\(Sec\. \d+\) *")

    # This regular expression is for punctuation that we wish to clean out
    # We also will split sentences into smaller phrase like units using this expression
    rePhraseBreaks = re.compile("[\"\!\?\)\]\}\,\:\;\*\-]*\s+\([0-9]+\)\s+[\(\[\{\"\*\-]*"                             
                                "|[\"\!\?\)\]\}\,\:\;\*\-]+\s+[\(\[\{\"\*\-]*"
                                "|\.\.+"
                                "|\s*\-\-+\s*"
                                "|\s+\-\s+"
                                "|\:\:+"
                                "|\s+[\/\(\[\{\"\-\*]+\s*"
                                "|[\,!\?\"\)\(\]\[\}\{\:\;\*](?=[a-zA-Z])"
                                "|[\"\!\?\)\]\}\,\:\;]+[\.]*$"
                             )
    
    # Regex for underbars
    regexUnderbar = re.compile('_')
    
    # Regex for space
    regexSpace = re.compile(' +')
 
    # Regex for sentence final period
    regexPeriod = re.compile("\.$")

    # Iterate through each document and do:
    #    (1) Split documents into sections based on section headers and remove section headers
    #    (2) Split the sections into sentences using NLTK sentence tokenizer
    #    (3) Further split sentences into phrasal units based on punctuation and remove punctuation
    #    (4) Remove sentence final periods when not part of a abbreviation 

    for i in range(len(textDataFrame)):
        # Extract one document from frame
        docID = textDataFrame[idColumnName][i]
        docText = textDataFrame[textColumnName][i] 

        # Set counter for output line count for this document
        lineIndex=0;

        # Split document into sections by finding sections headers and splitting on them 
        sections = reHeaders.split(str(docText))
        
        for section in sections:
            # Split section into sentence using NLTK tokenizer 
            sentences = tokenize.sent_tokenize(section)
            
            for sentence in sentences:
                # Split each sentence into phrase level chunks based on punctuation
                textSegs = rePhraseBreaks.split(sentence)
                numSegs = len(textSegs)
                
                for j in range(0, numSegs):
                    if len(textSegs[j]) > 0:
                        # Convert underbars to spaces 
                        # Underbars are reserved for building the compound word phrases                   
                        textSegs[j] = regexUnderbar.sub(" ", textSegs[j])
                    
                        # Split out the words so we can specially handle the last word
                        words = regexSpace.split(textSegs[j])
                        phraseOut = ""
                        # If the last word ends in a period then remove the period
                        words[-1] = regexPeriod.sub("", words[-1])
                        # If the last word is an abbreviation like "U.S."
                        # then add the word final perios back on
                        if "\." in words[-1]:
                            words[-1] += "."
                        phraseOut = " ".join(words)  

                        textDataOut.append([docID, lineIndex, phraseOut])
                        lineIndex += 1
    # Convert to Pandas DataFrame 
    frameOut = pd.DataFrame(textDataOut, columns=['DocID', 'DocLine', 'CleanedText'])
    logger.debug("Returned clean DataFrame shape: %d, %d" % (frameOut.shape[0], frameOut.shape[1]))

    if saveDF:
        logger.info("Saving the cleaned DataFrame in file: %s" % CLEANED_DATA_FILE_NAME)
        cleanedDataFile = get_shared_file_path(CLEANED_DATA_FILE_NAME)
        frameOut.to_csv(cleanedDataFile, sep='\t', index=False)
    else:
        logger.info("The cleaned and sentenced text data is not being saved.")

    return frameOut


# Load the processed text data file.
# If filename is None, file RECONSTITUTED_TEXT_FILE will be used, otherwise
# the function will first try to find the file in the shared folder. And if it 
# failed to do so, it then tries to download the file from Blob storage.
# If numPhrases is None, the file named filename will be used, otherwise
# it will automatically find the file with the corresponding number of phrases. 
def loadProcessedTextData(filename=None, numPhrases=0, sep='\t'):
    logger = logging.getLogger(__name__)

    # Specified the data file name, try to load it from the shared folder
    # or download it from Blob storage to the shared folder
    if filename is not None:
        fpath = get_shared_file_path(filename)
        if not os.path.exists(fpath):
            logger.debug("Cannot find file: %s under the shared folder" % filename)
            download_file_from_blob(filename)
        dataDF = pd.read_csv(fpath, sep=sep, encoding='ISO-8859-1')
        logger.info("Total documents in corpus: %d" % len(dataDF))
        return dataDF
    # No data file name specified, read the file defined by RECONSTITUTED_TEXT_FILE 
    else:
        if numPhrases == 0:
            # Use the default file name defined by RECONSTITUTED_TEXT_FILE
            fpath = get_shared_file_path(RECONSTITUTED_TEXT_FILE)
            if not os.path.exists(fpath):
                logger.error("Cannot find file: %s under the shared folder" % RECONSTITUTED_TEXT_FILE)
                return pd.DataFrame()
        else:
            # Use the file name with the corresponding number of phrases
            fnameSeps = RECONSTITUTED_TEXT_FILE.split('.')
            fpath = get_shared_file_path(fnameSeps[0] + '_phrase_' + str(numPhrases) + '.' + fnameSeps[1])
            if not os.path.exists(fpath):
                logger.error("Cannot find file: %s under the shared folder" % RECONSTITUTED_TEXT_FILE)
                return pd.DataFrame()
        logger.debug("Loading processed text file: %s" % fpath)
        dataDF = pd.read_csv(fpath, sep=sep, encoding='ISO-8859-1')
        logger.info("Total documents in corpus: %d" % len(dataDF))
        return dataDF


# Load the mapping of lower-cased vocabulary items to their most
# common surface form which is produced during the phrase learning step
def loadVocabToSurfaceForms(filename=None, numPhrases=0, sep='\t'):
    logger = logging.getLogger(__name__)

    fpath = ''
    if filename is not None and filename != '':
        fpath = get_shared_file_path(filename)
        if not os.path.exists(fpath):
            logger.debug("Cannot find file: %s under the shared folder" % filename)
            download_file_from_blob(filename)
    else:
        if numPhrases == 0:
            # Use the default file name defined by RECONSTITUTED_TEXT_FILE
            fpath = get_shared_file_path(SURFACE_MAPPING_FILE)
        else:
            # Use the file name with the corresponding number of phrases
            fnameSeps = SURFACE_MAPPING_FILE.split('.')
            fpath = get_shared_file_path(fnameSeps[0] + '_phrase_' + str(numPhrases) + '.' + fnameSeps[1])
    
    if not os.path.exists(fpath):
        raise ValueError("Mapping of vocabulary items to their most common surface form does not exists")
    
    vocabToSurfaceFormHash = {}
    with open(fpath, encoding='utf-8') as fp:
        for stringIn in fp.readlines():
            fields = stringIn.strip().split(sep)
            if len(fields != 2):
                logger.warning("Bad line in surface form mapping file: %s, ignored" % stringIn)
            elif field[0] == '' or fields[1] == '':
                logger.warning("Bad line in surface form mapping file: %s, ignored" % stringIn)
            else:
                vocabToSurfaceFormHash[fields[0]] = fields[1]
    return vocabToSurfaceFormHash

