import numpy as np
import pandas as pd
import re
import math
import gensim
import time
import sys
import logging
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scipy.sparse as sp

from multiprocessing import cpu_count
from gensim import corpora, models
from gensim.models.coherencemodel import CoherenceModel
from operator import itemgetter
from azureml.logging import get_azureml_logger
from collections import namedtuple
from datetime import datetime
from gensim.matutils import dirichlet_expectation
from scipy.special import gammaln
from .configs import *
from .preprocessText import *

try:
    # try importing from here if older scipy is installed
    from scipy.maxentropy import logsumexp
except ImportError:
    # maxentropy has been removed in recent releases, logsumexp now in misc
    from scipy.misc import logsumexp


class TopicModeler(object):
    """
    Topic modeling on corpus using Latent Dirichlet Allocation (LDA)
    """
    def __init__(self, textData=None, stopWordFile='', minWordCount=5, minDocCount=2, 
                    maxDocFreq=0.25, workers=1, numTopics=50, numIterations=100, passes=1, 
                    chunksize=2000, random_state=None, test_ratio=0.005):
        logger = logging.getLogger(__name__)

        # initialize the run logger
        self.run_logger = get_azureml_logger()
        self.run_logger.log('amlrealworld.document-collection-analysis.topicModeling', 'true')

        if not textData or not isinstance(textData, list):
            raise ValueError("Text data should be non-empty and in the format of list.")

        # The minimum word count in all documents
        self.minWordCount = minWordCount
        # The minimum count of documents that contain a specific word
        self.minDocCount = minDocCount
        # The maximum document frequency that contain a specific word
        self.maxDocFreq = maxDocFreq

        if workers > cpu_count() or workers <= 0:
            logger.warning("Worker number %d is greater than number of cores: %d, reduced it to the number of cores" % (workers, cpu_count()))
            self.workers = cpu_count()
        else:
            self.workers = workers

        self.numTopics = numTopics
        self.numIterations = numIterations
        self.passes = passes
        self.chunksize = chunksize
        self.random_state = random_state
        self.test_ratio = test_ratio

        if not stopWordFile:
            raise ValueError("Need to provide the file name of the stop word list")
        
        stopWordPath = get_shared_file_path(stopWordFile)
        if not os.path.exists(stopWordPath):
            download_file_from_blob(stopWordFile)
        
        self.stopWordHash = LoadListAsHash(stopWordPath)
        self.vocabHash = self.CreateVocabForTopicModeling(textData, self.stopWordHash)
        self.tokenizedDocs = self.TokenizeText(textData)
        self.id2token = None
        self.token2id = None
        self.BuildDictionary(self.tokenizedDocs)
        self.corpus = self.BuildCorpus(self.tokenizedDocs)

        # global variable for run log
        self.topics_list = []
        self.u_mass_list = []
        self.c_v_list = []
        self.c_uci_list = []
        self.c_npmi_list = []
        self.perplexity_list = []
        self.word_bound_list = []


    
    def CreateVocabForTopicModeling(self, textData, stopWordHash):
        """
        Create the vocabulary used for topic modeling.
        """
        logger = logging.getLogger(__name__)

        if not textData or not isinstance(textData, list):
            raise ValueError("Text data should be non-empty and in the format of list.")
        if stopWordHash is None or not isinstance(stopWordHash, dict):
            raise ValueError("The stop word should be non-empty and in the format of dictionary.")

        numDocs = len(textData)
        # The global word counts
        globalWordCountHash = {}
        # The word counts in all documents, each doc count 1
        globalDocCountHash = {}

        logger.info("Counting words in the corpus...")
        for textLine in textData:
            docWordCountHash = {}
            textLine = str(textLine).strip()
            for word in textLine.split():
                globalWordCountHash[word] = globalWordCountHash.get(word, 0) + 1
                if word not in docWordCountHash:
                    docWordCountHash[word] = 1
                    globalDocCountHash[word] = globalDocCountHash.get(word, 0) + 1

        logger.info("globalWordCountHash size = %d" % len(globalDocCountHash))
        logger.info("globalDocCountHash size = %d" % len(globalDocCountHash))

        vocabCount = 0
        vocabHash = {}
        excStopword = 0
        excNonalphabetic = 0
        excMinwordcount = 0
        excNotindochash = 0
        excMindoccount = 0
        excMaxdocfreq =0

        logger.info("Building vocabulary...")
        for word in globalWordCountHash.keys():
            if word in stopWordHash:
                excStopword += 1
            elif not re.search(r'[a-zA-Z]', word, 0):
                excNonalphabetic += 1
            elif globalWordCountHash[word] < self.minWordCount:
                excMinwordcount += 1
            elif globalDocCountHash[word] < self.minDocCount:
                excMindoccount += 1
            elif float(globalDocCountHash[word]) / float(numDocs) > self.maxDocFreq:
                excMaxdocfreq += 1
            else:
                vocabCount += 1
                vocabHash[word] = globalWordCountHash[word]
        logger.info("Excluded %d stop words" % excStopword)
        logger.info("Excluded %d non-alphabetic words" % excNonalphabetic)
        logger.info("Excluded %d words below word count threshold (%d)" % (excMinwordcount, self.minWordCount))
        logger.info("Excluded %d words below document count threshold (%d)" % (excMindoccount, self.minDocCount))
        logger.info("Excluded %d words above max document frequency (%.2f)" % (excMaxdocfreq, self.maxDocFreq))
        logger.info("Final Vocab size: %d words" % vocabCount)

        return vocabHash


    def TokenizeText(self, textData):
        """
        Tokenizing the full text string for each document into list of tokens
        Any token that is not in the pre-defined set of acceptable vocabulary words is excluded
        """
        logger = logging.getLogger(__name__)

        if not textData or not isinstance(textData, list):
            raise ValueError("Text data should be non-empty and in the format of list.")
        if self.vocabHash is None or not isinstance(self.vocabHash, dict):
            raise ValueError("The Vocab hash should be non-empty and in the format of dictionary.")

        logger.info("Text data size for tokenizing: %d" % len(textData))
        tokenizedText = []
        numTokens = 0
        for textLine in textData:
            textLine = str(textLine)
            line_tokens = [token for token in textLine.split() if token in self.vocabHash]
            tokenizedText.append(line_tokens)
            numTokens += len(line_tokens)
        logger.info("Total number of retained tokens: %d" % numTokens)
        return tokenizedText


    def BuildDictionary(self, tokenizedDocs):
        """
        Build ID to token dictionary from the tokenized documents
        """
        logger = logging.getLogger(__name__)

        if not tokenizedDocs or not isinstance(tokenizedDocs, list):
            raise ValueError("Tokenized text data should be non-empty and in the format of list.")

        self.id2token = corpora.Dictionary(tokenizedDocs)
        self.token2id = self.id2token.token2id
        logger.info("Size of token2id dictionary: %d" % len(self.token2id))

    
    def BuildCorpus(self, tokenizedDocs):
        logger = logging.getLogger(__name__)

        if not tokenizedDocs or not isinstance(tokenizedDocs, list):
            raise ValueError("Tokenized text data should be non-empty and in the format of list.")

        # Convert document (a list of words) into the bag-of-words format = list of (token_id, token_count) 2-tuples.
        corpus =[self.id2token.doc2bow(tokens) for tokens in tokenizedDocs]
        logger.info("Build corpus finished")
        return corpus

    
    def TrainLDA(self, saveModel=True):
        """
        Train an LDA model
        """
        logger = logging.getLogger(__name__)

        if self.id2token is None:
            raise ValueError("Need to build the corpus dictionary first by calling BuildDictionary() function.")
        if self.corpus is None:
            raise ValueError("Need to build corpus first by calling BuildCorpus() function.")

        # Train a LDA model
        logger.info("Start training the LDA model...")
        self.topics_list.append(self.numTopics)
        lda = models.ldamulticore.LdaMulticore(self.corpus,
                        id2word=self.id2token, 
                        num_topics=self.numTopics,
                        iterations=self.numIterations,
                        workers=self.workers,
                        passes=self.passes,
                        chunksize=self.chunksize,
                        random_state=self.random_state,
                        offset=1.0,
                        decay=0.5
                )
        logger.info("Train LDA model finished")

        if saveModel:
            name_seps = LDA_FILE.split('.')
            file_name = '_'.join([name_seps[0], str(MAX_NUM_PHRASE), str(self.numTopics), str(self.numIterations), str(self.passes), str(self.chunksize)])
            file_name += '.' + name_seps[1]

            fpath = get_shared_file_path(file_name)
            logger.info("Saving LDA model to file: %s" % fpath)
            lda.save(fpath)
        return lda


    def Transform(self, model, textData, saveFile=False):
        """
        Transfrom list of documents into a topic probability matrix
        """
        logger = logging.getLogger(__name__)

        if not textData or not isinstance(textData, list):
            raise ValueError("Text data should be non-empty and in the format of list.")
        if self.id2token is None:
            raise ValueError("Need to build the token dictionary first")
        
        # Build the corpus of the text documents
        tokenizedDocs = self.TokenizeText(textData)
        corpus = self.BuildCorpus(tokenizedDocs)
        logger.info("Build corpus of the text document finished")

        # To retrieve all topics and their probabilities we must set the 
        # LDA minimum probability setting to zero
        model.minimum_probability = 0

        docTopicprobs = np.zeros((len(corpus), model.num_topics))
        logger.info("Transforming %d documents with %d topics" % (len(corpus), model.num_topics))
        for docID in range(len(corpus)):
            for topicProb in model[corpus[docID]]:
                docTopicprobs[docID, topicProb[0]] = topicProb[1]

        if saveFile:
            name_seps = DOC_TOPIC_PROB_FILE.split('.')
            file_name = '_'.join([name_seps[0], str(MAX_NUM_PHRASE), str(self.numTopics), str(self.numIterations), str(self.passes), str(self.chunksize)])
            file_name += '.' + name_seps[1]

            fpath = get_shared_file_path(file_name)
            logger.info("Saving the document topic probability matrix to file: %s" % fpath)
            np.save(fpath, docTopicprobs)
        return docTopicprobs
    
    
    def EvaluateCoherence(self, model, coherence='c_v'):
        """
        Evaluate the coherence of the LDA model. 
        The core estimation code is based on the onlineldavb.py script by M. Hoffman, 
        see Hoffman, Blei, Bach: Online Learning for Latent Dirichlet Allocation, NIPS 2010.
        """
        logger = logging.getLogger(__name__)

        if isinstance(coherence, str):
            coherence = [coherence]
        elif not isinstance(coherence, list):
            raise ValueError("The coherence method should be either a list or a specific type")
        
        values = dict()
        supported = set(['u_mass', 'c_v', 'c_uci', 'c_npmi'])
        for ctype in coherence:
            if ctype not in supported:
                logger.warning("Coherence evaluation for type %s is not supported, ignored. Only support types %s" % (ctype, str(supported)))
                continue
            cm = CoherenceModel(model=model, texts=self.tokenizedDocs, corpus=self.corpus, 
                            dictionary=self.id2token, coherence=ctype, topn=10)
            values[ctype] = cm.get_coherence()

            # Add run log for the coherence values
            if ctype == 'u_mass':
                self.u_mass_list.append(values[ctype])
            elif ctype == 'c_v':
                self.c_v_list.append(values[ctype])
            elif ctype == 'c_uci':
                self.c_uci_list.append(values[ctype])
            else:
                self.c_npmi_list.append(values[ctype])
            logger.info("Coherence type: %s, coherence value = %.6f" % (ctype, values[ctype]))
        return values


    def EvaluatePerplexity(self, model):
        """
        Evaluate the perplexity given a test corpus to find the best number of
        topics need to learn
        """
        logger = logging.getLogger(__name__)
        logger.info("Evaluate perplexity on a held-out corpus")
        
        lencorpus = len(self.corpus)
        test_corpus = self.corpus[int((1.0 - self.test_ratio) * lencorpus):]
        lentest = len(test_corpus)
        number_of_words = sum(cnt for doc in test_corpus for _, cnt in doc)
        subsample_ratio = 1.0 * lencorpus / lentest

        perwordbound = model.bound(test_corpus, subsample_ratio=subsample_ratio) / (subsample_ratio * number_of_words)
        perplex = np.exp2(-perwordbound)
        self.perplexity_list.append(perplex)
        self.word_bound_list.append(perwordbound)
        logger.info("Per word bound value: %.4f, estimated based on a held-out corpus of %d documents with %d words" % (perwordbound, lentest, number_of_words))
        logger.info("Perplexity value: %.4f, estimated based on a held-out corpus of %d documents with %d words" % (perplex, lentest, number_of_words))
        return {'perplexity': perplex, 'per_word_bound': perwordbound}


    def CollectRunLog(self):
        # add all coherence values into the log
        if self.u_mass_list:
            self.run_logger.log('u_mass Coherence', self.u_mass_list)
        if self.c_v_list:
            self.run_logger.log('c_v Coherence', self.c_v_list)
        if self.c_uci_list:
            self.run_logger.log('c_uci Coherence', self.c_uci_list)
        if self.c_npmi_list:
            self.run_logger.log('c_npmi Coherence', self.c_npmi_list)

        # add perplexity evaluation results
        if self.perplexity_list:
            self.run_logger.log('Perplexity', self.perplexity_list)
        if self.word_bound_list:
            self.run_logger.log('Per Word Bound', self.word_bound_list)


    def PlotLineFigure(self, x_values, y_values, x_label, y_label, title):
        plt.style.use('ggplot')
        fig = plt.figure()

        plt.plot(x_values, y_values, 'go-')
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(title)
        
        output_path = os.path.join(OUTPUT_PATH, title.replace(' ', '_') + '_' + datetime.now().strftime("%Y_%m_%d_%H_%M_%S") + '.png')
        plt.savefig(output_path)


    def PlotRunLog(self):
        # Plot the coherence and perplexity values
        if self.u_mass_list:
            self.PlotLineFigure(self.topics_list, self.u_mass_list, 'Number of Topics', 'Coherence', 'u_mass Coherence')
        if self.c_v_list:
            self.PlotLineFigure(self.topics_list, self.c_v_list, 'Number of Topics', 'Coherence', 'c_v Coherence')
        if self.c_uci_list:
            self.PlotLineFigure(self.topics_list, self.c_uci_list, 'Number of Topics', 'Coherence', 'c_uci Coherence')
        if self.c_npmi_list:
            self.PlotLineFigure(self.topics_list, self.c_npmi_list, 'Number of Topics', 'Coherence', 'c_npmi Coherence')
        
        if self.perplexity_list:
            self.PlotLineFigure(self.topics_list, self.perplexity_list, 'Number of Topics', 'Perplexity', 'Perplexity Value')
        if self.word_bound_list:
            self.PlotLineFigure(self.topics_list, self.word_bound_list, 'Number of Topics', 'Bound', 'Per-word Bound Value')
        

