import pandas as pd
import re
import math
import concurrent.futures
import time
import gc
import os
import logging

from operator import itemgetter
from collections import namedtuple
from datetime import datetime
from multiprocessing import cpu_count
from azureml.logging import get_azureml_logger
from .configs import *
from .preprocessText import get_shared_file_path, LoadListAsHash



# This is the function that used to define how to compute Ngram stats
# This function will be executed in parallel as process pool executor
def ComputeNgramStatsJob(textList, functionwordHash, blacklistHash, reValidWord, jobId):
    logger = logging.getLogger(__name__)
    logger.debug("ComputeNgramStatsJob() - Starting batch execution %d" % (jobId + 1))
    startTS = datetime.now()

    # Create an array to store the total count of all ngrams up to 4-grams
    # Array element 0 is unused, element 1 is unigrams, element 2 is bigrams, etc.
    ngramCounts = [0] * 5;
    
    # Create a list of structures to tabulate ngram count statistics
    # Array element 0 is the array of total ngram counts,
    # Array element 1 is a hash table of individual unigram counts
    # Array element 2 is a hash table of individual bigram counts
    # Array element 3 is a hash table of individual trigram counts
    # Array element 4 is a hash table of individual 4-gram counts
    ngramStats = [ngramCounts, {}, {}, {}, {}]
    
    numLines = len(textList)
    logger.debug("ComputeNgramStatsJob() - Batch %d, received %d lines data" % (jobId+1, numLines))
    
    for i in range(0, numLines):
        # Split the text line into an array of words
        wordArray = textList[i].strip().split()
        numWords = len(wordArray)
        
        # Create an array marking each word as valid or invalid
        validArray = [reValidWord.match(word) != None for word in wordArray]
        
        # Tabulate total raw ngrams for this line into counts for each ngram bin
        # The total ngrams counts include the counts of all ngrams including those
        # that we won't consider as parts of phrases
        for j in range(1, 5):
            if j <= numWords:
                ngramCounts[j] += numWords - j + 1
        
        # Collect counts for viable phrase ngrams and left context sub-phrases
        for j in range(0, numWords):
            word = wordArray[j]

            # Only bother counting the ngrams that start with a valid content word
            # i.e., valids words not in the function word list or the black list
            if ((word not in functionwordHash) and (word not in blacklistHash) and validArray[j]):
                # Initialize ngram string with first content word and add it to unigram counts
                ngramSeq = word 
                if ngramSeq in ngramStats[1]:
                    ngramStats[1][ngramSeq] += 1
                else:
                    ngramStats[1][ngramSeq] = 1

                # Count valid ngrams from bigrams up to 4-grams
                stop = False
                k = 1
                while (k < 4) and (j + k < numWords) and not stop:
                    n = k + 1
                    nextNgramWord = wordArray[j + k]
                    # Only count ngrams with valid words not in the blacklist
                    if (validArray[j + k] and nextNgramWord not in blacklistHash):
                        ngramSeq += " " + nextNgramWord
                        if ngramSeq in ngramStats[n]:
                            ngramStats[n][ngramSeq] += 1
                        else:
                            ngramStats[n][ngramSeq] = 1 
                        k += 1
                        if nextNgramWord not in functionwordHash:
                            # Stop counting new ngrams after second content word in 
                            # ngram is reached and ngram is a viable full phrase
                            stop = True
                    else:
                        stop = True
    endTS = datetime.now()
    delta_t = (endTS - startTS).total_seconds()
    logger.debug("ComputeNgramStatsJob() - Batch %d finished, time elapsed: %0.1f seconds" % (jobId+1, delta_t))
    return ngramStats


# The job function to apply phrase rewrites to text data
# This function will be called by function ApplyPhraseRewrites()
def phraseRewriteJob(ngramRegex, text, ngramRewriteHash, jobId):
    logger = logging.getLogger(__name__)
    logger.debug("phraseRewriteJob() - Starting phrase rewrites batch execution %d" % (jobId + 1))

    startTS = datetime.now()
    retList = []
    
    for i in range(len(text)):
        # The regex substituion looks up the output string rewrite
        # in the hash table for each matched input phrase regex
        retList.append(ngramRegex.sub(lambda mo: ngramRewriteHash[mo.string[mo.start():mo.end()]], text[i]))
    
    endTS = datetime.now()
    delta_t = (endTS - startTS).total_seconds()
    logger.debug("Phrase rewrites batch %d finished, batch size: %d, time elapsed: %f seconds" % (jobId + 1, i, delta_t))
    return retList, jobId


# Apply the phrase rules to new documents, the learnedPhrases must be a list
def ApplyPhraseRewritesInPlace(textFrame, textColumnName, learnedPhrases):
    logger = logging.getLogger(__name__)

    # Make sure we have phrase to add
    if not learnedPhrases or not isinstance(learnedPhrases, list):
        logger.error("The learned phrases is empty or not a list - no phrases being applied to text data")
        return

    numPhraseRules = len(learnedPhrases)
    
    # Get text data column from frame
    textData = textFrame[textColumnName]
    numLines = len(textData)
    
    # Add leading and trailing spaces to make regex matching easier
    for i in range(numLines):
        textData[i] = " " + textData[i] + " "  

    # Precompile the regex for finding spaces in ngram phrases
    regexSpace = re.compile(' ')

    # Initialize some bookkeeping variables

    # Iterate through full set of phrases to find sets of 
    # non-conflicting phrases that can be apply simultaneously
    index = 0
    outerStop = False
    while not outerStop:
        # Create empty hash tables to keep track of phrase overlap conflicts
        leftConflictHash = {}
        rightConflictHash = {}
        prevConflictHash = {}
    
        # Create an empty hash table collecting the next set of rewrite rules
        # to be applied during this iteration of phrase rewriting
        phraseRewriteHash = {}
    
        # Progress through phrases until the next conflicting phrase is found
        innerStop = 0
        numPhrasesAdded = 0
        while not innerStop:
            # Get the next phrase to consider adding to the phrase list
            nextPhrase = learnedPhrases[index]            
            
            # Extract the left and right sides of the phrase to use
            # in checks for phrase overlap conflicts
            ngramArray = nextPhrase.split()
            leftWord = ngramArray[0]
            rightWord = ngramArray[-1] 

            # Stop if we reach any phrases that conflicts with earlier phrases in this iteration
            # These ngram phrases will be reconsidered in the next iteration
            if ((leftWord in leftConflictHash) or (rightWord in rightConflictHash) 
                or (leftWord in prevConflictHash) or (rightWord in prevConflictHash)): 
                innerStop = True
                
            # If no conflict exists then add this phrase into the list of phrase rewrites     
            else: 
                # Create the output compound word version of the phrase
                outputPhrase = regexSpace.sub("_", nextPhrase);
                
                # Keep track of all context words that might conflict with upcoming
                # propose phrases (even when phrases are skipped instead of added)
                leftConflictHash[rightWord] = 1
                rightConflictHash[leftWord] = 1
                prevConflictHash[outputPhrase] = 1           
                
                # Add extra space to input an output versions of the current phrase 
                # to make the regex rewrite easier
                outputPhrase = " " + outputPhrase
                lastAddedPhrase = " " + nextPhrase
                
                # Add the phrase to the rewrite hash
                phraseRewriteHash[lastAddedPhrase] = outputPhrase
                
                # Increment to next phrase
                index += 1
                numPhrasesAdded  += 1
    
                # Stop if we've reached the end of the phrases list
                if index >= numPhraseRules:
                    innerStop = True
                    outerStop = True

        # Now do the phrase rewrites over the entire set of text data
        # Compile a single regex rule from the collected set of phrase rewrites for this iteration
        regexPhrase = re.compile(r'%s(?= )' % "|".join(map(re.escape, phraseRewriteHash.keys())))
        
        # Apply the regex over the full data set
        for i in range(numLines):
            # The regex substitution looks up the output string rewrite  
            # in the hash table for each matched input phrase regex
            textData[i] = regexPhrase.sub(lambda mo: phraseRewriteHash[mo.string[mo.start():mo.end()]], textData[i]) 
    
    # Remove the space padding at the start and end of each line
    regexSpacePadding = re.compile('^ +| +$')
    for i in range(len(textData)):
        textData[i] = regexSpacePadding.sub("", textData[i])
    


class PhraseLearner(object):
    """
    The class that used to learn phrases from a collection of documents
    """
    def __init__(self, textFrame=None, textCol="", maxNumPhrases=25000, 
                    maxPhrasesPerIter=500, maxPhraseLength=7, minInstanceCount=5):
        logger = logging.getLogger(__name__)

        # initialize the run logger
        self.run_logger = get_azureml_logger()
        self.run_logger.log('amlrealworld.document-collection-analysis.phraseLearning', 'true')

        self.textFrame = textFrame
        self.textCol = textCol

        # Load the black list of words
        # This is a precreated hash table containing the list 
        # of black list words to be ignored during phrase learning
        self.black_list = get_shared_file_path(BLACK_LIST_FILE)
        self.blacklistHash = LoadListAsHash(self.black_list)

        # Load the function words
        # This is a precreated hash table containing the list 
        # of function words used during phrase learning
        self.function_words = get_shared_file_path(FUNCTION_WORDS_FILE)
        self.functionwordHash = LoadListAsHash(self.function_words)

        # Maximum number of phrases to learn
        # If you want to test the code out quickly then set this to a small
        # value (e.g. 100) and set verbose to true when running the quick test
        self.maxNumPhrases = maxNumPhrases

        # Maximum number of phrases to learn per iteration 
        # Increasing this number may speed up processing but will affect the ordering of the phrases 
        # learned and good phrases could be by-passed if the maxNumPhrases is set to a small number
        self.maxPhrasesPerIter = maxPhrasesPerIter

        # Maximum number of words allowed in the learned phrases 
        self.maxPhraseLength = maxPhraseLength

        # Minimum number of times a phrase must occur in the data to 
        # be considered during the phrase learning process
        self.minInstanceCount = minInstanceCount

        # The learned phrases
        self.learnedPhrases = []

        # Lower case the raw text column and save in a new column
        if self.textFrame is not None and self.textCol != '':
            self.LowerText(self.textFrame, self.textCol)
        else:
            logger.error("Create an instance with Null text DataFrame, please call self.LowerText() to convert text to lowercase.")
    

    # Function to convert the raw text into lower case and save in a new column
    def LowerText(self, textFrame, textCol):
        logger = logging.getLogger(__name__)
        
        lowercaseText = list()
        for textLine in self.textFrame[self.textCol]:
            lowercaseText.append(str(textLine).lower())
        self.textFrame['LowercaseText'] = lowercaseText
        logger.info("Convert the text column into lower case finished.")


    # This is Step 1 for each iteration of phrase learning
    # We count the number of occurances of all 2-gram, 3-ngram, and 4-gram
    # word sequences 
    def ComputeNgramStats(self, textData, functionwordHash, blacklistHash, numWorkers=1):
        logger = logging.getLogger(__name__)
        logger.debug("Start computing N-gram stats.")

        # Create a regular expression for assessing validity of words
        # for phrase modeling. The expression says words in phrases
        # must either:
        # (1) contain an alphabetic character, or 
        # (2) be the single charcater '&', or
        # (3) be a one or two digit number
        reWordIsValid = re.compile('[A-Za-z]|^&$|^\d\d?$')
        
        # Go through the text data line by line collecting count statistics
        # for all valid n-grams that could appear in a potential phrase
        numLines = len(textData)
        
        # Get the number of CPU to run the tasks
        if numWorkers > cpu_count():
            worker = cpu_count()
        else:
            worker = numWorkers
        logger.debug("ComputeNgramStats Worker number = %d" % worker)
        
        # Get the batch size for each execution job
        # The very last job executor may received more lines of data
        batch_size = int(numLines / worker)
        # logger.info("ComputeNgramStats -- numLines = %d, batch_size = %d" % (numLines, batch_size))
        batchIndexes = range(0, numLines, batch_size)
        
        batch_returns = []
        with concurrent.futures.ProcessPoolExecutor(max_workers=worker) as executor:
            # The job set
            jobs = set()
            # Map the task into multiple batch executions
            for idx in range(worker):
                # The very last job executor
                if idx == (worker - 1):
                    jobs.add(executor.submit(ComputeNgramStatsJob,
                                                textData[batchIndexes[idx]: ], 
                                                functionwordHash, 
                                                blacklistHash,
                                                reWordIsValid,
                                                idx))
                else:
                    jobs.add(executor.submit(ComputeNgramStatsJob,
                                                textData[batchIndexes[idx]:(batchIndexes[idx] + batch_size)], 
                                                functionwordHash, 
                                                blacklistHash,
                                                reWordIsValid,
                                                idx))
            
            # Get results from batch executions
            for job in concurrent.futures.as_completed(jobs):
                try:
                    ret = job.result()
                except Exception as e:
                    logger.error("ComputeNgramStatsJob() - exception while trying to get result from a batch: %s" % e)
                else:
                    batch_returns.append(ret)

        # Reduce the results from batche executions
        # Reuse the first return
        ngramStats = batch_returns[0]
        
        for batch_id in range(1, len(batch_returns)):
            result = batch_returns[batch_id]
            
            # Update the ngram counts
            ngramStats[0] = [x + y for x, y in zip(ngramStats[0], result[0])]
            
            # Update the hash table of ngram counts
            for n_gram in range(1, 5):
                for item in result[n_gram]:
                    if item in ngramStats[n_gram]:
                        ngramStats[n_gram][item] += result[n_gram][item]
                    else:
                        ngramStats[n_gram][item] = result[n_gram][item]
        logger.debug("Finished computing N-gram stats.")
        return ngramStats


    # Function to rank the N-grams
    def RankNgrams(self, ngramStats, functionwordHash, minCount):
        logger = logging.getLogger(__name__)
        logger.debug("Start computing N-gram WPMI and rank them.")

        # Create a hash table to store weighted pointwise mutual 
        # information scores for each viable phrase
        ngramWPMIHash = {}
            
        # Go through each of the ngram tables and compute the phrase scores
        # for the viable phrases
        for n in range(2, 5):
            i = n - 1
            for ngram in ngramStats[n].keys():
                ngramCount = ngramStats[n][ngram]
                if ngramCount >= minCount:
                    wordArray = ngram.split()
                    # If the final word in the ngram is not a function word then
                    # the ngram is a valid phrase candidate we want to score
                    if wordArray[i] not in functionwordHash: 
                        leftNgram = ' '.join(wordArray[:-1])
                        rightWord = wordArray[i]
                        
                        # Compute the weighted pointwise mutual information (WPMI) for the phrase
                        probNgram = float(ngramStats[n][ngram]) / float(ngramStats[0][n])
                        probLeftNgram = float(ngramStats[n-1][leftNgram]) / float(ngramStats[0][n-1])
                        probRightWord = float(ngramStats[1][rightWord]) / float(ngramStats[0][1])
                        WPMI = probNgram * math.log(probNgram / (probLeftNgram * probRightWord));

                        # Add the phrase into the list of scored phrases only if WMPI is positive
                        if WPMI > 0:
                            ngramWPMIHash[ngram] = WPMI
            logger.debug("Finished calculating N-gram = %d" % n)
        logger.debug("Finished calculating N-gram WPMI.")
        # Create a sorted list of the phrase candidates
        rankedNgrams = sorted(ngramWPMIHash, key=ngramWPMIHash.__getitem__, reverse=True)
        logger.debug("Sorted N-gram WPMI.")

        # Force a memory clean-up
        ngramWPMIHash = None
        logger.debug("Ranked N-gram size = %d" % len(rankedNgrams))
        return rankedNgrams


    # Function to apply phrase rewrites which will do it in multiple calls to 
    # function phraseRewriteJob in batches
    def ApplyPhraseRewrites(self, rankedNgrams, textData, learnedPhrases, maxPhrasesToAdd, 
                                maxPhraseLength, numWorkers=1):
        logger = logging.getLogger(__name__)
        logger.debug("Starting phrase rewrites process.")

        # If the number of rankedNgrams coming in is zero then
        # just return without doing anything
        numNgrams = len(rankedNgrams)
        if numNgrams == 0:
            logger.debug("The ranked N-gram list is empty, exit phrase rewrites.")
            return

        # This function will consider at most maxRewrite 
        # new phrases to be added into the learned phrase 
        # list as specified by the calling function
        maxRewrite = maxPhrasesToAdd

        # If the remaining number of proposed ngram phrases is less 
        # than the max allowed, then reset maxRewrite to the size of 
        # the proposed ngram phrases list
        if numNgrams < maxRewrite:
            maxRewrite = numNgrams

        # Create empty hash tables to keep track of phrase overlap conflicts
        leftConflictHash = {}
        rightConflictHash = {}
        
        # Create an empty hash table collecting the set of rewrite rules
        # to be applied during this iteration of phrase learning
        ngramRewriteHash = {}
        
        # Precompile the regex for finding spaces in ngram phrases
        regexSpace = re.compile(' ')

        # Initialize some bookkeeping variables
        numLines = len(textData)  
        numPhrasesAdded = 0
        numConsidered = 0
        lastSkippedNgram = ""
        lastAddedNgram = ""

        # Get the number of CPU to run the tasks
        if numWorkers > cpu_count():
            worker = cpu_count()
        else:
            worker = numWorkers
        logger.debug("ApplyPhraseRewrites Worker number = %d" % worker)
    
        # Collect list of up to maxRewrite ngram phrase rewrites
        stop = False
        index = 0
        while not stop:
            # Get the next phrase to consider adding to the phrase list
            inputNgram = rankedNgrams[index]

            # Create the output compound word version of the phrase
            # The extra space is added to make the regex rewrite easier
            outputNgram = " " + regexSpace.sub("_", inputNgram)

            # Count the total number of words in the proposed phrase
            numWords = len(outputNgram.split("_"))

            # Only add phrases that don't exceed the max phrase length
            if (numWords <= maxPhraseLength):
                # Keep count of phrases considered for inclusion during this iteration
                numConsidered += 1

                # Extract the left and right words in the phrase to use
                # in checks for phrase overlap conflicts
                ngramArray = inputNgram.split()
                leftWord = ngramArray[0]
                rightWord = ngramArray[-1]

                # Skip any ngram phrases that conflict with earlier phrases added
                # These ngram phrases will be reconsidered in the next iteration
                if (leftWord in leftConflictHash) or (rightWord in rightConflictHash): 
                    logger.debug("(%d) Skipping (context conflict): %s" % (numConsidered, inputNgram))
                    lastSkippedNgram = inputNgram
                # If no conflict exists then add this phrase into the list of phrase rewrites     
                else: 
                    logger.debug("(%d) Adding: %s" % (numConsidered, inputNgram))
                    ngramRewriteHash[" " + inputNgram] = outputNgram
                    learnedPhrases.append(inputNgram) 
                    lastAddedNgram = inputNgram
                    numPhrasesAdded += 1
                
                # Keep track of all context words that might conflict with upcoming
                # propose phrases (even when phrases are skipped instead of added)
                leftConflictHash[rightWord] = 1
                rightConflictHash[leftWord] = 1

                # Stop when we've considered the maximum number of phrases per iteration
                if (numConsidered >= maxRewrite):
                    stop = True
            # Increment to next phrase
            index += 1
        
            # Stop if we've reached the end of the ranked ngram list
            if index >= len(rankedNgrams):
                stop = True
        
        # Now do the phrase rewrites over the entire set of text data
        # Compile a single regex rule from the collected set of phrase rewrites for this iteration
        ngramRegex = re.compile(r'%s(?= )' % "(?= )|".join(map(re.escape, ngramRewriteHash.keys())))

        # Get the batch size for each execution job
        # The very last job executor may received more lines of data
        batch_size = int(numLines/worker)
        batchIndexes = range(0, numLines, batch_size)
        
        batch_returns = [[]] * worker
        with concurrent.futures.ThreadPoolExecutor(max_workers=worker) as executor:
            # The job set
            jobs = set()
            
            # Map the task into multiple batch executions
            for idx in range(worker):
                if idx == (worker-1):
                    jobs.add(executor.submit(phraseRewriteJob, 
                                            ngramRegex, 
                                            textData[batchIndexes[idx]: ], 
                                            ngramRewriteHash, 
                                            idx))
                else:
                    jobs.add(executor.submit(phraseRewriteJob, 
                                            ngramRegex, 
                                            textData[batchIndexes[idx]:(batchIndexes[idx] + batch_size)], 
                                            ngramRewriteHash, 
                                            idx))
            # Clean the raw text list
            textData.clear()
            
            # Get results from batch executions
            for job in concurrent.futures.as_completed(jobs):
                try:
                    ret, idx = job.result()
                except Exception as e:
                    logger.error("phraseRewriteJob() - exception while trying to get result from a batch: %s" % e)
                else:
                    batch_returns[idx] = ret
            textData += sum(batch_returns, [])


    def ApplyPhraseLearning(self, textData, learnedPhrases, num_workers=1):
        logger = logging.getLogger(__name__)
        logger.info("Starting apply phrase learning process.")

        stop = False
        iterNum = 0

        addPhrasesNumList = []
        iterationTimeList = []

        # Get the learning parameters from the structure passed in by the calling function
        maxNumPhrases = self.maxNumPhrases
        maxPhraseLength = self.maxPhraseLength
        functionwordHash = self.functionwordHash
        blacklistHash = self.blacklistHash
        minCount = self.minInstanceCount
        
        # Start timing the process
        functionStartTime = time.clock()
        
        numPhrasesLearned = len(learnedPhrases)
        logger.info("Start phrase learning with %d phrases of %d phrases learned" % (numPhrasesLearned, maxNumPhrases))

        while not stop:
            iterNum += 1
                    
            # Start timing this iteration
            startTime = time.clock()
    
            # Collect ngram stats
            ngramStats = self.ComputeNgramStats(textData, functionwordHash, blacklistHash, num_workers)

            # Uncomment this for more detailed timing info
            countTime = time.clock()
            elapsedTime = countTime - startTime
            logger.info("--- Counting time: %.2f seconds" % elapsedTime)
            
            # Rank ngrams
            rankedNgrams = self.RankNgrams(ngramStats, functionwordHash, minCount)
            
            # Uncomment this for more detailed timing info
            rankTime = time.clock()
            elapsedTime = rankTime - countTime
            logger.info("--- Ranking time: %.2f seconds" % elapsedTime)
            
            # Incorporate top ranked phrases into phrase list
            # and rewrite the text to use these phrases
            if len(rankedNgrams) > 0:
                maxPhrasesToAdd = maxNumPhrases - numPhrasesLearned
                if maxPhrasesToAdd > self.maxPhrasesPerIter:
                    maxPhrasesToAdd = self.maxPhrasesPerIter
                self.ApplyPhraseRewrites(rankedNgrams, textData, learnedPhrases, maxPhrasesToAdd, maxPhraseLength, num_workers)
                numPhrasesAdded = len(learnedPhrases) - numPhrasesLearned
            else:
                logger.warning("No ranked N-grams returned, no phrase rewrites will be applied.")
                stop = True
                
            # Uncomment this for more detailed timing info
            rewriteTime = time.clock()
            elapsedTime = rewriteTime - rankTime
            logger.info("--- Rewriting time: %.2f seconds" % elapsedTime)
            
            # Garbage collect
            ngramStats = None
            rankedNgrams = None
            gc.collect();
                
            elapsedTime = time.clock() - startTime
            numPhrasesLearned = len(learnedPhrases)
            logger.info("Iteration %d: Added %d new phrases in %.2f seconds (Learned %d of max %d)" % 
                (iterNum, numPhrasesAdded, elapsedTime, numPhrasesLearned, maxNumPhrases))

            addPhrasesNumList.append(numPhrasesAdded)
            iterationTimeList.append(elapsedTime)

            # This line maybe redundant to check
            if numPhrasesAdded >= maxPhrasesToAdd or numPhrasesAdded == 0:
                stop = True
        
        # Remove the space padding at the start and end of each line
        regexSpacePadding = re.compile('^ +| +$')
        for i in range(0, len(textData)):
            textData[i] = regexSpacePadding.sub("", textData[i])
        
        gc.collect()
    
        elapsedTime = time.clock() - functionStartTime
        elapsedTimeHours = elapsedTime / 3600.0;
        logger.info("*** Phrase learning completed in %.2f hours ***" % elapsedTimeHours) 

        self.run_logger.log('Added Phrases', addPhrasesNumList)
        self.run_logger.log('Iteration Time', iterationTimeList)


    # Function to run phrase learning using a setting configuration
    # This code could be reused for learning different number of phrases
    # by reusing the previous learned phrases. 
    def RunConfiguration(self, textData, learnedPhrases, addSpace=True, writeFile=True, num_workers=1):
        logger = logging.getLogger(__name__)
        logger.info("Run a Phrase Learning configuration:")
        logger.info("Already learned phrases: %d" % len(learnedPhrases))
        logger.info("Need to learn phrases: %d" % self.maxNumPhrases)

        # The algorithm does in-place replacement of learned phrases directly
        # on the text data structure it is provided and need to add extra space
        # before and after the text
        if addSpace:
            logger.info("Add space before and after the text.")
            for i in range(len(textData)):
                textData[i] = ' ' + textData[i] + ' '

        self.ApplyPhraseLearning(textData, learnedPhrases, num_workers)

        if writeFile:
            phraseFileName = LEARNED_PHRASES_FILE.split('.')[0]
            phraseFileName += '_phrase_' + str(self.maxNumPhrases) + '.' + LEARNED_PHRASES_FILE.split('.')[1]
            phraseFilePath = get_shared_file_path(phraseFileName)
            logger.info("Write the learned phrases into file: %s" % phraseFilePath)
            with open(phraseFilePath, 'w', encoding='utf-8') as fp:
                for phrase in learnedPhrases:
                    fp.write("%s\n" % phrase)

            phraseTextName = PHRASE_TEXT_FILE.split('.')[0]
            phraseTextName += '_phrase_' + str(self.maxNumPhrases) + '.' + PHRASE_TEXT_FILE.split('.')[1]
            phraseTextPath = get_shared_file_path(phraseTextName)
            logger.info("Write the text after phrase-rewrites process into file: %s" % phraseTextPath)
            with open(phraseTextPath, 'w', encoding='utf-8') as fp:
                for line in textData:
                    fp.write("%s\n" % line)
        logger.info("Finished to run a configuration of phrase learning, learned %d phrases." % self.maxNumPhrases)


    # Function to map every lowercased word and phrase used during latent topic modeling 
    # to its most common surface form in the text collection
    def MapVocabToSurfaceForms(self, originalTextCol, phraseTextCol, saveFile=False):
        logger = logging.getLogger(__name__)
        logger.info("Map vocabulary to its most common surface form")

        surfaceFormCountHash = {}
        vocabToSurfaceFormHash = {}
        regexUnderBar = re.compile('_')
        regexSpace = re.compile(' +')
        regexClean = re.compile('^ +| +$')
        
        # First go through every line of text, align each word/phrase with
        # it's surface form and count the number of times each surface form occurs
        for i in range(0, len(self.textFrame)):    
            origWords = regexSpace.split(regexClean.sub("", str(self.textFrame[originalTextCol][i])))
            numOrigWords = len(origWords)
            newWords = regexSpace.split(regexClean.sub("", str(self.textFrame[phraseTextCol][i])))
            numNewWords = len(newWords)
            origIndex = 0
            newIndex = 0

            while newIndex < numNewWords:
                # Get the next word or phrase in the lower-cased text with phrases and
                # match it to the original form of the same n-gram in the original text
                newWord = newWords[newIndex]
                phraseWords = regexUnderBar.split(newWord)
                numPhraseWords = len(phraseWords)
                matchedWords = " ".join(origWords[origIndex:(origIndex + numPhraseWords)])
                origIndex += numPhraseWords
                    
                # Now do the bookkeeping for collecting the different surface form 
                # variations present for each lowercased word or phrase
                if newWord in vocabToSurfaceFormHash:
                    vocabToSurfaceFormHash[newWord].add(matchedWords)
                else:
                    vocabToSurfaceFormHash[newWord] = set([matchedWords])

                # Increment the counter for this surface form
                if matchedWords not in surfaceFormCountHash:
                    surfaceFormCountHash[matchedWords] = 1
                else:
                    surfaceFormCountHash[matchedWords] += 1
                newIndex += 1
        
        # After aligning and counting, select the most common surface form for each
        # word/phrase to be the canonical example shown to the user for that word/phrase
        for ngram in list(vocabToSurfaceFormHash):
            if not ngram:
                vocabToSurfaceFormHash.pop(ngram)
                continue
            
            maxCount = 0
            bestSurfaceForm = ""
            for surfaceForm in vocabToSurfaceFormHash[ngram]:
                if surfaceFormCountHash[surfaceForm] > maxCount:
                    maxCount = surfaceFormCountHash[surfaceForm]
                    bestSurfaceForm = surfaceForm
            
            if bestSurfaceForm == "":
                logger.warning("Warning: NULL surface form for ngram '%s'" % ngram)
            vocabToSurfaceFormHash[ngram] = bestSurfaceForm
        logger.info("Finished mapping vocabulary to its most common surface form")

        if saveFile:
            surfaceTextName = SURFACE_MAPPING_FILE.split('.')[0]
            surfaceTextName += '_phrase_' + str(self.maxNumPhrases) + '.' + SURFACE_MAPPING_FILE.split('.')[1]

            fpath = get_shared_file_path(surfaceTextName)
            with open(fpath, 'w', encoding='utf-8') as fp:
                for key, val in vocabToSurfaceFormHash.items():
                    fp.write("%s\t%s\n" % (key, val))
            logger.info("Saved vocabToSurfaceFormHash to file: %s" % fpath)
        return vocabToSurfaceFormHash


    # Reconstruct the full processed text of each document and put it into a DataFrame
    def ReconstituteDocsFromChunks(self, idColumnName, textColumnName, writeFile=True):
        logger = logging.getLogger(__name__)
        logger.info("Reconstitute documents and put it into a DataFrame")

        dataOut = []
        currentDoc = ""
        currentDocID = ""

        for i in range(len(self.textFrame)):
            textChunk = self.textFrame[textColumnName][i]
            docID = str(self.textFrame[idColumnName][i])
            if docID != currentDocID:
                if currentDocID != "":
                    dataOut.append([currentDocID, currentDoc])
                currentDoc = textChunk
                currentDocID = docID
            else:
                currentDoc += ' ' + textChunk
        dataOut.append([currentDocID, currentDoc])

        frameOut = pd.DataFrame(dataOut, columns=[idColumnName, 'ProcessedText'])
        if writeFile:
            reconstituteFileName = RECONSTITUTED_TEXT_FILE.split('.')[0]
            reconstituteFileName += '_phrase_' + str(self.maxNumPhrases) + '.' + RECONSTITUTED_TEXT_FILE.split('.')[1]
            fpath = get_shared_file_path(reconstituteFileName)
            frameOut.to_csv(fpath, sep='\t', index=False)
            logger.info("Saved reconstituted document text to file: %s" % fpath)
        return frameOut





