# Document Collection Analysis

![Data_Diagram](https://www.usb-antivirus.com/wp-content/uploads/2014/11/tutorial-windwos-10-2-320x202.png)

* Documentation site for GitHub repository (TODO).

[//]: # (**The above info will be included in the Readme on GitHub**)

## Prerequisites

The prerequisites to run this example are as follows:

1. Make sure that you have properly installed Azure ML Workbench by following the [installation guide (TODO)](https://github.com/Azure/ViennaDocs/blob/master/Documentation/Installation.md).

1. This example could be run on any compute context. However, it is recommended to run it on a multi-core machine with at least of 16-GB memory and 5-GB disk space.

## Introduction

This example showcases how to analyze a large collection of documents, which includes phrase learning, topic modeling, and topic model analysis using Azure ML Workbench. With Azure ML Workbench, one can easily scale up and out if the collection of documents is huge. And it can help tune hyper-parameters on different compute contexts. The capability of using iPython notebooks within Azure ML Workbench also enables ease of development.

## Overview

With large amount of data (especially unstructured text data) collected every day, a significant challenge is to organize, search, and understand vast quantities of these texts. The Document Collection Analysis example is aimed to demonstrate an efficient and automated end to end workflow on how to analyze large document collection and serve the downstream NLP tasks.

The key learnings delivered by this example are as follows:

1. Learn phrases from documents, which can provide meaningful information versus individual words.

1. Discover hidden topic patterns that present itself across different documents.

1. Annotate documents (document embedding) according to those topics and serve as a featurizer at a document level.

1. Use the learned topics to better organize, search, and summarize documents.

This example could fit into a vast majority of NLP tasks, including (but not limited to): sentiment analysis, document classification, document semantic similarity, and document search. It is a critical workload for many industries and companies. For instance, government document ingestion and analysis, law firms legal documents analysis, and insurance claim process.

The machine learning techniques/algorithms used in this example include:

1. Text processing pipeline to scrub documents and then find n-grams that best categorize them. Those n-grams can then be used to learn phrases, which could provide more meaningful information than purely using individual words.

1. A Latent Dirichlet Allocation (LDA) topic model is built to learn latent structure in documents collection. The model can be updated with more documents collected.

1. The learned topic model works as a featurizer at document level where new documents can be mapped to a topic vector. By using the mapped topic vector, we can then use it as features for various NLP tasks mentioned earlier.

## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.microsoft.com.

When you submit a pull request, a CLA-bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (for example, label, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information, see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.
