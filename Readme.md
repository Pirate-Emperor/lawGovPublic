# [lawGov: Judicial Component of GovUnityXplorer](https://github.com/Pirate-Emperor/lawGovPublic)

## --- Note: This is a public repository of the lawGov Model ---

## Social Links

[![GitHub](https://img.shields.io/badge/GitHub-Community-brightgreen?logo=github)](https://github.com/Pirate-Emperor)
[![Reddit](https://img.shields.io/badge/Reddit-Announcement%20%2B%20Community-orange?logo=reddit)](https://www.reddit.com/r/GovUnityXplorer/)
[![Twitter](https://img.shields.io/badge/Twitter-Announcement-blue?logo=twitter)](https://twitter.com/PirateKingRahul)
[![Discord](https://img.shields.io/badge/Discord-Community-blueviolet?logo=discord)](https://discord.com/channels/1200760563043663912/1200760563509235814)
[![Telegram](https://img.shields.io/badge/Telegram-Community-informational?logo=telegram)](https://t.me/+TUZqu7663DtlYWE1)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Announcement-9cf?logo=linkedin)](https://www.linkedin.com/in/piratekingrahul)
[![Skype](https://img.shields.io/badge/Skype-Contact-important?logo=skype)](https://join.skype.com/invite/yfjOJG3wv9Ki)
[![Medium](https://img.shields.io/badge/Medium-Announcement%20%2B%20Post-brightgreen?logo=medium)](https://medium.com/@piratekingrahul)

## Table of Contents:
- [Introduction](#introduction)
- [Research Question](#research-question)
- [Features](#features)
- [UN Goals](#un-goals-and-targets-aligned-with-the-project)
- [Installation](#installation)
- [Machine Learning Strategy and Results: Ensemble vs. Unified Model](#machine-learning-strategy-and-results-ensemble-vs-unified-model)
- [Models](#models)
  - [Doc2Vec](#doc2vec)
  - [1D-CNN](#1d-cnn)
  - [TextVectorization with TF-IDF](#textvectorization-with-tf-idf)
  - [GloVe](#glove)
  - [BERT](#bert)
  - [Long Short-Term Memory (LSTM)](#long-short-term-memory-lstm)
  - [FastText](#fasttext)
- [Experiments](#experiments)
- [Training](#training)
- [Final Steps](#final-steps)
- [Additional](#additional)
- [Challenges](#challenges)
- [Reference](#reference)

## Introduction

Welcome to lawGov - your ultimate legal AI assistant designed to revolutionize legal decision-making.

lawGov is an advanced natural language processing (NLP) application that predicts the ideological direction of legal judgments with unparalleled accuracy. Our primary goal is to provide legal professionals with the tools they need to make informed decisions, save time, and improve success rates in the legal system.

## Research Question

My goal was to predict the ideological direction ("liberal" or "conservative" as defined below) of Supreme Court case decisions based mainly on the facts of the case as known before the Court rules or even hears arguments (e.g. area of the law it relates to, or lower-court outcomes as described below). Parties interested in an upcoming case could obviously use this tool to assess their prospects for success and plan accordingly. In litigation strategy, they could also use feature importances to assess whether a particular justice or constellation of justices has a particularly strong or atypical voting tendency when it comes to a particular issue (e.g. taxes, or civil rights).

An interesting methodological question I attempted to answer was whether it is more effective to model decisions of the Court as a whole (i.e. the single outcome of generally nine justices' votes), or to separately model decisions of individual justices in probability terms and then aggregate these probabilities to predict the outcome of the case (ensemble method). Although I found the ensemble method not to add accuracy with past cases, it is a useful approach if the Court is changing or if I know a particular combination of justices will hear the case (say, a reduced eight-justice panel when Congress is unable to confirm a nominee).

### Target
I selected ideological direction (i.e. "liberal" vs. "conservative" as detailed on pages 50-52 of the codebook, included in this repo) as my target for prediction since this is generally what people have an intuition for and want to know. Prior research by Katz, Bommarito and Blackman targeted case disposition, coded in the database as any of 11 categories but essentially indicating whether the Court decided to affirm or reverse a lower-court decision. However, a significant number of decisions (15% of their dataset, which included pre-1946 cases), at least at the justice-centered level, were ambiguous. Ideological direction, in contrast, is "unspecifiable" in only 1.7% of modern cases and blank for only 0.4%. Ideological direction has the additional advantage of being balanced close to 50/50% between liberal and conservative.

### Features
As the lengthy codebook shows, all of the columns are essentially categorical. In choosing predictor variables for my models, I first left aside non-predictive columns with unique values for each case (name of case, identification numbers, date of case). Next I needed to eliminate any columns related to the outcome of a case. This is because, in a genuine prediction scenario, I would not have data for anything not known prior to the Court's decision's being made public. Thus I do not train my models with even tangentially outcome-related columns, such as the authority cited for a decision (could be 7 different values, including "statutory construction" and "federal common law") as this could contain some clue about the decision itself and would not be known prior to the case's hearing.

## UN Goals and Targets Aligned with the Project:
- **Goal 16: Peace, Justice, and Strong Institutions**
  - Target 16.6: Develop effective, accountable, and transparent institutions at all levels.
- **Goal 9: Industry, Innovation, and Infrastructure**
  - Target 9.5: Enhance scientific research, upgrade technological capabilities, and promote innovation.
- **Goal 10: Reduced Inequalities**
  - Target 10.2: Empower and promote the social, economic, and political inclusion of all, irrespective of age, sex, disability, race, ethnicity, origin, religion, or economic or other status.

## Installation:
To run lawGov locally, follow these steps:

1. Clone the repository: `git clone https://github.com/Pirate-Emperor/lawGovPublic`
2. Create virtual environment: `pip install venv
                                python -m venv env310
                                env310\Scripts\activate.bat`
3. Install dependencies: `pip install -r requirements.txt`
4. Navigate to the src directory: `cd src`
5. Download the GloVe pre-trained embeddings from the following link:
   - [glove.6B.zip](https://nlp.stanford.edu/data/glove.6B.zip)
   - [glove.42B.300d.zip](https://nlp.stanford.edu/data/glove.42B.300d.zip)
   - [glove.840B.300d.zip](https://nlp.stanford.edu/data/glove.840B.300d.zip)
   - [glove.twitter.27B.zip](https://nlp.stanford.edu/data/glove.twitter.27B.zip)
6. Create a directory called glove and put the downloaded all glove files inside it.
7. Download Pretrained Models from the following link: [Models](https://drive.google.com/drive/folders/1UPY1pIEjDVOsApjBgCR1n3ZsSkRJEB3I)
8. Create a directory called models and put the downloaded models inside:
9. Run the application: `streamlit run src/main.py nul`

## Machine Learning Strategy and Results: Ensemble vs. Unified Model

My initial strategy was the simpler of the two main approaches I took: I split the case-centered data into testing and training sets and tried various machine learning techniques to predict the ideological direction of the full Court's decisions. I didn't use individual justice identities (except for who was chief justice at the time) and vote counts. The results were encouraging: around 70% test accuracy (75% ROC-AUC), but I wanted to make use of the justice-centered data.

To do so I split the justice-centered data into testing and training sets (by case so that justices on a particular case were either all in the testing set or all in the training set). Then I split the training set by justice and trained models for each of the 37 justices, tuning hyper-parameters for each using GridSearchCV. The justice-centered models had anywhere from 5,087 rows for Brennan to 82 rows for Gorsuch. These could be used as-is if individual-justice decisions are of interest, but I decided to use them as an ensemble to predict full-Court decisions.

For each case, I assembled a vector of probabilities for the justices who voted on that case (usually 9 but as few as 5 historically). These were the predict_proba outputs of individual justice models for the case in question. Using a Poisson binomial distribution (since the probabilities are uneven), I found the probability that a majority of the justices' votes are liberal vs. conservative. This was computed by adding up the Poisson-binomial probability mass function for, say, 5, 6, 7, 8, or 9 justices voting liberal if there were 9 justices sitting. The calculation was adjusted depending on how many justices were voting: the probability mass function was added for 4, 5, or 6 justices voting liberal if only 6 justices were voting.

I expected the ensemble method to improve on the initial strategy since the individual justices' models were individually tuned; but for now, any gain in accuracy, ROC-AUC, and so forth has been minimal. I plan to investigate further.

### Note on Data Leakage
I had to be very careful to use the same test-train splits within each justice's model that I used for the case-centered approach. That is, if a case was used for training in the case-centered approach, it had to be used only for training in each of the 9 or so justice-centered models where it was relevant. Otherwise, if even one justice's model used the case for testing rather than training, the ensemble method would unfairly have "better" information about the test set than the case-centered strategy. As a result, the test-train splits for each justice deviated randomly from the 70/30% split imposed on the case-centered data. Another disadvantage of this necessary precaution was that it disallowed automatic k-folds cross-validation as I needed to keep the same test-train split across all models.

## Models

lawGov was trained using 7 different models:

### Doc2Vec

**Doc2Vec** is a natural language processing (NLP) technique that allows machines to understand the meaning of entire documents, rather than just individual words or phrases.

### 1D-CNN

**CNN (Convolutional Neural Network)** is a type of artificial neural network commonly used in text classification and sentiment analysis.

### TextVectorization with TF-IDF

**TF-IDF (Term Frequency-Inverse Document Frequency)** is a technique used to measure the importance of words in a document or corpus.

### GloVe

**GloVe (Global Vectors for Word Representation)** is an unsupervised learning algorithm for generating word embeddings.

### BERT

**BERT (Bidirectional Encoder Representations from Transformers)** is a pre-trained language model developed by Google, commonly used in natural language processing tasks.

### Long Short-Term Memory (LSTM)

**LSTM (Long Short-Term Memory)** is a type of recurrent neural network architecture, effective in modeling sequential data.

### FastText

**FastText** is a library and approach for efficient text classification and representation learning developed by Facebook AI Research.

## Experiments

To achieve the best results, we tried different experiments in **lawGov**:

- **Data Preprocessing**: Including removing stopwords, lowercasing all letters, stemming, and removing non-alphabet characters except the `_` letter, punctuation, and digits.
- **Data Anonymization**: Replacing parties' names from the case facts with a generic `_PARTY_` tag to ensure models are not biased towards parties' names.
- **Label Class Imbalance**: Dealing with class imbalance as a standalone preprocessing step to see if there was an impact on the final accuracy of the **lawGov** models or not.

Each experiment can be made or not, so we ended up with 8 (2 to the power of 3) possible combinations. 

## Training

For training, the dataset was divided into training and testing sets with a proportion of 80:20, and this division was constant for all of lawGov's models to ensure comparability. The training data was divided into 4 parts (folds) using 4-fold cross-validation. Testing accuracies were obtained for each fold, resulting in 4 testing accuracies per combination. 

## Final Steps

After training the models, an ensemble learning approach was employed to combine the predictions of each model, using **voting** to determine the final prediction for each case.

## Additional

For a more detailed explanation of **lawGov** and to see the results of its models in much more detail, please refer to each model's notebook. Thank you.

## Challenges:
One of the main challenges in legal judgment prediction using NLP is the complexity and variability of legal language. Legal documents often use technical terminology, jargon, and complex sentence structures that can be difficult for NLP models to analyze accurately. Additionally, legal cases can be influenced by various factors, including the specific circumstances of the case, the legal jurisdiction, and the judge's personal beliefs and biases.

## Reference:
- ML Conflict: https://www.sciencedirect.com/science/article/abs/pii/016792369390034Z
- Conflict Resolution Strategies In Artificial Intelligence. https://conflictresolved.com/conflict-resolution-strategies-in-artificial-intelligence/.
- Artificial Intelligence Techniques for Conflict Resolution. https://link.springer.com/article/10.1007/s10726-021-09738-x.
- Machine Learning and Conflict Prediction: A Use Case. https://stabilityjournal.org/articles/10.5334/sta.cr.
- Using Artificial Intelligence to provide Intelligent Dispute Resolution .... https://link.springer.com/article/10.1007/s10726-021-09734-1.
- Machine Learning-based SON function conflict resolution | IEEE .... https://ieeexplore.ieee.org/document/8969675/.