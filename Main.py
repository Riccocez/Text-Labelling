
# coding: utf-8

# # Automatic Labelling of Sentences in Research Papers

# The problem we intend to solve involves performing automatic labelling from a set of research papers from three different fields.
# 
# Due to the short period of time to solve the task, the approach taken comprises the usage of techniques used in NLP like tokenization and POS tagging, together with simple, yet powerful, algorithms such as Naive Bayes and SVM.
# 
# The corpus used is available in UCI ML Repository: https://archive.ics.uci.edu/ml/machinelearningdatabases/00311/
# 
# This dataset contains sentences from the abstract and introduction of 30 annotated research papers. These articles come from three different domains:
# 
#     1. PLoS Computational Biology (PLOS)
#     2. The machine learning repository on arXiv (ARXIV)
#     3. The psychology journal Judgment and Decision Making (JDM)
# 
# 
# The 5 classes (or labels) contained in each article are:
# 
#     1. AIMX. The specific research goal of the paper
#     2. OWNX. The author’s own work, e.g. methods, results, conclusions
#     3. CONT. Contrast, comparison or critique of past work
#     4. BASE. Past work that provides the basis for the work in the article.
#     5. MISC. Any other sentences
#     
# Based on these, the task consists on solving a multi-class classification problem. The 5 labels contained are the 5 classes that our model will try to predict. These classes are mutually exclusive so, we only expect that our model label each sentence of a given article with only one of the 5 possibilities.   
#     
# For simplicity, we used NLTK to perform most of the NLP related techniques. Likewise, we considered the scikit-learn Classifiers included in the same NLTK library. 
# 
# The steps we performed to build our Classifiers are the following:
#     
#     1. Analyse the structure of the corpus
#     2. Preprocess data
#     3. Feature Selection
#     4. Split corpus into Training and Testing set
#     5. Training / Testing Models
#     7. Model Selection
#     8. Conclusions

# In[1]:

get_ipython().magic('matplotlib inline')
get_ipython().magic("config InlineBackend.figure_format='retina'")


# In[2]:

import nltk
import os
import time
import string
import pickle
from Files import Files
import re
from Preprocessing import Preprocessing as preproc
from nltk.corpus import stopwords as sw
import pandas as pd
import numpy as np

from nltk.classify import SklearnClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.svm import SVC
from sklearn.metrics import precision_recall_curve
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils

import matplotlib.pyplot as plt


# ## 1. Analyse the structure of the data
# To give an example of the shape of the raw data, below it is shown a short section of one of the articles used, containing sentences of its corresponding abstract and introduction.

# In[3]:

sample = Files().get_sample_file(path="SentenceCorpus/labeled_articles/",
                    extension="1.txt")

sample 


# As observed above, the raw data is mostly preprocessed. Each of the section names is sorrounded by a set of "#" characters. Each of the labeled sentences is preceded by its corresponding label, which is in capital letters. 
# 
# Moreover, (it is not observed above but) all citations contained in the articles have been replaced with the word "CITATION". All numbers have been replaced with the mark "NUMBER". All equations and symbols were replaced with the mark "SYMBOL".
# 
# In general, the number and lenght of sentences contained in the abstract are smaller than the sentences of the introduction. In addition, it is quite rare that the abstract sentences contains equations (that is "SYMBOL" marks) or CITATION marks, but it is quite common that these marks appear in the introduction sentences.
# 
# Similarly, there are different patterns observed in sentences of each class like CITATION marks for the case the class CONT.
# 
# We took advantage of all these observations in order to select the most relevant features that could lead to a competitive performance during the automatic labelling process.

# ## 2. Preprocess data

# Based on the results of structure analysis, we perform preprocessing and feature selection in both sections (the abstract and the introduction) separately. The preprocessing stage was built in parallel with the feature selection stage. Therefore, appart from manipulating the corpus, we also used this step to extract all relevant features from each section and sentence, as well as to split the data into a training and testing set.
# 
# For each section, we built a dictionary with a relevant set of features, which were the result of the following procedure:
# 
#     1. Paired each sentence with its corresponding label
#     2. Each sentence was lowered case and tokenized
#     3. Each lowered sentence was tagged with the simplest POS tagger of NLTK
#     4. Performed frequency distribution of each POS tag in each sentence
#     5. Included additional features like whether the sentence belongs to the introduction or abstract section
#     6. Built a standarized feature set, which includes all the features included in the abstract and introduction 
#     7. Mapped features of each sentence into the form of the standarized feature set built in 6
#     8. Split corpus of abstract and introduction separately into a testing and training set
#     9. Joined splits training set of introduction and abstract to conform a unique training set
#     10. Performed the same joint of 9 using the testing set to build a unique testing set
#     
# 

# In[4]:

files = Files()
preprocessing = preproc()
files.get_all_files(path="SentenceCorpus/labeled_articles/",
                    extension="1.txt")
files.split_abstract_intro_files()

files.super_tag_dict = preprocessing.sum_all_tags(files.get_tag_sets())
preprocessing.reset_all_tags()


# ## 3. Feature Selection
# 
# We considered using 44 features to train the different classifiers proposed. In summary, they contain information related to:
# 
#     - Distintic tags in the data
#     - The section in which the sentence appears  
#     - Whether a different tag in the sentence is not included in the standarized feature set
#     
# As mentioned in the analysis of data structure, word distribution plays an important role when diferentiating abstract from introduction sentences. One of the most efficient ways to capture this distribution was by considering the distribution of the POS tags contained in each sentence.
# 
# If we were intended to capture the distribution of each word that may exists, we could be facing diverse inconvenients that could make our task more complex than necessary. For instance, the sparsity and dimensionality present in language could lead us either to have an extremely large feature set, with most of the features with 0 values, or to incur into building an infinite feature set that never ends to contain all the words contained in a sentence.
# 
# Therefore, despite we are aware using POS tag distributions have some limitations, we are certain that this distribution is able to capture the essential structure in the corpus, which will help the classification algorithms to show a competitive performance when labeling the sentences of unseen scientific articles.
# 
# 

# In[5]:

feats_abs = preprocessing.build_feature_set(files.label_abstract,
                                            'abstract')
feats_intro = preprocessing.build_feature_set(files.label_intro,
                                              'introduction')


# ## 4. Split corpus into Training and Testing set

# In order to capture most of the diversity in the corpus, we split the data into a train_set, a dev_set and a test_set. As explained before, we took a training and testing set from each of the sections, and them we combine them into a unique training and testing set. This help us to make sure that both unique sets contain sentences from the abstract and the introduction. As a result, we will be more confident that the performance and results obtained from each model comprise the cases we are interested in.
# 
# As a default, we consider 30% of corpus should be testing and 15% of the training set should be used for the dev_set. The benefit of including the dev_set is helping our model to tune the fitted model. However, because of limit fo time, we are not going to use it for this first version of the trained models.
# 

# In[6]:

train_set,dev_set,test_set = preprocessing.split_dataset(
                                        feats_abs,feats_intro)

train_data, test_data, dev_data = preprocessing.extract_feats(
                                                train_set,dev_set,test_set)


# In[7]:

test = [sentence[0] for sentence in dev_data]
y_labels = [sentence[1] for sentence in dev_data]


# ## 5. Training/Testing Models

# In[8]:

def save_prod_dist(model,test_data):
    """
    * Extract probability distributions obtained for each class of 
    * the classifier
    """
    
    y_label = 0
    modeldist_probs = []
    
    test = [sentence[0] for sentence in test_data]
    y_labels = [sentence[1] for sentence in test_data]
    
    for pdist in model.prob_classify_many(test):
        modeldist_probs.append([pdist.prob('AIMX'), pdist.prob('OWNX'),
                        pdist.prob('CONT'), pdist.prob('BASE'),
                        pdist.prob('MISC'),y_labels[y_label]])
        y_label +=1
    return  modeldist_probs


# In[9]:

def plot_prec_recall(precision,recall,modelName):
    plt.clf()
    plt.plot(recall, precision,lw=3, color='navy',
         label='Precision-Recall curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Precision-Recall of ' + modelName)
    plt.legend(loc="lower left")
    plt.show()
    return

def classify(model, featureset):
    return model.prob_classify(featureset).max()

def compute_metrics(model,testset, y_labels, modeldist_probs,modelName):
    
    y_true = []
    
    prob_dists = [max(dist[0:4]) for dist in modeldist_probs]
    prob_dists = np.array(prob_dists)
    
    preds = [classify(NBclassifier,featset) for featset in testset]

    for i in range(0, len(y_labels)):
        if preds[i] == y_labels[i]:
            y_true.append(1)
        else:
            y_true.append(0)
            
    precision, recall, thresholds = precision_recall_curve( y_true, prob_dists)
    
    plot_prec_recall(precision,recall,modelName)
    
    return


# ## Naive Bayes Classifier

# The accuracy of the Naive Bayes classifier shows the poorest performance among the three classifiers. Its accuracy is 55% of the classifications.
# 
# Moreover, we can see that the tags VBP, NNS and PRP are among the most informative parameters for the model to determine the class of any given sentence.

# In[10]:

NBclassifier = nltk.NaiveBayesClassifier.train(train_data)
print(nltk.classify.accuracy(NBclassifier,dev_data))


# In[11]:

NBdist_probs = save_prod_dist(NBclassifier,test_data)
NBclassifier.show_most_informative_features()


# In[12]:

compute_metrics(NBclassifier,test,y_labels,NBdist_probs,'NBClassifier')


# # SklearnClassifier

# SKlearn classifier is the best classifier among the three algorithms according to accuracy. It's accuracy is over 60.5%. 

# In[13]:

SKclassifier = SklearnClassifier(BernoulliNB()).train(train_data)
print(nltk.classify.accuracy(SKclassifier,dev_data))


# In[14]:

SKdist_probs = save_prod_dist(SKclassifier,test_data)


# In[15]:

compute_metrics(SKclassifier,test,y_labels,SKdist_probs,'SKClassifier')


# # Support Vector Classifier

# Finally, SVC proves to have a slightly better performance than the Naive Bayes model. The accuracy of SVC is close to 54.5%. 

# In[16]:

SVclassifier = SklearnClassifier(SVC(), sparse=False).train(train_data)
print(nltk.classify.accuracy(SVclassifier,test_data))


# In[17]:

SVdist_probs = save_prod_dist(SVclassifier,test_data)


# In[18]:

compute_metrics(SVclassifier,test,y_labels,SVdist_probs,'SVClassifier')


# ## 6. Model Selection
# 
# According to the results based on accuracy, the best model we would select is SKlearn Classifier. The performance proven by this model is better than the other two simple models. However, as we will discuss within the conslusions we need to be aware of the implications of selecting this model solely based in this metric.

# ## 7. Conclusions
# 
# The classifiers we used for performing automatic labelling of sentences are one of the simplest algorithms that we could use to solve text/sentence classification. Their simplicity and easiness to train them are the main reasons of why we decided to use and compare them. There are alternative solutions that migth be extremely helpful for solving this task, and example of that could be RNNs. Nevertheless, for instance, using this sort of approach would involve increasing the complexity and computational costs for building the classifier model. Therefore, as we are not certain that using a more a complex approach generates the optimal performance, we preferred to maintain the complexity and computational costs as simple as possible.  
# 
# 
# Focusing on the performance of presented models, it is clear that the complexity of the approach taken is not directly related to a better performance when classifying. For example, the Support Vector Classifier (SVC) has a similar performance to the Naive Bayes Classifier (NBC). These couple of approaches are sensitive to parameter optimization. Therefore, we preferred focusing on devising an optimal feature engineering that lead to a very competitive performance using these models. However, as this is a simple version of a robust classifier, we would need to improve our model with additional parameters that could help the model to learn a better generalization of the patterns hidden within the corpus.
# 
# On the other hand, there were different problems in language that we needed to consider in order for solving the task easier. For instance, we dealt with two problems: 1)word frequencies and 2) data sparsity. These two things can difficult the classification task, and mainly the capture of the most relevant information within the corpus. So, we took advantage of using POS tags frequencies to solve these problems. 
# 
# A consequent problem of using POS tags frequencies is the accuracy of the POS taggings. We are aware that any innacuracy of the POS tagger impacts directly on the performance of the classifiers. The POS tagger used is the one provided by the NLTK library. Thus, eventhough the performance of tagging is performing reasonable well, rare or infrequent words might not be tagged properly. This might be critical for our task since we are trying to label sentences from scientific articles. In this sort of documents, rare or infrequent words tend to be the most important so it's very liekly that we could be affecting the classification taskdramatically  
# 
# A next step for improving the problem of POS tagging innacuracy would be either adding more freatures to the model, and then, reduce the impact of these innacuracies, or training a new tagger with tagged data related to the domain we are interesting to tag. In this case, train a new tagger with sentences tagged of scientific articles related to Biology,Psychology and Machine Learning.
# 
# Finally, in order to improve the performance of the current models we could include more informative features related to the structure of the corpus. For instance, we can consider building a set of the most frequent words appearing in each of the classes and add the number of words in the sentence that appear in this set of words. In addition, we could try to look for synonyms of rare words so that our model can identify easier and more accurately the correct POS tag. Finally, we could consider prefixes and suffixes as part of the feature set so that the model can distinguish unseen words properly. 
# 
# 
# 
# 
