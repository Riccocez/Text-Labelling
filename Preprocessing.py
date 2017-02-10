
# coding: utf-8

# In[ ]:

import nltk
from random import shuffle


# In[ ]:

class Preprocessing(object):
    
    def __init__(self,label_abstract = [],label_intro = []):
        """
        * A preprocessing object which allows perform most of the 
        * required transformations to manipulate dataset
        """
        
        self.label_abstract = label_abstract
        self.label_intro = label_intro
        self.super_dict = {}
        self.tag_feats = {}
        
  
    def tokenize_section(self,section):
        """
        * Splits each section into a list of sentences 
        """
        
        #Split section by sentence
        tokenized_section = section.split("\n")
        
        return tokenized_section
    
    def label_data(self,tokenized_section):
        """
        * Labels each sentence of tokenized section
        * additionally returns the tags frequencies of each
        * of the sentences contained in the tokenized section
        """
        
        label_section = []
        tags_freq_section = []
        
        for sentence in tokenized_section:
            
            sentence_tagged,tags_freq = self.preprocess_sentence(sentence[5:]) 
            preproc_sentence = (sentence_tagged, sentence[:4])
            
            tags_freq_section.append(tags_freq)
            label_section.append(preproc_sentence)
            
        return label_section, tags_freq_section

    
    def preprocess_sentence(self,sentence):
        """
        * Lowercases a given sentence
        """
        
        lowered_sentence = []
        sentence_tokenized = self.tokenize_sentence(sentence)
        
        for word in sentence_tokenized:
            
            if not word.isupper():
                lowered_sentence.append(word.lower())

            else:
                lowered_sentence.append(word)
        
        sentence_tagged,tags_freq = self.tagging_data(lowered_sentence)
                
        return sentence_tagged,tags_freq
    
    def tokenize_sentence(self,sentence):
        """
        * Tokenize a given sentence
        """
        
        
        sentence_tokenized = nltk.wordpunct_tokenize(sentence)
        return sentence_tokenized
    
    
    def tagging_data(self,lowered_sentence):
        """
        * Performs all tag-related tasks to the lowered_sentence
        """
        
        sentence_tagged = self.pos_tag_sentence(lowered_sentence)
        tags_freq = self.tag_counter(sentence_tagged)
        
        return sentence_tagged,tags_freq
    
    def pos_tag_sentence(self,lowered_sentence):
        """
        * Performs the POS_tag of the lowered_sentence
        """
        
        sentence_tagged = nltk.pos_tag(lowered_sentence)
        
        return sentence_tagged
    
    def tag_counter(self,sentence_tagged):
        """
        * Counts the tag distribution of a sentence_tagged
        """
        
        tags_freq = {}
        
        for token in sentence_tagged:
            if token[1] in tags_freq:
                tags_freq[token[1]] +=1
            else:
                tags_freq.update({token[1]:1})
                    
        return tags_freq
            
    def sum_all_tags(self,list_tag_freqs):
        """
        * Aggregates the distinct tags of list_tag_freqs
        * into a unique dictionary
        """
        
        for set_tag_freqs in list_tag_freqs:
            for dictlist in set_tag_freqs:
                for dic in dictlist:
                    for k,v in dic.items():
                        if k in self.super_dict:
                            self.super_dict[k] += v
                        else:
                            self.super_dict.update({k:v})
                            
        return self.super_dict
    
    def reset_all_tags(self):
        """
        * Resets the values of all tags without removing the keys
        * of the dictionary involved
        """
        
        additional_feats = ['abstract','introduction','paper','unknown']
        
        self.tag_feats = dict((k,0)for k in self.super_dict)
        
        for feat in additional_feats:
            self.tag_feats.update({feat:0})
            
        return 
    
    def build_feature_set(self,label_section,sectionName):
        """
        * Builds the feature set of each sentence contained in the 
        * label_section
        """
        
        section_dataset = []
        for doc in label_section:
            for sentence in doc:
                
                feature_set,sentence = self.feature_freq(sentence)
                
                feats_sentence = self.build_sentence_features(feature_set[0])
                
                
                if sectionName == "abstract":
                    feats_sentence[sectionName] = 1

                elif sectionName == "introduction":
                    feats_sentence[sectionName] = 1
                
                
                section_set = ((feats_sentence,feature_set[1]),sentence)
                section_dataset.append(section_set)
                
        return section_dataset
                
     
    def feature_freq(self,sentence):
        """
        * Pairs a label with its corresponding 
        * tag distribution of a sentence_tagged
        """
        
        tags_freq = {}
        
        for token in sentence[0]:
            
            if token[1] in tags_freq:
                tags_freq[token[1]] +=1
            else:
                tags_freq.update({token[1]:1})
            
        return (tags_freq,sentence[1]),sentence
            
    def build_sentence_features(self,feature_set):  
        """
        * Maps values of feature_set into the form of the unique feature_set
        """
        
        self.reset_all_tags()
        
        
        for k,v in feature_set.items():
            if k in self.tag_feats:
                self.tag_feats[k] += v
            else:
                self.tag_feats['unknown'] += v
        
        return self.tag_feats
    
    def split_dataset(self,feats_abs,feats_intro,size=0.3,devSize = 0.15):
        """
        * Splits the dataset into a training,testing and dev set
        """
        
        train_set = []
        dev_set = []
        test_set = []
        
        shuffle(feats_abs)
        shuffle(feats_intro)
        
        thres_train =  int(len(feats_abs)*size)
        thres_dev =  int(len(feats_abs)*devSize) 
        
        tmp_abs_train = feats_abs[0:thres_train]
        tmp_dev_train = tmp_abs_train[-thres_dev:]
        tmp_abs_test = feats_abs[thres_train:]
        
        self.split_section(thres_train,thres_dev,feats_abs)
        
        tmp_intro_train = feats_intro[0:thres_train]
        tmp_dev_test = tmp_intro_train[-thres_dev:]
        tmp_intro_test = feats_intro[thres_train:]
        
        train_set = tmp_abs_train + tmp_intro_train
        dev_set = tmp_dev_train + tmp_dev_test
        test_set = tmp_abs_test + tmp_intro_test
        
        return train_set,dev_set,test_set
    
    def split_section(self,thres_train,thres_dev,feats_section):
        """
        * Splits section into a training,testing and dev set
        """
        
        tmp_sec_train = feats_section[0:thres_train]
        tmp_dev_train = tmp_sec_train[-thres_dev:]
        tmp_sec_test = feats_section[thres_train:]
        
        return tmp_sec_train,tmp_dev_train,tmp_sec_test
        
    def extract_feats(self,train_set,dev_set,test_set):
        """
        * Extract feature set from the training,testing and dev corpora
        """
        
        train_data = self.extract_feat_set(train_set)
        test_data = self.extract_feat_set(test_set)
        dev_data = self.extract_feat_set(test_set)
        
        return train_data, test_data, dev_data
    
    def extract_feat_set(self,feature_set):
        """
        * Extracts feature set from the feature_set
        """
        
        feat_set = [feats[0] for feats in feature_set]
        
        return feat_set

