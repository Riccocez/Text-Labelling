
# coding: utf-8

# In[1]:

import os
import re
import nltk
from Preprocessing import Preprocessing as preproc


# In[ ]:

class Files(object):
    
    def __init__(self, files = [], super_tag_dict = {}):
        """
        * A Files object containing a set of files from a given directory
        """
        
        self.files = files
        self.tmpFile = ''
        self.abstracts = []
        self.intros = []
        self.label_abstract = []
        self.label_intro = []
        self.abs_tag_freq = []
        self.intro_tag_freq = []
        self.super_tag_dict = super_tag_dict
        
    def get_sample_file(self,path,extension):
        """
        * Returns a sample file to be shown in the Main Notebook
        """
        
        for fileName in os.listdir(path):
            if self.filter_file(fileName, extension):
                sample = self.get_file(path,fileName)
                sample = sample[0:416] + sample[1000:1376] 
                return sample
                
            
    def get_all_files(self, path, extension):
        """
        * Gets all files from the path and extension given
        """
        
        for fileName in os.listdir(path):
            
            if self.filter_file(fileName, extension):
                tmpFile = self.get_file(path, fileName)
                self.files.append(tmpFile)
                
        return
    
    def filter_file(self, fileName, extension):
        """
        * Detects if a fileName corresponds to the extension desired
        """
        
        if fileName.endswith(extension):
            return True
        
        else:
            return False
        
    def get_file(self,path,fileName):
        """
        * Gets a file from the given path and fileName
        """
        
        tmpFile = ''
        with open(path + fileName,"r") as f:
            
            for line in f.read():
                tmpFile += line
                
        return tmpFile
    
    def split_abstract_intro_files(self):
        """
        * Splits each file from self.files into abstract and introduction
        """
        
        for fileText in self.files:
            
            self.split_abstract_intro_file(fileText)
                
        return
            
    def split_abstract_intro_file(self,text):
        """
        * Splits the text file into abstract and introduction
        """
        
        tmp_text = text.split("\n###")
        self.store_abstract_intro(tmp_text)
        
        return 
        
  
    def store_abstract_intro(self,tmp_text): 
        """
        * Stores the abstract and introduction within the Files object
        """
         
        section_1 = 'abstract'
        section_2 = 'introduction'
        
        tmp_abst = tmp_text[0].split('###\n')[1]
        tmp_intro = tmp_text[1].split('###\n')[1]
        
        self.store_section(tmp_abst,section_1)
        self.store_section(tmp_intro,section_2)
        
        return
    
    def store_section(self,section,sectionName):
        """
        * Stores the labels, tag frequencies and raw texts
        * of a given section according to its sectionName
        """
        
        preprocess = preproc()
        
        tokenized_section = preprocess.tokenize_section( 
                                section)
        
        if sectionName is "abstract":
            
            self.abstracts.append(tokenized_section)
            label_abstract,abs_tag_freq = preprocess.label_data(tokenized_section)
            self.label_abstract.append(label_abstract)
            self.abs_tag_freq.append(abs_tag_freq)
            
        elif sectionName is "introduction":
             
            self.intros.append(tokenized_section)
            label_intro,intro_tag_freq = preprocess.label_data(tokenized_section)
            self.label_intro.append(label_intro)
            self.intro_tag_freq.append(intro_tag_freq)
        
        return
    
    def get_tag_sets(self):
        """
        *  Returns a list of tag frequencies of both
        *  the introduction and the abstract section
        """
        
        tag_sets = [self.abs_tag_freq,self.intro_tag_freq]
        
        return tag_sets
        


# In[ ]:



