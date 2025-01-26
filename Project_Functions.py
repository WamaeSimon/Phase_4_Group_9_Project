#!/usr/bin/env python
# coding: utf-8

# In[9]:



# Importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# from wordcloud import WordCloud
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Download NLTK resources
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')


# #### General Exploratory

# In[29]:



# Function to Load the data
def Load_dataset(data):
    df = pd.read_csv(data, encoding='latin1')
    return df

#show the shape of the dataset
def data_shape(df):
    print(df.shape)
    
#Check for the dataset information   
def check_Info(df):
    print("===============The dataset Shape=================")
    print(df.shape)
    print("===============The dataset columns=================")
    print(df.columns)
    print("===============The data_types=================")
    print(df.dtypes)
    print("===============The dataset information=================")
    print(df.info())
    print("===============Check for Missing values=================")
    print (df.isnull().sum())
    print("===============Check for Duplicated Rows=================")
    print(df.duplicated().sum())
    print("===============The dataset Description=================")
    print(df.describe())
    

#Show the columns
def Columns(df):
    print(df.columns)
    
# Check the columns data_types
def data_types(df):
    print(df.dtypes)
    
# Check for missing values
def check_for_missing_values(df):
    print (df.isnull().sum())
    
# Check for the duplicates
def Duplicates(df):
    print(df.duplicated().sum())
    
# The description of the data(descriptives)
def Describe_data(df):
    print(df.describe())


