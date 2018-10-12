# -*- coding: utf-8 -*-
"""
Created on Mon Jul 23 22:03:11 2018

@author: alex
"""

import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.feature_extraction.text import TfidfVectorizer
from time import time

newsgroups = datasets.fetch_20newsgroups(
                    subset='all', 
                    categories=['alt.atheism', 'sci.space']
             )
def size_mb(docs):
    return sum(len(s.encode('utf-8')) for s in docs) / 1e6

newsgroups_size_mb = size_mb(newsgroups.data)
t0 = time()
vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(newsgroups.data)
duration = time()-t0
print("done in %fs at %0.3fMB/s" % (duration, newsgroups_size_mb / duration))