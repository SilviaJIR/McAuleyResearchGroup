
# coding: utf-8

# In[1]:


import numpy
import urllib
import scipy.optimize
import math
import random
import collections
from collections import defaultdict


# In[2]:


print("Reading data...")
# READ IN DATA
print("done")


# In[3]:


# Create a function which given a string of text will create dictionary 
# with all the words in the document and their word count
def feat(d):
    docDict = collections.defaultdict(int)
    for word in d.split():
        docDict[word] += 1
    return docDict


# In[4]:


# Cosine Similarity for lists of integers
def cosineSimilarity(queryVector, reviewVector):
    numerator = 0
    mag1 = 0
    mag2 = 0
    result = zip(queryVector, reviewVector);
    resultList = list(result);
    print(resultList)
    for pair in resultList:
        numerator = numerator + pair[0] * pair[1]
        mag1 = mag1 + pair[0]**2
        mag2 = mag2 + pair[1]**2
    if mag1 > 0 and mag2 > 0:
        return (numerator/(mag1*mag2)**0.5)


# In[5]:


# Cosine Similarity for text
def cosineSimilarity1(query, review):
    numerator = 0
    mag1 = 0
    mag2 = 0
    
    # Make dictionaries for the query and review
    queryDict = feat(query)
    reviewDict = feat(review)
    
    # Find the words the 2 dictionaries have in common
    querySet = set(queryDict.keys())
    reviewSet = set(reviewDict.keys())
    commonWords = querySet.union(reviewSet)
    print(commonWords)
    
    # Find the cosine similarity
    for word in commonWords:
        numerator = numerator + queryDict[word] * reviewDict[word]
        mag1 = mag1 + queryDict[word]**2
        mag2 = mag2 + reviewDict[word]**2
    if mag1 > 0 and mag2 > 0:
        return (numerator/(mag1*mag2)**0.5)


# In[6]:


# Documents from https://gist.github.com/anabranch/48c5c0124ba4e162b2e3

document_0 = "China has a strong economy that is growing at a rapid pace. However politically it differs greatly from the US Economy."
document_1 = "At last, China seems serious about confronting an endemic problem: domestic violence and corruption."
document_2 = "Japan's prime minister, Shinzo Abe, is working towards healing the economic turmoil in his own country for his view on the future of his people."
document_3 = "Vladimir Putin is working hard to fix the economy in Russia as the Ruble has tumbled."
document_4 = "What's the future of Abenomics? We asked Shinzo Abe for his views"
document_5 = "Obama has eased sanctions on Cuba while accelerating those against the Russian Economy, even as the Ruble's value falls almost daily."
document_6 = "Vladimir Putin is riding a horse while hunting deer. Vladimir Putin always seems so serious about things - even riding horses. Is he crazy?"

all_documents = [document_0, document_1, document_2, document_3, document_4, document_5, document_6]


# In[7]:


# Fill in a dictionary of dictionaries 
# The keys are the documents and the values are dictionaries with the terms and their counts
docDict = collections.defaultdict(lambda: collections.defaultdict(int))
for doc in all_documents:
    docDict[doc] = feat(doc)


# In[8]:


# Term frequency
# Returns number of times a term appears in a specific document
def tf(term, document, docDict):
    return docDict[document][term]/len(document.split())


# In[9]:


# Inverse document frequency
# Log of number of documents divided by the number of documents in the documentSet which contain the term
def idf(term, docDict):
    count = 0
    for doc in docDict:
        if term in doc:
            count += 1
    return math.log(len(docDict)/count)


# In[10]:


# Document is a string of text
# DocumentSet is an array or set
# Returns tfidf score given a term, document, and set of documents
def tfidf(term, document, docDict):
    return tf(term, document, docDict) * idf(term, docDict)


# In[11]:


# k and b are tuning parameters (EX: k = 1.2-2.0, b = 0.75)
def bm25(term, document, docDict, k, b):
    avgLength = 0
    for doc in docDict.keys():
        avgLength += len(doc.split())
    avgLength = avgLength/len(docDict)
    return idf(term, docDict) * (tf(term, document, docDict) * (k+1))/(tf(term, document, docDict) + k *(1-b+b*(len(document.split()))/avgLength))


# In[12]:


# TEST
queryVector = [2,2,3,4]
reviewVector = [6,6,3,3]
#print(cosineSimilarity(queryVector,reviewVector))

words = ["hello", "my", "name", "is", "hello", "name"]
words1 = ["is"]

#print(cosineSimilarity1(words,words1))

#print(feat(words))


# In[13]:


print(bm25("China", document_1, docDict,1.5,0.75))

