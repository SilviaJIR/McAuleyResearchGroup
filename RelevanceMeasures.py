import numpy
import urllib
import scipy.optimize
import math
import random
from collections import defaultdict

print("Reading data...")
# READ IN DATA
print("done")

# Given a document (string of text), creates a dictionary 
# with all the words in the document and their word count

def feat(d):
    docDict = collections.defaultdict(int)
    # USE SLICING AND REMOVE PUNCTUATION TO GET THE WORDS
    for word in d:
        docDict[word] += 1
    return docDict


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


# Cosine Similarity for text
def cosineSimilarity(query, review):
    numerator = 0
    mag1 = 0
    mag2 = 0
    
    # Make dictionaries for the query and review
    queryDict = feat(query)
    reviewDict = feat(review)
    
    # Find the words the 2 dictionaries have in common
    querySet = set(queryDict.keys())
    reviewSet = set(reviewDict.keys())
    commonWords = querySet.intersection(reviewSet)
    print(commonWords)
    
    # Find the cosine similarity
    for word in commonWords:
        numerator = numerator + queryDict[word] * reviewDict[word]
        mag1 = mag1 + queryDict[word]**2
        mag2 = mag2 + reviewDict[word]**2
    if mag1 > 0 and mag2 > 0:
        return (numerator/(mag1*mag2)**0.5)


# GET THE DOCUMENT SET FROM THE DATA
D = []


# Fill in a dictionary of dictionaries 
# The keys are the documents and the values are dictionaries with the terms and their counts
docDict = collections.defaultdict(lambda: collections.defaultdict(int))
for doc in D:
    docDict[doc] = feat(doc)


# Term frequency
# Returns number of times a term appears in a specific document
def tf(term, document, docDict):
    return docDict[document[term]]


# Inverse document frequency
# Log of number of documents divided by the number of documents in the documentSet which contain the term
def idf(term, documentSet, docDict):
    count = 0
    for doc in docSet:
        if docSet[doc[term]] in docSet:
            count += 1
    return math.log(len(documentSet)/count)


# Document is a string of text
# DocumentSet is an array or set
# Returns tfidf score given a term, document, and set of documents
def tfidf(term, document, documentSet, docDict):
    tf(term, document, documentSet, docDict) * idf(term, documentSet, docDict)


def bm25():
    return;


# TEST
queryVector = [1,2,3,4]
reviewVector = [6,6,3,3]
print(cosineSimilarity(queryVector,reviewVector))

words = ["hello", "my", "name", "is", "hello", "name"]
print(feat(words))
