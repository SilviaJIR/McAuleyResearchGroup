
# coding: utf-8

# In[ ]:



import numpy
import random
from collections import defaultdict
import urllib
import scipy.optimize
import math
import random
import collections


# In[2]:


def parse(path):
  file = open(path, 'r')
  for l in file:
    yield eval(l)

print("Reading data...")
products = list(parse("C:/Users/Moi/Downloads/reviews.json"))
qa = list(parse("C:/Users/Moi/Downloads/qa.json"))
print("done")


# In[3]:


reviews = defaultdict(lambda: [])
for product in products:
    reviews[product["asin"]].append(product["reviewText"])


# In[7]:


def randomQuestion(qa, reviews):
    numReviews = 0
    
    
    while numReviews < 10:
        question = random.choice(qa)
        questionText = question['question']
        asin = question['asin']

        numReviews = len(reviews[asin])
        
    reviewsDoc = reviews[asin]
    
    cosSimList = {review : cosineSimilarity(questionText, review) for review in reviewDoc}
    cosSimList = sorted(cosSimList.items(), key=lambda x:x[1])
    
    reviewSample = [review[0] for review in cosSimList[:5]]
    reviewSample = reviewSample + [review[0] for review in cosSimList[-5:]]
    
    return (questionText, reviewSample)
    

# In[12]:


with open("C:/Users/Moi/Downloads/out.csv", 'w') as the_file:
    for i in range(0,10):
        question, reviewSample = randomQuestion(qa, reviews)
        for review in reviewSample:
            the_file.write(question.replace(',', ''))
            the_file.write(',')
            the_file.write(review.replace(',', ''))
            the_file.write('\n')


# In[ ]:


# Create a function which given a string of text will create dictionary 
# with all the words in the document and their word count
def feat(d):
    docDict = collections.defaultdict(int)
    for word in d.split():
        docDict[word] += 1
    return docDict


# In[ ]:


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


# In[ ]:


# Linear Regression

