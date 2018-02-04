
# coding: utf-8

# In[1]:


import numpy
import random
from collections import defaultdict


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
        

    reviewSample = random.sample(reviews[asin], 5)
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

