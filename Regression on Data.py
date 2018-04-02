
# coding: utf-8

# In[1]:


import numpy
import random
from collections import defaultdict
import urllib
import math
import random
import collections
import string
import csv
import sys
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn import svm
import sklearn.metrics
from sklearn.svm import LinearSVC
from nltk.corpus import stopwords


# In[2]:


def parseLabeledData(path):
    file=open(path, 'r')
    dataList = []
    for line in csv.reader(file):        
        if len(line) >= 5:
            dataList.append(
                {"asin":line[0], 
                 "question":line[1],
                 "review":line[2],
                 "answer":line[3],
                 "label":line[4]}
            )     
    return dataList
        

print("Reading labeled data...")
data = parseLabeledData("C:/Users/Moi/Downloads/out.csv")
queries = [d['question'] for d in data]
answers = [d['answer'] for d in data]
reviews = [d['review'] for d in data]
labels = [d['label'] for d in data]
print("done")


# In[3]:


def parseAllQueries(path):
    file = open(path, 'r')
    dataList = defaultdict(lambda: [])
    for line in file:
        line = eval(line)
        dataList[line["asin"]].append(line)
      
    return dataList

def parseAllReviews(path):
    file = open(path, 'r')
    dataList = defaultdict(lambda: [])
    for line in file:
        line = eval(line)
        dataList[line["asin"]].append(line)
      
    return dataList

print("Reading all reviews & all questions...")
allReviews = parseAllReviews("C:/Users/Moi/Downloads/reviews.json")
allQuestions = parseAllQueries("C:/Users/Moi/Downloads/qa.json")

# do we have to remove questions that have no reviews or reviews that have no questions??
docSet = []
for entry in allReviews.values():
    for review in entry:
        docSet.append(review["reviewText"])

for entry in allQuestions.values():
    for question in entry:
        docSet.append(question["question"])

docLen = [len(d.split()) for d in docSet]
avgdl = sum(docLen) / len(docLen)

print("done")


# In[5]:


def countAllWords():
    allWords = defaultdict(int)
    englishStopWords = stopwords.words('english')
    for r in allReviews.values():
        for review in r:
            review = review["reviewText"]
            exclude = set(string.punctuation)
            review = ''.join(ch for ch in review if ch not in exclude)
            for w in review.lower().split():
                if w not in englishStopWords:
                    allWords[w] += 1

    for q in allQuestions.values():
        for question in q:
            question = question["question"]
            exclude = set(string.punctuation)
            question = ''.join(ch for ch in question if ch not in exclude)
            for w in question.lower().split():
                if w not in englishStopWords:
                    allWords[w] += 1
    
    
    return allWords

allWords = countAllWords()


# In[6]:


commonWords = sorted(allWords, key=lambda x: -allWords[x])[:1000]


# In[7]:


# idfDict = defaultdict(float)
#for word in commonWords:
#    count = 0         
#    for doc in docSet:
#        if word in doc.lower():
#            count += 1
#    idfScore = math.log(len(docSet)/(count+1))
#     
#    idfDict[word] = idfScore


# In[4]:


# @param a word whose frequency in the document we are calculating
# @param document a string of a review or a question
# @return the frequency of term in document div length of document

def tf(term, document):
    count = collections.defaultdict(int)
    exclude = set(string.punctuation)
    document = ''.join(ch for ch in document if ch not in exclude)
    for word in document.split():
        count[word] += 1

    return count[term]/(len(document.split()) + 1)


# In[90]:


idfDict = defaultdict(float)

def idf(term):
    term = term.lower()
    if (term in idfDict):
        return idfDict[term]

    count = 0
    for doc in docSet:
        #exclude = set(string.punctuation)
        #doc = ''.join(ch for ch in doc if ch not in exclude)
        if term in doc.lower():
            count += 1
        
    idfScore = math.log(1 + len(docSet) / (count+1))
    idfDict[term] = idfScore
    return idfScore


# In[109]:


okapidict = {}

def OkapiBM25(review, question, k1, b):
    if ((review, question, k1, b) in okapidict):
        return okapidict[review, question, k1, b]
    
    question = question.lower()
    question = ''.join([c for c in question if not (c in string.punctuation)])
    
    score = 0
    for q in question.split():
        num = tf(q, review) * (k1 + 1)
        den = tf(q, review) + k1 * (1 - b + b*len(review.split()) / avgdl) 
        score += idf(q) * num / den
        
    #print(score, review, question)
    okapidict[review, question, k1, b] = score
    return score


# In[121]:


tfidfdict = {}

def tfidf(document):
    if (document in tfidfdict):
        return tfidfdict[document]
    
    doc = document.lower()
    doc = ''.join([c for c in doc if not (c in string.punctuation)])
        
    feat = collections.defaultdict(int)
    for term in doc.split():
        tfscore = tf(term, doc)
        idfscore = idf(term)
        feat[term] = tfscore * idfscore
        
    tfidfdict[document] = feat
    return feat


# In[122]:


def wordToIndex(term):
    if term in commonWords:
        return commonWords.index(term)
    else:
        return -1


# In[232]:


def numCommonWords(review, question):
    review = review.lower()
    review = ''.join([c for c in review if not (c in string.punctuation)])
    
    question = question.lower()
    question = ''.join([c for c in question if not (c in string.punctuation)])
    
    filtered_words = [word for word in question.split() if word not in stopwords.words('english')]
    words = set(filtered_words)
    
    num = 0
    for word in words:        
        if word in review:
            num += 1
  
    #print(num)
    return num

#numCommonWords("This is a red and blue review about a car", "I am, a BLUE and yellow quESTION about a car!")


# In[233]:


def lengthDiff(review, question):
    return abs(len(review.split()) - len(question.split()))


# In[234]:


# queryFeat is a feature vector for the query and reviewFeat is the feature vector for the review
def cosineSimilarity(queryFeat, reviewFeat):
    # Find the words the 2 dictionaries have in common
    querySet = set(queryFeat.keys())
    reviewSet = set(reviewFeat.keys())
    allWords = querySet.union(reviewSet)
    
    # Find the cosine similarity
    numerator = 0
    mag1 = 0
    mag2 = 0
    for word in allWords:
        numerator = numerator + queryFeat[word] * reviewFeat[word]
        mag1 = mag1 + queryFeat[word]**2
        mag2 = mag2 + reviewFeat[word]**2
    if mag1 > 0 and mag2 > 0:
        return (numerator/((mag1*mag2)**0.5))
    else:
        return -1


# In[235]:


def uniqueWords(review, question, num):
    exclude = set(string.punctuation)
    question = ''.join(ch for ch in question if ch not in exclude)
    review = ''.join(ch for ch in review if ch not in exclude)
    review = review.lower()
    
    qFreq = {word : allWords[word] for word in question.lower().split()}
    
    topUnique = [word for word in sorted(qFreq, key=lambda x: qFreq[x]) if allWords[word] != 0]
    
    if num <= len(topUnique):
        topUnique = topUnique[:num]
    else:
        topUnique += [''] * (num - len(topUnique))
    
    #print(qFreq)
    #print(topUnique)
    
    feat = []
    for word in topUnique:
        #feat.append(review.split().count(word))
        feat.append(1 if word in review and word != '' else 0)
        
    return feat

#uniqueWords("The color of this item is RED red red Great", "Hello, this color ReD? I think great", 10)


# In[236]:


def constrain(elems, point):    
    for elem in elems:
        if (elem > point): yield 1
        else: yield 0
        #if elem > 1: yield 1
        #elif elem < 0: yield 0
        #else: yield elem


# In[237]:


def normalize(featList):
    
    max = 0
    min = float('inf')
    for feat in featList:
        if feat > max: max = feat
        if feat < min: min = feat        
    
    for i in range(0,len(featList)-1):
        if (max - min) == 0: 
            max = 1
            min = 0
        featList[i] = (featList[i] - min) / (max - min)

    return featList


# In[238]:


def feature_tfidf(review, question):
    feat = [1]
    feat.append(cosineSimilarity(tfidf(review), tfidf(question)))
    return feat


# In[239]:


def feature_okapi(review, question):
    feat = [1] 
    feat.append(OkapiBM25(review, question, 1.5, 0.75))
    return feat


# In[250]:


def feature(review, question, length):
    feat = [1]
    
    #number of Common Words
    #difference in length
    #length of review
    #length of question
    feat.append(numCommonWords(review, question))
    #feat.append(lengthDiff(review,question))
    #feat.append(len(review.split()))
    #feat.append(len(question.split()))
    cosine = cosineSimilarity(tfidf(review), tfidf(question))
    feat.append(cosine)
    feat.append(OkapiBM25(review, question, 1.5, 0.75))
    feat = feat + uniqueWords(review, question, length)
    
    return feat


# In[251]:


def test(y, y_hat):
    #print(sklearn.metrics.r2_score(y, y_hat))
    
    
    accuracy = sklearn.metrics.accuracy_score(y, y_hat)
    precision = sklearn.metrics.precision_score(y, y_hat)
    recall = sklearn.metrics.recall_score(y, y_hat)
    
    return "{0:.2f}, {1:.2f}, {2:.2f}".format(accuracy, precision, recall)

def pipeline(X, y, breakpoint=0.15):
    random.seed(4410)
    
    for j in range(0,len(X[0])):
        featList = []
        for i in range(0,len(X)-1):
            featList.append(X[i][j])
        featList = normalize(featList)
        for i in range(0,len(X)-1):
            X[i][j] = featList[i]
    
    keys = list(range(1, len(labels)))
    points = dict(zip(keys, zip(X, y)))
    random.shuffle(keys)
    X_rand = [points[key][0] for key in keys]
    y_rand = [points[key][1] for key in keys]
    
    X_train = X[:len(X_rand)*2//3]
    y_train = y[:len(y_rand)*2//3]
    
    X_test = X[len(X_rand)*2//3:]
    y_test = y[len(y_rand)*2//3:]
       
    
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_hatTrain = lr.predict(X_train)
    y_hatTest = lr.predict(X_test)
    
    #print(X_train)
    
    #yesmean = 0
    #yesnum = 0
    #nomean = 0
    #nonum = 0
    #for elem, label in zip(y_hatTrain, y_train):
    #    if label == 1:
    #        yesmean = yesmean + elem
    #        yesnum += 1
    #    else:
    #        nomean = nomean + elem
    #        nonum += 1
    #    #print(elem, " ", label)
    # 
    #print(yesmean, yesnum, nomean, nonum)
    #point = ((yesmean / yesnum) + (nomean / nonum)) / 2
    #print(point)
    
    return (y_hatTrain, y_hatTest, y_train, y_test)
    
    y_hatTrain = list(constrain(y_hatTrain, breakpoint))
    y_hatTest = list(constrain(y_hatTest, breakpoint))
    
    # Test percentage that the top 3 relevant reviews are relevant.
    for i in range(0, len(X) - 1):
        x = X[i]
        label = y[i]
        pred = lr.predict([x])
        review = data[i]["review"]
        query = data[i]["question"]
        
        #print(label, pred, query, review)
    
    
    train_string = test(y_train, y_hatTrain)
    test_string = test(y_test, y_hatTest)
    
    print(train_string, "\t", test_string, "\n")
    

for length in range(0, 5):
    print("Our model looking at", length, "unique words:")
    X = [feature(d["review"], d["question"], length) for d in data]
    y = [1 if l == "Y" else 0 for l in labels]
    y_hatTrain, y_hatTest, y_train, y_test = pipeline(X, y)
    
    for breakpoint in range(0, 10):
    
        y_hatTrain_c = list(constrain(y_hatTrain, breakpoint / 20))
        y_hatTest_c = list(constrain(y_hatTest, breakpoint / 20))

        train_string = test(y_train, y_hatTrain_c)
        test_string = test(y_test, y_hatTest_c)

        print(breakpoint / 20, ":\t", train_string, "\t", test_string)
    print()
    


print("TF-IDF baseline")
X = [feature_tfidf(d["review"], d["question"]) for d in data]
y = [1 if l == "Y" else 0 for l in labels]
pipeline(X, y)


print("Okapi BM25 baseline")
X = [feature_okapi(d["review"], d["question"]) for d in data]
y = [1 if l == "Y" else 0 for l in labels]
pipeline(X, y)


print("Always false")
random.seed(171727)
y = [1 if l == "Y" else 0 for l in labels]
keys = list(range(1, len(labels)))
points = dict(zip(keys, zip(X, y)))
random.shuffle(keys)
y = [points[key][1] for key in keys]
y_train = y[:len(y)//2]
y_test = y[len(y)//2:]

y_hatTrain = [0 for y in y_train]
y_hatTest = [0 for y in y_test]

train_string = test(y_train, y_hatTrain)
test_string = test(y_test, y_hatTest)    
print("training:", train_string, "\t test:", test_string)
print("\n")

