README.md

By Mikhail Boulgakov, Lavanya Satyan, Silvia Gong
Prof. McAuley
2018

Check out the state of the Research document for more information on our
research topic and background.

This github is split into 2 folder. The folder titled Old contains our previous
versions of the files used in this project. The Final folder contains the final
versions of all our code as well as the supporting data. 

The Regression on Data file trains a linear regression model on the labelled 
input data (out.csv). This data was extracted from 
http://jmcauley.ucsd.edu/data/amazon/qa/ using votingData.ipynb and was then 
labelled by hand (Y if review was relevant to question, and N otherwise).

The output of the regression is highestReviewData2. It is the input to the
voting function, which trains a logistic regression to output either yes or no
on a question using reviews as input.

All of the ipynb files can be run in Jupyter Notebook. You will need to
download sklearn (scipy toolkit) and nltk (natural language processing toolkit)
to run this code.

FinalPoster.pdf contains a detailed description of our model.

For more information about machine learning, recommender systems, and natural
language processing check out UCSD CSE 158 - Recommender Systems and Web Mining
podcasts by Prof. McAuley.