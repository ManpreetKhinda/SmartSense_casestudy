# SmartSense_casestudy
Here I have discussed my approach to classify news based on their headlines.
## Dataset
Dataset link: https://www.kaggle.com/datasets/rmisra/news-category-dataset

Starting with data collection was the most crucial part as from the problem statement it can be seen that it is a classification problem involving NLP, I have gone through various datasets and found the dataset linked to this problem one of the optimal ones that can aid to understand as well as work well with the problem, the best part about this dataset is that this file contains 210,294 records between 2012 and 2022. Hence the data is less prone to timely event outliers and also the distribution is not aggressively biased through less dominant categories such as gossip, sports, trade, etc because through personal experience I have found that most newspapers media news channels focus on politics, followed by entertainment , geographical or travel knowledge, etc.

## Installation

`pip install numpy`  
`pip install scikit-learn`
`pip install jsonlines`  
`pip install nltk` 
`pip install string` 
`pip install sklearn` 
`pip install warnings` 

 With the installation of these files, one should be good to go to run the news_classification.py file

 ## Approach
 I have used the TF-IDF vectorizer for vectorizing data after text processing, here the IF-IDF vectorizer is used for information retrieval to represent a collection of documents as numerical vectors, the data is represented capturing the importance of words in the context of the entire document collections.
 Then I used the Gaussian naive Bayes model for the classification because the Bayesian models are very lightweight for our task, have low memory requirements are suitable for large datasets, and are memory efficient.
 



