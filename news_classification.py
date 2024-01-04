

#This part of code all necessary libararies
import pandas as pd
import numpy as np
import warnings
import jsonlines
import string
warnings.filterwarnings('ignore')
from sklearn.preprocessing import LabelEncoder
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
import re
from nltk import word_tokenize
nltk.download('punkt')
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')
nltk.download('omw-1.4')
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,classification_report,f1_score
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer
from sklearn.naive_bayes import GaussianNB


#This part of code traverse through
# the json object for each line it traverses it help in formation of a dataframe

data_list = []
file_path="/content/drive/MyDrive/ML-SMART SENSE/News_Category_Dataset_v3.json"
with jsonlines.open(file_path) as reader:
    for line in reader:
        data_list.append(line)
df=pd.DataFrame(data_list)

# this section is for removing unnecessary data columns which are not useful in
# headline classification

c_drop=["link","authors","date","short_description"]
df=df.drop(columns=c_drop)
df = df.dropna()
des_c=["POLITICS","WELLNESS","ENTERTAINMENT","TRAVEL"]
df=df[df['category'].isin(des_c)]
# here we are downsampling the dataset at it has more than  70000 rows which can be comutationally expensive
# also since we want to design a topology so it serves the purpose
df=df.sample(frac=0.2,random_state=42)



## the below part is sequential part where we do text processing
## Text preprocessing scheme:- lowering of alphabets=>removal of
#punctuations=>removing non alphanumeric keywords and numerical
#values=>stopword removal=>lemmetization
df['headline']=df['headline'].str.lower()
def handle_punctuation(text, replace_with=''):
  ## here we are replacing punctuations from string with "" using the .translate function
    translator = str.maketrans('', '', string.punctuation)
    cleaned_text = text.translate(translator)
    if replace_with:
        cleaned_text = cleaned_text.replace(' ', replace_with)
    return cleaned_text
# Apply the handle_punctuation function to the 'headline' column
df['headline'] = df['headline'].apply(handle_punctuation)



def clean_headline(text, replace_with=''):
  ## in this function we are replacing number and non-alphabetical characters
  ##with blank spaces using for loop for iterationg through each string
    # Remove numbers
    cleaned_text = ''.join(char if not char.isdigit() else replace_with for char in text)

    # Remove non-alphabetic characters (including spaces)
    cleaned_text = ''.join(char if char.isalpha() or char.isspace() else replace_with for char in cleaned_text)

    return cleaned_text
# Apply the clean_headline function to the 'headline' column
df['headline'] = df['headline'].apply(clean_headline)



def remove_stopwords(text):
  ## here we tokenize the text that is to break it into smaller individual string then we match these words
  ## with nltk stopwords to eliminate them
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text)
    return [x for x in words if x not in stop_words]
# Apply the remove_stopwords function to the 'headline' column
df['headline'] = df['headline'].apply(remove_stopwords)



def lemmatizing_words(text):
  ## here we are using lemmetization to reduce words from thier non base form to base form
  ##
  wordnet_lemmatizer=WordNetLemmatizer()
  return " ".join([wordnet_lemmatizer.lemmatize(word) for word in text])
df['headline']=df['headline'].apply(lemmatizing_words)


## here we encode the labels as
# politics:0
# wellness:1
# entertainment:2
#travel:3
def label_encode_column(df, column_name):
    le = LabelEncoder()
    df[column_name] = le.fit_transform(df[column_name])
    return df
df = label_encode_column(df, 'category')


##here for fitting the data and vectorization we are using numpy arrays
## we are using unigrams and bigrams for vetorization
x_df=np.array(df.iloc[:,1].values)
y_df=np.array(df.category.values)
tfidf_vec=TfidfVectorizer( ngram_range=(1,2), min_df=0.005)
x_df=tfidf_vec.fit_transform(df.headline).toarray()
x_train, x_test, y_train, y_test = train_test_split(x_df, y_df, test_size=0.1 ,random_state = 0)


gnb=GaussianNB()
gnb.fit(x_train,y_train)




pred=gnb.predict(x_test)
print(pred)

