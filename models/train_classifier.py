import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

# import libraries
import numpy as np
import pandas as pd

import sys
import os
import re
from sqlalchemy import create_engine
import pickle

from scipy.stats import gmean
# import relevant functions/modules from the sklearn
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import fbeta_score, make_scorer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.base import BaseEstimator,TransformerMixin


def load_data_from_db(database_filepath):
    """
    Load Data from the SQLite database

    Arguments:
        database_filepath: Path to SQLite database

    Output:
        X: dataframe containing features
        y: dataframe containing labels
        category_names: List of categories name
    """

    engine = create_engine('sqlite:///' + database_filepath)
    table_name = database_filepath.replace(".db","") + "_table"
    df = pd.read_sql_table(table_name,engine)

    #Remove child alone as it has all zeros only
    df = df.drop(['child_alone'],axis=1)

    # Given value 2 in the related field are neglible so it could be error. Replacing 2 with 1 to consider it a valid response.
    # Alternatively, we could have assumed it to be 0 also. In the absence of information I have gone with majority class.
    df['related']=df['related'].map(lambda x: 1 if x == 2 else x)

    X = df['message']
    y = df.iloc[:,4:]

    category_names = y.columns

    return X, y, category_names


def tokenize(text,url_place_holder_string="urlplaceholder"):
    """
    Tokenize the text function

    Arguments:
        text: orginal text message
    Output:
        clean_tokens: List of tokens extracted from the original text
    """

    # Replace all urls with a urlplaceholder string
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'

    # Extract all the urls from the provided text
    detected_urls = re.findall(url_regex, text)

    # Replace url with a url placeholder string
    for detected_url in detected_urls:
        text = text.replace(detected_url, url_place_holder_string)

    # Extract the word tokens from the provided text
    tokens = nltk.word_tokenize(text)

    #Lemmanitizer to remove inflectional and derivationally related forms of a word
    lemmatizer = nltk.WordNetLemmatizer()

    # List of clean tokens
    clean_tokens = [lemmatizer.lemmatize(w).lower().strip() for w in tokens]

    return clean_tokens


# Build a custom transformer which extracts the starting verb of a sentence
class StartingVerbExtractor(BaseEstimator, TransformerMixin):
    """
    Starting Verb Extractor class

    This class extract the starting verb of a sentence,
    creating a new feature for the ML classifier
    """

    def starting_verb(self, text):
        sentence_list = nltk.sent_tokenize(text)
        for sentence in sentence_list:
            pos_tags = nltk.pos_tag(tokenize(sentence))
            first_word, first_tag = pos_tags[0]
            if first_tag in ['VB', 'VBP'] or first_word == 'RT':
                return True
        return False

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_tagged = pd.Series(X).apply(self.starting_verb)
        return pd.DataFrame(X_tagged)


def build_model():
    """
    Build a ML pipeline

    Output:
        A ML Pipeline that process text messages and apply a classifier from sklearn.
    """

    pipeline = Pipeline([
        ('features', FeatureUnion([

            ('text_pipeline', Pipeline([
                ('count_vectorizer', CountVectorizer(tokenizer=tokenize)),
                ('tfidf_transformer', TfidfTransformer())
            ])),

            ('starting_verb_transformer', StartingVerbExtractor())
        ])),

        ('classifier', MultiOutputClassifier(AdaBoostClassifier()))
    ])

    # Add GridSearchCV to find optimal parameters
    parameters = {
        'classifier__estimator__learning_rate': [0.01, 0.02, 0.05],
        'classifier__estimator__n_estimators': [10, 20, 40]
        }

    cv = GridSearchCV(pipeline, param_grid=parameters, scoring='f1_micro', n_jobs=-1)

    return cv


def evaluate_model(model, X_test, y_test, category_names):
    """
    This function applies a ML pipeline to a test set and prints out the model performance (accuracy and f1score)

    Arguments:
        pipeline -> A valid scikit ML Pipeline
        X_test -> Test features
        Y_test -> Test labels
        category_names -> label names (multi-output)
    """

    y_pred = model.predict(X_test)
    overall_accuracy = (y_pred == y_test).mean().mean()

    print('Average overall accuracy {0:.2f}%'.format(overall_accuracy*100))

    # Print the whole classification report.
    y_pred = pd.DataFrame(y_pred, columns = y_test.columns)

    for column in y_test.columns:
        print('Model Performance with Category: {}'.format(column))
        print(classification_report(y_test[column],y_pred[column]))


def save_model_as_pickle(model, pickle_filepath):
    """
    Save Pipeline function

    This function saves trained model as Pickle file, to be loaded later.

    Arguments:
        pipeline -> GridSearchCV or Scikit Pipelin object
        pickle_filepath -> destination path to save .pkl file

    """
    pickle.dump(model, open(pickle_filepath, 'wb'))


def main():
    """
    Main function to apply Machine Learning Pipeline:
        1) Extract data from SQLite db
        2) Train ML model on training set
        3) Estimate model performance on test set
        4) Save trained model as .pkl file
    """

    if len(sys.argv) == 3:
        database_filepath, pickle_filepath = sys.argv[1:]
        print('Loading data from {} ...'.format(database_filepath))
        X, y, category_names = load_data_from_db(database_filepath)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        print('Training pipeline ...')
        model = build_model()
        model.fit(X_train, y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, y_test, category_names)

        print('Saving pipeline to {} ...'.format(pickle_filepath))
        save_model_as_pickle(model, pickle_filepath)

        print('The ML model is saved!')

    else:
         print("Please provide the arguments correctly: \nSample Script Execution:\n\
            > python train_classifier.py ../data/disaster_response_db.db classifier.pkl \n\
            Arguments Description: \n\
            1) Path to SQLite destination database (e.g. disaster_response_db.db)\n\
            2) Path to pickle file name where ML model needs to be saved (e.g. classifier.pkl")


if __name__ == '__main__':
    main()
