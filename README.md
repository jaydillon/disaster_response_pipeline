# Disaster Response Pipeline
Analyze disaster data from Figure Eight to build a model for an API that classifies disaster messages.

## Table of Contents
1. Description
2. Dependencies
3. Files
4. Authors
5. License
6. Acknowledgement

## Description
This Project is part of Data Science Nanodegree Program by Udacity in collaboration with Figure Eight. The dataset contains pre-labelled tweet and messages from real-life disaster events. The project aim is to build a Natural Language Processing (NLP) model to categorize messages on a real time basis.

This project is divided in the following key sections:

* Processing data, building an ETL pipeline to extract data from source, clean the data and save them in a SQLite DB
* Build a machine learning pipeline to train the which can classify text message in various categories
* Run a web app which can show model results in real time

## Dependencies
* Python 3.5+
* Machine Learning Libraries: NumPy, SciPy, Pandas, Sciki-Learn
* Natural Language Process Libraries: NLTK
* SQLlite Database Libraqries: SQLalchemy
* Model Loading and Saving Library: Pickle
* Web App and Data Visualization: Flask, Plotly

## Files
1. **app/templates/***: templates/html files for web app
2. **data/process_data.py**: Extract Train Load (ETL) pipeline used for data cleaning, feature extraction, and storing data in a SQLite database
3. **models/train_classifier.py**: A machine learning pipeline that loads data, trains a model, and saves the trained model as a .pkl file for later use
4. **run.py**: This file can be used to launch the Flask web app used to classify disaster messages

## Authors
Jay Ding

## License
License: MIT

## Acknowledgements
Udacity for providing an amazing Data Science Nanodegree Program
Figure Eight for providing the relevant dataset to train the model
