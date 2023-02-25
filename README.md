# response-pipeline

**response-pipeline** spans the activities of the Disaster Response Pipeline project of Udacity's Data Science nanodegree program.

In this project we implemented a message classifier that aims to indicate what class a emergency/disaster message belongs to.

The classifier was trained using disaster data from [Appen](https://www.figure-eight.com).

How it works? Given a text input on the web app it returs the classes it might belong to. At total there are 36 classes that ranges from *medical aid* to *clothing*.


## Source code
You can clone this repository in your machine with the command:

    git clone https://github.com/idocarmo/response-pipeline.git

## Project Setup
### Dependencies

response-pipeline project requires:
~~~~~~~~~~~~
  - python=3.8
  - numpy
  - pandas
  - scikit-learn
  - nltk
  - sqlalchemy < 2.0
  - plotly
  - flask
~~~~~~~~~~~~

### Environment
You can prepare the pyhton environment  with all the dependencies using ``conda``:

    conda env create -f environment.yml

## Repository Content

- 📂 [app](https://github.com/idocarmo/response-pipeline/tree/main/app) contains files used in the deploy ofthe web app;
    - 📂 [templates](https://github.com/idocarmo/response-pipeline/tree/main/app/templates) contains the html templates of the web app.
    - 📄 run.py is the script with the deploying of the web app.
- 📂 [data](https://github.com/idocarmo/response-pipeline/tree/main/data) contains the files used and exported on the data processing step;
    - 📄 disaster_categories.csv is the raw data with categories names for each message
    - 📄 disaster_messages.csv is the raw data with the text messages
    - 📄 disaster_response.db is the SQL database generated when running  *process_data.py*
    - 📄 disaster_response.db is the python script with data cleaning and exporting.
- 📂 [model](https://github.com/idocarmo/response-pipeline/tree/main/model) contains the files used and exported on the classifier model building step;
    - 📄 train_classifier.py is the python script with the model building and training.
    - 📄 trained_classifier.pkl is the trained Random Forest Classifier saved when executing *train_classifier.py*
- 📄 environment.yml is the file with the instructions for python environment building.  

## How to Run

1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/disaster_response.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/trained_classifier.pkl`

2. Go to `app` directory: `cd app`

3. Run your web app: `python run.py`

4. Click the `PREVIEW` button to open the homepage

5. Insert the message you wish to classify

## About the Classifier

The model used in the project was the RandomForestClassifier implemented on the Scikit-Learn Python package.

As some of the 36 classes are extremely imbalanced, the Random Forest hyperparameter *class_weight* was set to 'balanced'. Also, to better improve the model performance on those imbalanced classes, the metric utilized on the grid search was the F1 score, that is, the harmonic mean of precision and recall.  

