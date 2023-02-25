# response-pipeline

**response-pipeline** spans the activities of the Disaster Response Pipeline project of Udacity's Data Science nanodegree program.

 

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

    conda env create -f src/environment.yml

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

The main idea of the analysis was to understand what are the main characteristics that define a Airbnb superhost.

Having the business problem in hands and following the CRISP-DM steps, we started taking a look at the available datasets, the meaning of their fields and their data types.

