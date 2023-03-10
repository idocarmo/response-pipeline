import sys

import re
import pickle
import pandas as pd

from sqlalchemy import create_engine

import nltk
nltk.download(['stopwords', 'punkt','wordnet', 'omw-1.4'])
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV

sys.path.append('../')


def load_data(database_filepath):
    """Load the training data.

    Loads the processed messages dataframe and returns is
    splited as features, targets and targets names.   

    Args:
        database_filepath:
            String with the SQL database file path.

    Returns:
        X:
            Dataframe with features for model training.
        Y:
            Dataframe with the targets for model training.
        Classes:
            List with the names of classes of training data.
    """
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('messages', con=engine)

    X = df['message']
    Y = df.loc[: , 'related':]

    return X, Y, Y.columns


def tokenize(text):
    """Tokenize text.

    remove special character, convert to lower case, tokenize and
    lemmatize the input text.   

    Args:
        text:
            Text string to be tokenized.

    Returns:
        List of normalized and lemmatized tokens of the input text.
    """
    text = re.sub(r'[^a-zA-Z]', ' ', text.lower())
    
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok, pos='v').strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    """Buld a classifier model.

    Builds a pipeline of data transforming with CountVectorizer and
    TFidfTransformer for further multi output classification with a
    Random Forest Classifier. Optmize the classifier with a grid 
    search with cross validation.   

    Args:
        None

    Returns:
        A grid search object.
    """
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize, stop_words=stopwords.words('english'))),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier(max_depth=12, class_weight='balanced_subsample')))
    ])

    parameters = {
        'vect__ngram_range': ((1, 1), (1, 2)),
        'clf__estimator__min_samples_split': [2, 3, 4],
        'clf__estimator__n_estimators': [50, 100],
    }

    cv = GridSearchCV(
        pipeline, 
        param_grid=parameters, 
        scoring='f1_micro',
        cv=3, 
        verbose=2
    )

    return cv



def evaluate_model(model, X_test, Y_test, category_names):
    """Evaluate the classifier

    Prints the precision, recall and f1 score for both negative 
    andposite classes for all categories present on the data   

    Args:
        model:
            Classifier model.
        X_test:
            Test data for categories to be predicted.
        Y_test:
            True values of test data categories.
        category_names:
            Name of the categories.

    Returns:
        None
    """
    
    Y_pred = model.predict(X_test)
    
    for i, category in enumerate(category_names):
        print('Class: ' + category)
        print(classification_report(Y_test[category], Y_pred[:, i]))
        print('')


def save_model(model, model_filepath):
    """Save the model as a picklefile  

    Args:
        model:
            Classifier model.
        model_filepah:
            String with the path of the model to be saved

    Returns:
        None
    """    
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()