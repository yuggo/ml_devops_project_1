# library doc string
'''
Testing and Logging Script for
Project 1 of Machine Learning Dev Ops
'''

# import libraries
from ast import Assert
from math import ceil

import os
import logging
import glob
import churn_library as cl

logging.basicConfig(
    filename='./logs/churn_library.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')


def test_import(import_data):
    '''
    test data import - this example is completed for you to assist with the other test functions
    '''
    try:
        df = import_data("./data/bank_data.csv")
        logging.info("Testing import_data: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing import_eda: The file wasn't found")
        raise err

    try:
        assert df.shape[0] > 0
        assert df.shape[1] > 0
    except AssertionError as err:
        logging.error(
            "Testing import_data: The file doesn't appear to have rows and columns")
        raise err


def test_eda(perform_eda, import_data):
    '''
    test perform eda function
    '''
    df = import_data("./data/bank_data.csv")
    try:
        files = glob.glob("./images/eda/*")
        if len(files) == 0:
            logging.info("Testing EDA Folder Empty")
        else:
            logging.info("Testing EDA Folder Has Files")
    except FileNotFoundError as err:
        logging.info("Testing EDA: FAIL NO FOLDER FOUND")

    try:
        for file in files:
            os.remove(file)
        logging.info("Testing EDA Emptying Files: SUCCESS")
    except BaseException:
        logging.info("Testing EDA: FAIL DELETING FILES")

    try:
        perform_eda(df)
        files = glob.glob("./images/eda/*")
        assert len(files) == 5
        logging.info("Testing EDA Exploratory Data Analysis: SUCCESS")
    except AssertionError as err:
        logging.info(
            "Testing EDA Exploratory Data Analysis: FAIL EDA PLOT NUMBER")


def test_encoder_helper(encoder_helper, import_data):
    '''
    test encoder helper
    '''
    df = import_data("./data/bank_data.csv")
    cat_columns = [
        'Gender',
        'Education_Level',
        'Marital_Status',
        'Income_Category',
        'Card_Category'
    ]
    df['Churn'] = df['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)

    try:
        encoded_df = encoder_helper(df, cat_columns, "_cat")
        logging.info(
            "Testing Encoding: SUCCESS RUNNIN ENCODER HELPER - \
				encoder_helper(df, cat_columns, 'cat'")

        assert encoded_df.equals(df) is False
        assert encoded_df.columns.equals(df.columns) is False
        assert len(encoded_df.columns) == (len(df.columns) + len(cat_columns))

    except AssertionError as err:
        logging.info(
            "Testing Encoding: FAILED ASSERTION - \
				encoder_helper(df, cat_columns, 'cat')")

    except BaseException:
        logging.info(
            "Testing Encoding: FAILED RUNNIN ENCODER HELPER - \
				encoder_helper(df, cat_columns, 'cat')")

    try:
        encoded_df = encoder_helper(df, cat_columns)
        logging.info(
            "Testing Encoding: SUCCESS RUNNIN ENCODER HELPER - \
				encoder_helper(df, cat_columns)")

        assert encoded_df.equals(df) is False
        assert encoded_df.columns.equals(df.columns) is True

    except AssertionError as err:
        logging.info(
            "Testing Encoding: FAILED ASSERTION - encoder_helper(df, cat_columns)")

    except BaseException:
        logging.info(
            "Testing Encoding: FAILED RUNNIN ENCODER HELPER - \
				encoder_helper(df, cat_columns)")

    try:
        encoded_df = encoder_helper(df, [])
        logging.info(
            "Testing Encoding: SUCCESS RUNNIN ENCODER HELPER - \
				encoder_helper(df, [])")

        assert encoded_df.equals(df) is True
        assert encoded_df.columns.equals(df.columns) is True

    except AssertionError as err:
        logging.info(
            "Testing Encoding: FAILED ASSERTION - encoder_helper(df, [])")

    except BaseException:
        logging.info(
            "Testing Encoding: FAILED RUNNIN ENCODER HELPER - \
				encoder_helper(df, [])")


def test_perform_feature_engineering(perform_feature_engineering, import_data):
    '''
    test perform_feature_engineering
    '''
    df = import_data("./data/bank_data.csv")

    try:
        X_train, X_test, y_train, y_test = perform_feature_engineering(
            df, response="_cat")
        logging.info(
            "Testing Feature Engineering: SUCCESFUL RUNNING perform_feature_engineering(...))")

        assert X_train is not None
        assert (X_test.shape[0] == ceil(df.shape[0] * 0.3)) is True

    except AssertionError as err:
        logging.info("Testing Feature Engineering: FAILED Assertion")

    except BaseException:
        logging.info(
            "Testing Feature Engineering: FAILED RUNNING perform_feature_engineering(...))")


def test_train_models(train_models, import_data, perform_feature_engineering):
    '''
    test train_models
    '''
    df = import_data("./data/bank_data.csv")
    X_train, X_test, y_train, y_test = perform_feature_engineering(
        df, response="_cat")

    try:
        files = glob.glob("./images/results/*")
        if len(files) == 0:
            logging.info("Testing Model Results Folder Empty")
        else:
            logging.info("Testing Model Results Folder Has Files")
    except FileNotFoundError as err:
        logging.info("Testing Model Results: FAIL NO FOLDER FOUND")

    try:
        for file in files:
            os.remove(file)
        logging.info("Testing Model Results Emptying Files: SUCCESS")
    except BaseException:
        logging.info("Testing Model Results: FAIL DELETING FILES")

    try:
        train_models(X_train, X_test, y_train, y_test)
        logging.info("Testing Model Results: SUCCESS")

        files = glob.glob("./images/results/*")
        assert len(files) == 4
        logging.info("Testing Model Results - images: SUCCESS")

    except AssertionError as err:
        logging.info("Testing Model Results: FAIL PLOT NUMBER")

    except BaseException:
        logging.info("Testing Model Results: MODEL FAILED")

    try:
        files_models = glob.glob("./models/*")
        assert len(files_models) == 2
        logging.info("Testing Model Results - model files: SUCCESS")

    except AssertionError as err:
        logging.info("Testing Model Results: FAIL MODEL OUTPUT")


if __name__ == "__main__":
    test_import(cl.import_data)
    # df = cl.import_data()
    test_eda(cl.perform_eda, cl.import_data)
    test_encoder_helper(cl.encoder_helper, cl.import_data)
    test_perform_feature_engineering(
        cl.perform_feature_engineering, cl.import_data)
    test_train_models(
        cl.train_models,
        cl.import_data,
        cl.perform_feature_engineering)
