import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import pickle
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

st.title('Iris Classifier')
st.write("This app uses 4 inputs to predict the species of iris using "
         "a model built on the iris dataset. Use the form below"
         " to get started!")

iris_file = st.file_uploader('Upload your own iris data')

if iris_file is None:
    rf_pickle = open('random_forest.pickle', 'rb')
    map_pickle = open('output.pickle', 'rb')

    rfc = pickle.load(rf_pickle)
    unique_iris_mapping = pickle.load(map_pickle)

    rf_pickle.close()
else:
    iris_df = pd.read_csv(iris_file)
    iris_df = iris_df.dropna()

    output = iris_df['variety']
    features = iris_df[['sepal.length', 'sepal.width', 'petal.length', 'petal.width']]

    features = pd.get_dummies(features)

    output, unique_iris_mapping = pd.factorize(output)

    x_train, x_test, y_train, y_test = train_test_split(
        features, output, test_size=.8)

    rfc = RandomForestClassifier(random_state=15)
    rfc.fit(x_train, y_train)

    y_pred = rfc.predict(x_test)

    score = round(accuracy_score(y_pred, y_test), 2)

    st.write('We trained a Random Forest model on these data,'
             ' it has a score of {}! Use the '
             'inputs below to try out the model.'.format(score))

with st.form('user_inputs'):
    sepal_length = st.number_input(
        'Iris sepal length', min_value=0.0, value=7.0)
    sepal_width = st.number_input(
        'Iris sepal width', min_value=0.0, value=4.0)
    petal_length = st.number_input(
        'Iris petal length', min_value=0.0, value=6.0)
    petal_width = st.number_input(
        'Iris petal width', min_value=0.0, value=2.0)
    st.form_submit_button()

new_prediction = rfc.predict([[sepal_length, sepal_width, petal_length,
                               petal_width,]])
prediction_species = unique_iris_mapping[new_prediction][0]
st.write('We predict your penguin is of the {} species'.format(prediction_species))
