# IMPORTING REQUIRED FILES
import pandas as pd
import pickle
import streamlit as st
import numpy as np



# CREATING A HEADER FOR BROWSER
st.set_page_config(page_title="IRIS PROJECT")



# ADDING A TITLE IN BROWSER'S BODY
st.title("{ IRIS END TO END PROJECT BY BALAPRASAD KC }")



# ADDING A SUBHEADER ASKING FOR VALUE INPUT
st.subheader('INPUT ANY VALUE BELOW : ')



# TAKING SEPAL LENGTH , SEPAL WIDTH , PETAL LENGTH , PETAL WIDTH AS INPUT
sep_len = st.number_input('SEPAL LENGTH : ', min_value=0.00, step=0.01 )
sep_wid = st.number_input('SEPAL WIDTH : ', min_value=0.00, step=0.01)
pet_len = st.number_input('PETAL LENGTH : ', min_value=0.00, step=0.01)
pet_wid = st.number_input('PETAL WIDTH : ', min_value=0.00, step=0.01)



# ADDING A PREDICT BUTTON TO PREDICT SPECIES
submit = st.button('PREDICT')



# WRITING A FUNCTION TO PREDICT SPECIES ALONG WITH PROBABILITY
def predict_species(pre_path, model_path):
    # getting the inputs in dataframe format
    xnew = pd.DataFrame([sep_len, sep_wid, pet_len, pet_wid]).T
    xnew.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']

    # loading the preprocessor with picle
    with open(pre_path, 'rb') as file1:
        pre = pickle.load(file1)
    
    # transform xnew
    xnew_pre = pre.transform(xnew)

    # loading the model with pickle
    with open(model_path, 'rb') as file2:
        model = pickle.load(file2)
    
    # getting the predictions
    preds = model.predict(xnew_pre)

    # getting the probability of predictions
    probs = model.predict_proba(xnew_pre)

    # getting maximum probability
    max_prob = np.max(probs)
    return preds, max_prob



# ADDING SUBHEADER TO SHOW RESULTS
st.subheader('RESULTS FOR THE ABOVE INPUTS ARE : ')



# PREDICTING RESULTS ON THE WEB APP AFTER SUBMIT BUTTON IS PRESSED
if submit:
    # getting pre path and model path
    pre_path = "IRIS END TO END PROJECT/preprocessor.pkl"
    model_path = "IRIS END TO END PROJECT/model.pkl"
    # getting the predictions along with probability
    pred, max_prob = predict_species(pre_path, model_path)

    # printing the results
    st.subheader(f'{pred[0]} is the predicted species  ')
    st.subheader(f'probability of prediction is {max_prob:.4f}')

    # showing probability in progress bar
    st.progress(max_prob)
