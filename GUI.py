import streamlit as st
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import openai

# try:
#     import openai
# except ImportError:
#     st.error("The 'openai' module is not installed. Please install it using 'pip install openai'.")

# # Rest of your Streamlit app code


#Load the Model and Scaler
model = pickle.load(open('Best_Model_LogReg.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

# Define a dictionary to map the column names to the original names
Unique_Values = np.load('unique_values.npy', allow_pickle=True)

#Assigning the unique value lists
gender = Unique_Values[()]['gender']
age =  Unique_Values[()]['age']
hypertension = Unique_Values[()]['hypertension']
heart_disease = Unique_Values[()]['heart_disease']
ever_married =   Unique_Values[()]['ever_married']
work_type = Unique_Values[()]['work_type']
Residence_type = Unique_Values[()]['Residence_type']
avg_glucose_level = Unique_Values[()]['avg_glucose_level']
bmi	 =  Unique_Values[()]['bmi']
smoking_status = Unique_Values[()]['smoking_status']
age_group =	 Unique_Values[()]['age_group']
bmi_group = Unique_Values[()]['bmi_group']
avg_glucose_level_group = Unique_Values[()]['avg_glucose_level_group']

#Load Label Encoders
LE_Age_Group = pickle.load(open('LabelEncoder_age_group.pkl', 'rb'))
LE_Avg_Glucose_Level_Group = pickle.load(open('LabelEncoder_avg_glucose_level_group.pkl', 'rb'))
LE_Bmi_Group = pickle.load(open('LabelEncoder_bmi_group.pkl', 'rb'))
LE_Ever_Married = pickle.load(open('LabelEncoder_ever_married.pkl', 'rb'))
LE_Gender = pickle.load(open('LabelEncoder_gender.pkl', 'rb'))
LE_Residence_Type = pickle.load(open('LabelEncoder_Residence_type.pkl', 'rb'))

# Define a function to preprocess the input data
def preprocess_input(data):
    data['age_group'] = LE_Age_Group.transform(data['age_group'])
    data['avg_glucose_level_group'] = LE_Avg_Glucose_Level_Group.transform(data['avg_glucose_level_group'])
    data['bmi_group'] = LE_Bmi_Group.transform(data['bmi_group'])
    data['ever_married'] = data['ever_married'].map({'Yes': 1, 'No': 0})
    data['gender'] = LE_Gender.transform(data['gender'])
    data['Residence_type'] = LE_Residence_Type.transform(data['Residence_type'])

    data = scaler.transform(data)
    return data

# Define the input form
st.title('Stroke Detection GUI')
st.subheader('Enter the following patient information:')

#Numerical Inputs for age,hypertension,heart_disease,avg_glucose_level,bmi,stroke
Age = st.number_input('age', min_value=0.0, max_value=100.0, value=0.0)
Hypertension = st.number_input('Hypertension', min_value=0.0, max_value=100.0, value=0.0)
Heart_disease = st.number_input('heart_disease', min_value=0.0, max_value=100.0, value=0.0)
Avg_glucose_level = st.number_input('avg_glucose_level', min_value=0.0, max_value=100.0, value=0.0)
Bmi = st.number_input('bmi', min_value=0.0, max_value=100.0, value=0.0)

# Categorical Inputs for ever_married,work_type,Residence_type,smoking_status,age_group,bmi_group,avg_glucose_level_group
Gender = st.selectbox('gender', options=gender)
Ever_married = st.selectbox('ever_married', options=ever_married)
Work_type = st.selectbox('work_type', options=work_type)
residence_type = st.selectbox('Residence_type', options=Residence_type)
Smoking_status = st.selectbox('smoking_status', options=smoking_status)
Age_group = st.selectbox('age_group', options=age_group)
Bmi_group = st.selectbox('bmi_group', options=bmi_group)
Avg_glucose_level_group = st.selectbox('avg_glucose_level_group', options=avg_glucose_level_group)

#Dummies of WOrk Type
if Work_type == 'Govt_job':
    work_type_Govt_job = 1
    work_type_Never_worked = 0
    work_type_Private = 0
    work_type_Selfemployed = 0
    work_type_children = 0
elif Work_type == 'Never_worked':
    work_type_Govt_job = 0
    work_type_Never_worked = 1
    work_type_Private = 0
    work_type_Selfemployed = 0
    work_type_children = 0
elif Work_type == 'Private':
    work_type_Govt_job = 0
    work_type_Never_worked = 0
    work_type_Private = 1
    work_type_Selfemployed = 0
    work_type_children = 0
elif Work_type == 'Selfemployed':
    work_type_Govt_job = 0
    work_type_Never_worked = 0
    work_type_Private = 0
    work_type_Selfemployed = 1
    work_type_children = 0
elif Work_type == 'children':
    work_type_Govt_job = 0
    work_type_Never_worked = 0
    work_type_Private = 0
    work_type_Selfemployed = 0
    work_type_children = 1

#Dummies of smoking_status
if Smoking_status == 'formerly smoked':
    smoking_status_formerly_smoked = 1
    smoking_status_never_smoked = 0
    smoking_status_smokes = 0
elif Smoking_status == 'never smoked':
    smoking_status_formerly_smoked = 0
    smoking_status_never_smoked = 1
    smoking_status_smokes = 0
elif Smoking_status == 'smokes':
    smoking_status_formerly_smoked = 0
    smoking_status_never_smoked = 0
    smoking_status_smokes = 0

pred_click = st.button('Predict')
if pred_click:
    Dict = {'gender': Gender,
        'age': Age,
        'hypertension': Hypertension,
        'heart_disease': Heart_disease,
        'ever_married': Ever_married,
        'Residence_type': residence_type,
        'avg_glucose_level': Avg_glucose_level,
        'bmi': Bmi,
        'age_group': Age_group,
        'bmi_group': Bmi_group,
        'avg_glucose_level_group': Avg_glucose_level_group,
        'work_type_Govt_job': work_type_Govt_job,
        'work_type_Never_worked': work_type_Never_worked,
        'work_type_Private': work_type_Private,
        'work_type_Self-employed': work_type_Selfemployed,
        'work_type_children': work_type_children,
        'smoking_status_formerly smoked': smoking_status_formerly_smoked,
        'smoking_status_never smoked': smoking_status_never_smoked,
        'smoking_status_smokes': smoking_status_smokes}
    
    data = pd.DataFrame(Dict, index=[0])

    # Preprocess the input data
    df = preprocess_input(data)

    print(df)

    # Make the prediction
    prediction = model.predict_proba(df)[:, 1][0]

    # Display the prediction using openai custom prompt
    # Get your OpenAI API key.
    api_key = "sk-DMSLqyh0uU3vCBMzwljFT3BlbkFJc8pnhERvTEjaTYjom6"
    openai.api_key = api_key
    #Get response from OpenAI API
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=f"I have a patient who has a {prediction}% risk of stroke, using bulletpoints advise and give some basic recommendations to the patient.",
        n=1,
        max_tokens=500,
    )

    # Parse the response and extract the completion text.
    completion = response.choices[0]

    #Display the prediction
    st.write(f'The predicted probability of stroke is: {prediction:.2%}')
    st.write(f'Some advice for the patient: {completion.text}')
