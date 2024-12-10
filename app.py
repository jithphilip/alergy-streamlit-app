import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
import streamlit as st




# Load the data
df = pd.read_csv("Updated.csv")

# Separate features and target variable
X = df[['Gender','Age', 'Residential Area','Smoking Exposure','Family History', 
        'Chest Tightness', 'Itching', 'Watery Eyes', 'Cough Severity','Sneezing Frequency']]
y = df['Allergy']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

# Preprocess categorical variables
categorical_features = ['Gender', 'Residential Area','Smoking Exposure','Family History']
categorical_transformer = OneHotEncoder(drop='first')

# Preprocess numerical variables
numerical_features = ['Age']
numerical_transformer = StandardScaler()

# Combine transformers
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Create a pipeline with the classifier
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=10, random_state=42))
])

# Fit the model
model.fit(X_train, y_train)

# Streamlit UI
st.title("Allergy Prediction App")

# User input layout in a grid with 2 rows and 5 columns
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    gender = st.radio("Gender", ['Male', 'Female'])

with col2:
    age = st.number_input("Age", min_value=int(df['Age'].min()), max_value=int(df['Age'].max()), value=int(df['Age'].median()))

with col3:
    residence_type = st.selectbox("Residential Area", df['Residential Area'].unique())

with col4:
    smoking_exposure = st.selectbox("Smoking Exposure", df['Smoking Exposure'].unique())

with col5:
    family_history = st.selectbox("Family History", df['Family History'].unique())

col6, col7, col8, col9, col10 = st.columns(5)

with col6:
    chest_tightness = st.selectbox("Chest Tightness", df['Chest Tightness'].unique())

with col7:
    itching = st.selectbox("Itching", df['Itching'].unique())

with col8:
    watery_eyes = st.selectbox("Watery Eyes", df['Watery Eyes'].unique())

with col9:
    cough_severity = st.selectbox("Cough Severity", df['Cough Severity'].unique())

with col10:
    sneezing_frequency = st.selectbox("Sneezing Frequency", df['Sneezing Frequency'].unique())



# Button to trigger prediction
if st.button("Predict"):
    # Convert user input to a DataFrame
    user_data = pd.DataFrame({
        'Gender': [gender],
        'Age': [age],
        'Residential Area': [residence_type],
        'Smoking Exposure':[smoking_exposure],
        'Family History': [family_history],
        'Chest Tightness': [chest_tightness],
        'Itching': [itching],
        'Watery Eyes': [watery_eyes],
        'Cough Severity': [cough_severity],
        'Sneezing Frequency': [sneezing_frequency]
    })

    # Make a prediction
    prediction = model.predict(user_data)

    # Display prediction
    st.subheader("Prediction Result:")
    if prediction[0] == 1:
        st.error("The model predicts allergy.")
    else:
        st.success("The model predicts no allergy.")
