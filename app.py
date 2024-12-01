'''
app.py
This file will the main runner for the the project. This runs the streamlit application for the MedAI symptom diagnosis system

This will start with importng all the necessary libraries to start within the a anaconda environment that runs 3.10.15. 
Next the tokenizer model and the dataset paths will be extracted for the best use of the data with symptomsand diagnosis.

Also the fairness threshold will also be set for the best possible model output.

Then the methods to input the symptoms from the user, run the symptom checker through the models and then ending with the feature analysis and model running in the end
'''
import os
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from main import (
    initialize_and_train_transformer,
    initialize_and_train_neural,
    evaluate_models,
    generate_explanations,
    register_user_main,
    login_user_main,
)
from DataLoader import load_multiple_files
from Transform import preprocess_text_with_stanza  # Import Stanza preprocessing
from Neural import PyTorchMulticlassNN
import numpy as np

# Initialize Transformer model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
transformer_model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=3)

# Predefined paths for datasets
SYMPTOM_DATASET_PATHS = [
    "C:/Users/agovi/OneDrive - The University of Colorado Denver/Documents/Fall 2024/CSCI 6970 Course Project/Data/Symptom Data/Training.csv",
    "C:/Users/agovi/OneDrive - The University of Colorado Denver/Documents/Fall 2024/CSCI 6970 Course Project/Data/Symptom Data/disease_data.csv",
    "C:/Users/agovi/OneDrive - The University of Colorado Denver/Documents/Fall 2024/CSCI 6970 Course Project/Data/Symptom Data/Disease_prediction_based_on_symptoms.csv",
    "C:/Users/agovi/OneDrive - The University of Colorado Denver/Documents/Fall 2024/CSCI 6970 Course Project/Data/Symptom Data/Diseases_Symptoms1.csv",
    "C:/Users/agovi/OneDrive - The University of Colorado Denver/Documents/Fall 2024/CSCI 6970 Course Project/Data/Symptom Data/FInal_Train_Data1.csv",
    "C:/Users/agovi/OneDrive - The University of Colorado Denver/Documents/Fall 2024/CSCI 6970 Course Project/Data/Symptom Data/infectious_diseases.csv",
    "C:/Users/agovi/OneDrive - The University of Colorado Denver/Documents/Fall 2024/CSCI 6970 Course Project/Data/Symptom Data/medical_data.csv",
    "C:/Users/agovi/OneDrive - The University of Colorado Denver/Documents/Fall 2024/CSCI 6970 Course Project/Data/Symptom Data/Testing.csv"
]

'''
XAI_DATASET_PATHS = [
    "C:/Users/agovi/OneDrive - The University of Colorado Denver/Documents/Fall 2024/CSCI 6970 Course Project/Data/Disease Data/alzheimers_disease_data.csv",
    "C:/Users/agovi/OneDrive - The University of Colorado Denver/Documents/Fall 2024/CSCI 6970 Course Project/Data/Disease Data/breast_cancer_wisconsin_data.csv",
    "C:/Users/agovi/OneDrive - The University of Colorado Denver/Documents/Fall 2024/CSCI 6970 Course Project/Data/Disease Data/cancer_patient_data_sets.csv",
    "C:/Users/agovi/OneDrive - The University of Colorado Denver/Documents/Fall 2024/CSCI 6970 Course Project/Data/Disease Data/cirrhosis.csv",
    "C:/Users/agovi/OneDrive - The University of Colorado Denver/Documents/Fall 2024/CSCI 6970 Course Project/Data/Disease Data/Dataset_Heart_Disease.csv",
    "C:/Users/agovi/OneDrive - The University of Colorado Denver/Documents/Fall 2024/CSCI 6970 Course Project/Data/Disease Data/diabetes.csv",
    "C:/Users/agovi/OneDrive - The University of Colorado Denver/Documents/Fall 2024/CSCI 6970 Course Project/Data/Disease Data/diabetes_data_upload.csv",
    "C:/Users/agovi/OneDrive - The University of Colorado Denver/Documents/Fall 2024/CSCI 6970 Course Project/Data/Disease Data/heart.csv",
    "C:/Users/agovi/OneDrive - The University of Colorado Denver/Documents/Fall 2024/CSCI 6970 Course Project/Data/Disease Data/heart_data.csv",
    "C:/Users/agovi/OneDrive - The University of Colorado Denver/Documents/Fall 2024/CSCI 6970 Course Project/Data/Disease Data/heart_failure_clinical_records_dataset.csv",
    "C:/Users/agovi/OneDrive - The University of Colorado Denver/Documents/Fall 2024/CSCI 6970 Course Project/Data/Disease Data/risk_factors_cervical_cancer.csv",
    "C:/Users/agovi/OneDrive - The University of Colorado Denver/Documents/Fall 2024/CSCI 6970 Course Project/Data/Disease Data/risk_factor_prediction_of_chronic_kidney_disease.csv",
]
 '''
# Predefined paths for saved models
PRETRAINED_TRANSFORMER_PATH = 'C:/Users/agovi/OneDrive - The University of Colorado Denver/Documents/Fall 2024/CSCI 6970 Course Project/Code/trained_models/transformer/transformer_model.pth'
PRETRAINED_NEURAL_PATH = 'C:/Users/agovi/OneDrive - The University of Colorado Denver/Documents/Fall 2024/CSCI 6970 Course Project/Code/trained_models/neural/neural_model.pth'

# Application state and constants
fairness_threshold = 0.1  # Threshold for fairness metric


# Preprocess user input using Stanza for real-time prediction with  User-entered symptom description.
# then return the preprocessed symptom description.
def preprocess_user_input_with_stanza(symptom_input):
    preprocessed_text = preprocess_text_with_stanza([symptom_input])
    return preprocessed_text[0]  # Extract the processed text

# with the transformer model, we will find the prediction along with the neural model
def make_prediction_with_transformers(symptoms_text):
    """Predict diagnosis using a transformer model with Stanza preprocessing."""
    inputs = tokenizer(symptoms_text, return_tensors="pt")
    outputs = transformer_model(**inputs)
    prediction = torch.argmax(outputs.logits, dim=1).item()
    return prediction, False  #fairness check

# this will be the mechanism for the training, prediction, and explanations
# this will set into motion the training and the final decsion in the end
#Displays the features for training, prediction, and explainability.
def show_model_features():
    st.subheader("Model Features: Training, Prediction, and Explanation")

    # Display dataset paths
    #If need to display the datasets
    #st.write("### Symptom Datasets:")
    for path in SYMPTOM_DATASET_PATHS:
        st.text(path)

    #st.write("### XAI Datasets:")
    #for path in XAI_DATASET_PATHS:
    #    st.text(path)

# this button will futher train the the transformer model and will run the model for the new patient
if st.button("Train Transformer Model"):
    with st.spinner("Training Transformer model..."):
        transformer_model = initialize_and_train_transformer(
            SYMPTOM_DATASET_PATHS, 
            num_classes=3, 
            epochs=5, 
            pretrained_model_path=PRETRAINED_TRANSFORMER_PATH
        )
        st.success("Transformer model trained and saved!")

# this button will futher train the the symptom neural network model and will run the model for the new patient
if st.button("Train Neural Network Model"):
    with st.spinner("Training Neural Network model..."):
        neural_model = initialize_and_train_neural(
            SYMPTOM_DATASET_PATHS, 
            input_size=128, 
            hidden_size1=64, 
            hidden_size2=32, 
            output_size=3, 
            epochs=5,
            pretrained_model_path=PRETRAINED_NEURAL_PATH
        )
        st.success("Neural Network model trained and saved!")

    # Evaluate models for the metrics
    if st.button("Evaluate Models"):
        with st.spinner("Evaluating models..."):
            evaluate_models(SYMPTOM_DATASET_PATHS)
            st.success("Model evaluation completed!")

    # this is the main area of the project which is the chatbot feature that will interact with the user
    # Input symptoms for prediction
    symptom_input = st.text_area("Enter symptoms or description of symptoms")
    if st.button("Predict Diagnosis"):
        if symptom_input:
            processed_input = preprocess_user_input_with_stanza(symptom_input)
            prediction, fairness_check = make_prediction_with_transformers(processed_input)
            st.write(f"Predicted Diagnosis: {prediction}")

            # This part will evaluate the fairness of the model to make sure the patient is getting the best diagnosis
            if fairness_check:
                st.error("Fairness criteria not met. Demographic parity difference exceeded.")
            else:
                st.success("Fairness criteria met.")
        else:
            st.warning("Please enter symptoms to proceed.")

    # The expalainable AI aspect with the why the prediction was made and which features played a bigger role
    st.subheader("Explain Predictions")
    sample_input = st.text_area("Enter sample input (comma-separated values):", "0.1, 0.2, 0.3")
    method = st.selectbox("Choose explanation method:", ["shap", "lime", "counterfactual"])

    # this button will allow the patient to take in the explanation at their discretion
    if st.button("Generate Explanation"):
        with st.spinner("Generating explanation..."):
            sample_input_array = np.array([float(x) for x in sample_input.split(",")]).reshape(1, -1)
            explanations = generate_explanations(
                PRETRAINED_NEURAL_PATH, 
                sample_input_array, 
                method=method
            )
            st.write(f"{method.upper()} Explanation:", explanations)

# This is the main method to run all the methods in each of the file
def main():
    st.title("MedAI Diagnosis System")
    st.image('MedAI_Logo.gif')  # Replace with the actual path to the logo file

    # Sidebar for user authentication
    st.sidebar.header("User Authentication")
    auth_option = st.sidebar.selectbox("Choose an option:", ["Register", "Login", "Continue as Guest"])

    if auth_option == "Register":
        st.subheader("Register New Account")
        username = st.text_input("Enter your username")
        password = st.text_input("Enter your password", type="password")
        contact_method = st.selectbox("Contact Method", ["email", "phone"])
        contact_value = st.text_input(f"Enter your {contact_method}")

        # Account credentials aspect
        if st.button("Register"):
            response = register_user_main(username, password, contact_method, contact_value)
            if response["success"]:
                st.success(response["message"])
            else:
                st.error(response["message"])

    elif auth_option == "Login":
        st.subheader("Login to Your Account")
        username = st.text_input("Enter your username")
        password = st.text_input("Enter your password", type="password")

        if st.button("Send MFA Code"):
            response = login_user_main(username, password)
            if response["success"]:
                st.success(response["message"])
                # Prompt for MFA code
                mfa_code = st.text_input("Enter the MFA code sent to your contact")
                if mfa_code:
                    if response["success"]:  # Assume MFA verification is done in `login_user_main`
                        st.success("MFA verified successfully. Access granted!")
                        show_model_features()
                    else:
                        st.error("Invalid MFA code.")
            else:
                st.error(response["message"])

    elif auth_option == "Continue as Guest":
        st.info("You are accessing the system as a guest.")
        show_model_features()

# runner
if __name__ == "__main__":
    main()