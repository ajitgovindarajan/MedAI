'''
omniExpAI.py
 This file will be the explanable AI hub of the project.

 With the assiatnce of omniXAI in the python 3.10.15 version, it will intake the models of the symptom classifiers 
 and take the use of the tabular classfier method to explain the data and the decision by the model to the user. Start with
 setting the scaler, uploading the model, setting the tabular explainers, initializing the explainaers for the different techniques.

 Then when the methods are initialized, the explaining will happen with the methods that is requested from the app.py 
 and then the integration will occur with app.py with the help of the integration method
'''
import pandas as pd
import torch
from omnixai.explainers.tabular import TabularExplainer
#from omnixai.explainers.text import TextExplainer
import numpy as np
#import tensorflow as tf
import matplotlib.pyplot as plt
from omnixai.data.image import Image
from omnixai.explainers.vision import CounterfactualExplainer
from sklearn.preprocessing import MinMaxScaler
from Neural import PyTorchMulticlassNN
from DataLoader import preprocess_and_split_data
# Import Stanza preprocessing
from Transform import preprocess_text_with_stanza

# Initialize scalers for consistent feature scaling
scaler = MinMaxScaler()

#   Model Initialization  

#  Initializes and loads the PyTorch neural network model for explanation purposes.
# this method yield the loaded model that is uploaded
# the method will require input features, sizes of the hidden sizes, the output classes and the saved model weights
def initialize_neural_model(input_size, hidden_size1, hidden_size2, output_size, model_path):
    # the extracted model to classify
    model = PyTorchMulticlassNN(input_size, hidden_size1, hidden_size2, output_size)
    #load the model
    model.load_state_dict(torch.load(model_path))
    # run the evaluator
    model.eval()
    return model


#   Explainability Frameworks  

# Initializes SHAP, LIME, and counterfactual explainers for tabular data.
# Input Arguments:The trained PyTorch model.
#Tabular data used for explainability in a scaled frame
#TabularExplainer: An initialized OmniXAI Tabular Explainer instance.
 
def initialize_tabular_explainers(model, data):
    explainer = TabularExplainer(
        model=model,
        data=data,
        mode="classification",
        feature_columns=[f"Feature {i}" for i in range(data.shape[1])],
    )
    return explainer


#Initializes the Counterfactual Explainer for tabular data.
# need to intake the trained Neural network and the data frame in the counterfactual generation
# return the counterfactual explanation
def initialize_counterfactual_explainer(model, data, target_column=None):
    explainer = CounterfactualExplainer(
        model=model,
        data=data,
        target_column=target_column,
        mode="classification",
    )
    return explainer


#   Explanation Functions  
# Explains the predictions using SHAP and LIME with Stanza preprocessing with outputting the LIME and SHAP emplanations
# take in the trained Neural network model, the dataset and the sample input 
def explain_with_shap_lime(model, dataset_path, sample_input, batch_size=32):
    # Load and preprocess dataset
    df = pd.read_csv(dataset_path)
    input_ids, attention_mask = preprocess_text_with_stanza(df["symptoms"].tolist())
    labels = torch.tensor(df["label"].tolist())
    train_loader, _ = preprocess_and_split_data(input_ids, attention_mask, labels)
    # run the smapler for the model the get the best results
    data_sample = next(iter(train_loader))[0]  # Use training data for explainers
    scaled_data = scaler.fit_transform(data_sample.numpy())

    # Initialize explainers
    explainer = initialize_tabular_explainers(model, scaled_data)
    explanations = explainer.explain(sample_input)
    return explanations

#Explains predictions using counterfactual examples with Stanza preprocessing
# with the help of the symptom neural network, the preprocessing dataset, a sample input, and the batch size
def explain_with_counterfactual(model, dataset_path, sample_input, batch_size=32):
    # Load and preprocess dataset
    df = pd.read_csv(dataset_path)
    input_ids, attention_mask = preprocess_text_with_stanza(df["symptoms"].tolist())
    labels = torch.tensor(df["label"].tolist())
    train_loader, _ = preprocess_and_split_data(input_ids, attention_mask, labels)
# get the data sampler for the explainers
    data_sample = next(iter(train_loader))[0]  # Use training data for explainers
    scaled_data = scaler.fit_transform(data_sample.numpy())
    # Initialize counterfactual explainer
    explainer = initialize_counterfactual_explainer(model, scaled_data)
    explanations = explainer.explain(sample_input)
    return explanations


#   Integration with App  
#Unified interface for generating explanations using various techniques with Stanza preprocessing.
# the neural network path, datast path, sample input and the XA technique
def explain_model_integration(model_path, dataset_path, sample_input, method="shap"):
    # Load the trained neural network model
    input_size, hidden_size1, hidden_size2, output_size = 128, 64, 32, 3  # Example sizes
    model = initialize_neural_model(input_size, hidden_size1, hidden_size2, output_size, model_path)

    # Generate explanations based on the method
    if method == "shap":
        return explain_with_shap_lime(model, dataset_path, sample_input)
    elif method == "counterfactual":
        return explain_with_counterfactual(model, dataset_path, sample_input)
    else:
        raise ValueError(f"Unsupported explanation method: {method}")