'''
main.py

this file initialize the transformer model from the Transform.py and also train and keep the neural network and also use the pandas, PyTorch
libraries to find the metrics with the machine learning models.

Once the model part is finished, the register and login parts of the account capabilities, both with registering and login parts
This file will set the stage prior to the app.py file to make sure the models are evalauted and ran properly.
'''
import torch
import pandas as pd
from Neural import PyTorchMulticlassNN, train_pytorch_model, evaluate_pytorch_model
from Transform import TransformerSymptomClassifier, train_transformer_model, evaluate_transformer_model, preprocess_text_with_stanza
from DataLoader import load_multiple_files, preprocess_and_split_data
from account import register_user, login_user  # Import account functions

FAIRNESS_THRESHOLD = 0.1  # Set a threshold for demographic parity difference


#   Model Initialization and Training  

#     Initializes and trains a Transformer-based model with Stanza preprocessing.
# this function will take in the dataset paths, the output classes for classification, the training epochs and the batch size
# to yeidl the model of the transformer
def initialize_and_train_transformer(file_paths, num_classes, epochs=3, batch_size=32):
    # Load and preprocess data
    combined_data = load_multiple_files(file_paths, column_mappings={"Symptoms": "symptoms", "Disease": "label"})
    input_ids, attention_mask = preprocess_text_with_stanza(combined_data['symptoms'].tolist())
    labels = torch.tensor(combined_data['label'].tolist())
    train_loader, test_loader = preprocess_and_split_data(input_ids, attention_mask, labels, test_size=0.2)

    # Initialize the model
    model = TransformerSymptomClassifier(num_classes=num_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    criterion = torch.nn.CrossEntropyLoss()

    # Train the model
    print("Training the Transformer model...")
    train_transformer_model(model, train_loader, optimizer, criterion, epochs)

    # Save the model
    model.save_model("transformer_symptom_classifier.pth")
    print("Transformer model trained and saved.")
    return model

# Initializes and trains a PyTorch-based neural network model.
#This function will intake the datasets paths, input features, hidden layer neurons for both first and second, training epochs and the batch size 
# to output the trained PyTorch model for the symptom prediction
def initialize_and_train_neural(file_paths, input_size, hidden_size1, hidden_size2, output_size, epochs=3, batch_size=32):
    
    # Load and preprocess data
    combined_data = load_multiple_files(file_paths, column_mappings={"Symptoms": "symptoms", "Disease": "label"})
    input_ids, attention_mask = preprocess_text_with_stanza(combined_data['symptoms'].tolist())
    features = torch.cat([input_ids, attention_mask], dim=1)  # Combine input_ids and attention_mask
    labels = torch.tensor(combined_data['label'].tolist())
    train_loader, test_loader = preprocess_and_split_data(features, labels, test_size=0.2)

    # Initialize the model
    model = PyTorchMulticlassNN(input_size=input_size, hidden_size1=hidden_size1,
                                 hidden_size2=hidden_size2, output_size=output_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()

    # Train the model
    print("Training the neural network...")
    train_pytorch_model(model, train_loader, criterion, optimizer, epochs)

    # Save the model
    torch.save(model.state_dict(), "neural_symptom_classifier.pth")
    print("Neural network model trained and saved.")
    return model


#   Model Evaluation  
# Evaluates both Transformer-based and PyTorch-based neural network models.
# with the help of the dataset paths, the pretrained models as well
def evaluate_models(file_paths, transformer_model=None, neural_model=None):
    # extract the datasets
    combined_data = load_multiple_files(file_paths, column_mappings={"Symptoms": "symptoms", "Disease": "label"})
    # intake the mask and the ids
    input_ids, attention_mask = preprocess_text_with_stanza(combined_data['symptoms'].tolist())
    # get the test loader and the labels to together
    labels = torch.tensor(combined_data['label'].tolist())
    _, test_loader = preprocess_and_split_data(input_ids, attention_mask, labels, test_size=0.2)

    # evaluate
    if transformer_model:
        print("Evaluating Transformer model...")
        evaluate_transformer_model(transformer_model, test_loader)

    # evaluate
    if neural_model:
        print("Evaluating Neural Network model...")
        evaluate_pytorch_model(neural_model, test_loader)


#   User Authentication Wrappers  

# with the account.py functionality through the registration factor
# using the user inputting username, password, the method of contact to check the one-time password
# also to intake the value inputting the one-time password
def register_user_main(username, password, contact_method, contact_value):
    try:
        register_user(username, password, contact_method, contact_value)
        return {"success": True, "message": f"User {username} registered successfully."}
    except Exception as e:
        return {"success": False, "message": str(e)}

# with the account.py functionality through the login factor
# using the user inputting username, password, to chack if the patient is the holder of the account requested
def login_user_main(username, password):
    try:
        success = login_user(username, password)
        if success:
            return {"success": True, "message": f"One-Time code sent to {username}'s registered contact."}
        else:
            return {"success": False, "message": "Login failed. Check credentials."}
    except Exception as e:
        return {"success": False, "message": str(e)}