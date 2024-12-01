'''
Neural.py

this file initialize the tNeural model through PyTorch with the capability of training and testing, forwarding
libraries to find the metrics with the machine learning models.

This will be the center of the generative AI model that will output the solution to the patient.
'''
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch.utils.data import DataLoader, TensorDataset

# A Multilayer Perceptron (MLP) for multiclass classification using PyTorch.
class PyTorchMulticlassNN(nn.Module):
    # initialize the method for the neural network through the hidden layers' sizes
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        super(PyTorchMulticlassNN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size1),
            nn.ReLU(),
            nn.Linear(hidden_size1, hidden_size2),
            nn.ReLU(),
            nn.Linear(hidden_size2, output_size)
        )

# this method will provide the forward pass throgh the network with the input tensor
    def forward(self, x):
        return self.network(x)

# training of the neural network with the dataloader and the loss funtion for the optimal loss and optimization and 
# the amount of training epochs specified in the app.py
def train_pytorch_model(model, dataloader, criterion, optimizer, epochs=10):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for inputs, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss:.4f}")

# the evaluation of the model with the neural network model and the dataloader for the test data to return the metrics
def evaluate_pytorch_model(model, dataloader):
    model.eval()
    all_predictions = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in dataloader:
            outputs = model(inputs)
            predictions = torch.argmax(outputs, dim=1)
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
# this will be the evaluation process to make sure of the optimal network
    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions, average="weighted")
    recall = recall_score(all_labels, all_predictions, average="weighted")
    f1 = f1_score(all_labels, all_predictions, average="weighted")
# print out the metrics
    print(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")
    return accuracy, precision, recall, f1

# this is will integrate the features and labels with the features and the batch size.
def create_dataloader(features, labels, batch_size=32):
    dataset = TensorDataset(features, labels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader