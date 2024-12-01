'''
Transform.py
This file will start with the Natural Language Processing part of this project will run the language model which can take
the symptoms from the patient input using the class 
The main method will be run through the Transformers library with the uncased Bert classfier with mutiple hidden layers for the optimal language model to help the patients.

The next methods will involved NLP with stanza of StanfordNLP with those evauations and metrics from the model.

The final part will process the input from the data to see the type of sentences that need to be formed
'''
import torch
from transformers import BertTokenizer, BertModel
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import stanza

# Initialize the tokenizer and BERT model
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Initialize the Stanza pipeline
stanza.download('en')  # Download the English model if not already downloaded
nlp = stanza.Pipeline(lang='en', processors='tokenize,lemma')

# Transformer-based classification model with a linear classifier head.
class TransformerSymptomClassifier(nn.Module):
# the initialization method will start with take the hidden layers in the Transformer method and
# also need the number of classes.
# based on lectures and knowledge, the hidden layer sizes will be 768
    def __init__(self, hidden_size=768, num_classes=None):
        super(TransformerSymptomClassifier, self).__init__()
        # error handling 
        if num_classes is None:
            raise ValueError("num_classes must be specified.")

        # specify the best model and then run the network
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

# forward pass of th transformer model
# the forward pass in a model in python requires the input ids and the attention mask to find the logits from the 
# classfier
    def forward(self, input_ids, attention_mask):
        # deliniate the outputs from the bert model
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # the token represenation from CLS
        cls_output = outputs.last_hidden_state[:, 0, :]
        return self.classifier(cls_output)

# Save the model to the specified path.
    def save_model(self, path):
        torch.save(self.state_dict(), path)
        print(f"Model saved to {path}")

    @classmethod
    # Load the model from the specified path wth the classes associated to load the loaded model.
    def load_model(cls, path, num_classes):
        # label the model
        model = cls(num_classes=num_classes)
        # load the dictionary
        model.load_state_dict(torch.load(path))
        print(f"Model loaded from {path}")
        return model

# Preprocess a list of texts using Stanza for lemmatization and BERT tokenizer for input IDs.
# we also need the maximu sequence length which will best in 128- bit lengths
def preprocess_text_with_stanza(text_list, max_len=128):
    # sart with the clean up for the stanza methods
    def stanza_clean(text):
        doc = nlp(text)
        lemmas = [word.lemma for sent in doc.sentences for word in sent.words]
        return " ".join(lemmas)
# output the cleaned text for misc characters
    cleaned_texts = [stanza_clean(text) for text in text_list]
# then run the encoding wth the inputids and the attention mask
    try:
        encoding = tokenizer.batch_encode_plus(
            cleaned_texts,
            max_length=max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        return encoding["input_ids"], encoding["attention_mask"]
# eeor handling
    except Exception as e:
        print(f"Error during preprocessing: {e}")
        raise

# Train the Transformer model.
# with the existing model, the dataloader, the optimzation of the model and the loss function to ensure the best model
# along with the optimal amount of epochs
def train_transformer_model(model, dataloader, optimizer, criterion, epochs=3):
    # train the model
    model.train()
    # track the loss through the epochs and make sure the loss stays low for the best model during epoch
    for epoch in range(epochs):
        total_loss = 0
        for input_ids, attention_mask, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss:.4f}")

#     Evaluate the Transformer model through taking the model and the dataloader with evaluation to obtain
# the accuracy, precision, recall, and F1-score
def evaluate_transformer_model(model, dataloader):
    # evalaute the model
    model.eval()
    # initialize the predictions 
    all_predictions = []
    # initialize the labels dictionary
    all_labels = []
    # as long as the network dosesnt have the gradient, then the cpu will work through the model to get the metrics
    with torch.no_grad():
        for input_ids, attention_mask, labels in dataloader:
            outputs = model(input_ids, attention_mask)
            predictions = torch.argmax(outputs, dim=1)
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
# the final metrics 
    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions, average="weighted")
    recall = recall_score(all_labels, all_predictions, average="weighted")
    f1 = f1_score(all_labels, all_predictions, average="weighted")
# printing of the final metrics
    print(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")
    return accuracy, precision, recall, f1


# Create a DataLoader from input features and labels with features, labels and batch size through the tensor library.
def create_dataloader(features, labels, batch_size=32):
    # set the datatset
    dataset = TensorDataset(features, labels)
    # obtain the dataloader variable
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader
