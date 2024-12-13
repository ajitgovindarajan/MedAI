'''
DataLoader.py
This file will be the way that data is extracted from the datasets whether they are XAI datasets and symptom datasets
 The files will be combines from the path and will split into 20-80 split with the seeding of the random state
'''
import matplotlib.pyplot as plt
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from Transform import preprocess_text_with_stanza
from torch.utils.data import DataLoader, TensorDataset

# this method will help to train the neural network with the mulitple files through the paths in 
def load_multiple_files(file_paths, column_mappings=None):
    dfs = []
    for path in file_paths:
        try:
            df = pd.read_csv(path)
            if column_mappings:
                df.rename(columns=column_mappings, inplace=True)
            dfs.append(df)
        except Exception as e:
            print(f"Error reading {path}: {e}")
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


# Categorize datasets into symptom and EXAI datasets based on file naming or content.
# with the list of datasets from app.py and the identifier on the datasets output a dictionary of data
# one dictionary with symptoms and XAI data
def load_and_categorize_files(file_paths, exai_keyword="EXAI"):
    # initialize the data dictionaries
    symptom_data = []
    exai_data = []
# loop the loading of the those datasets
    for file_path in file_paths:
        try:
            df = pd.read_csv(file_path)

            # Check if file is EXAI-related based on naming or columns
            if exai_keyword in file_path or "shap" in map(str.lower, df.columns):
                exai_data.append(df)
            else:
                symptom_data.append(df)
            # need to have eror handling
        except Exception as e:
            print(f"Error loading {file_path}: {e}")

    return {
        "symptom_data": pd.concat(symptom_data, ignore_index=True) if symptom_data else None,
        "exai_data": pd.concat(exai_data, ignore_index=True) if exai_data else None,
    }

# Preprocess EXAI datasets for analysis and evaluation.
# this method will take the dataframe of the XAi datasets and use the explanation technique to rorganize the data
def preprocess_exai_data(exai_data):
    try:
        shap_data = exai_data[exai_data["explanation_type"] == "SHAP"]
        lime_data = exai_data[exai_data["explanation_type"] == "LIME"]
        counterfactual_data = exai_data[exai_data["explanation_type"] == "Counterfactual"]

        return {
            "shap": shap_data,
            "lime": lime_data,
            "counterfactual": counterfactual_data,
        }
    except Exception as e:
        print(f"Error preprocessing EXAI data: {e}")
        raise

# Preprocess and split the data into training and testing sets.
# with the new dataframe, the 80-20 train-test split, the random seeding allows for the dataloaders of the training 
# and testing datasets
# Preprocess and split the data into training and testing sets.
def preprocess_and_split_data(df, test_size=0.2, random_state=42, disease_filter=None):
    try:
        if disease_filter:
            df = df[df['disease_category'] == disease_filter]

        # Ensure required columns are present
        if 'symptoms' not in df.columns or 'label' not in df.columns:
            raise ValueError("The DataFrame must contain 'symptoms' and 'label' columns.")

        # Extract symptoms and labels
        texts = df['symptoms'].tolist()
        labels = df['label'].tolist()

        # Preprocess symptoms using the BERT tokenizer
        input_ids, attention_mask = preprocess_text_with_stanza(texts)

        # Convert labels to tensor
        labels_tensor = torch.tensor(labels)

        # Split into training and testing datasets
        train_ids, test_ids, train_mask, test_mask, train_labels, test_labels = train_test_split(
            input_ids, attention_mask, labels_tensor, test_size=test_size, random_state=random_state
        )

        # Create DataLoaders
        train_dataloader = create_dataloader(train_ids, train_mask, train_labels)
        test_dataloader = create_dataloader(test_ids, test_mask, test_labels)
        return train_dataloader, test_dataloader
    except Exception as e:
        print(f"Error preprocessing and splitting data: {e}")
        raise

# this method will look a the corpus data in the tokenization through distributions, vocabulary size
def evaluate_corpus_data(df, text_column="conversation"):
    try:
        if text_column not in df.columns:
            raise ValueError(f"The DataFrame must contain a '{text_column}' column for corpus evaluation.")

        # Tokenize the text data using whitespace
        all_tokens = [token for text in df[text_column].dropna() for token in text.split()]
        token_counts = Counter(all_tokens)
        vocab_size = len(token_counts)
        token_distribution = list(token_counts.values())

        # Corpus statistics
        print(f"Vocabulary Size: {vocab_size}")
        print(f"Total Tokens: {sum(token_distribution)}")
        print(f"Top 10 Most Frequent Tokens: {token_counts.most_common(10)}")

        # Visualize token distribution
        plt.figure(figsize=(10, 6))
        plt.hist(token_distribution, bins=50, color="blue", alpha=0.7)
        plt.title("Token Distribution in Corpus")
        plt.xlabel("Frequency")
        plt.ylabel("Count")
        plt.show()

        return {
            "vocab_size": vocab_size,
            "total_tokens": sum(token_distribution),
            "most_frequent_tokens": token_counts.most_common(10),
            "token_distribution": token_distribution,
        }
    except Exception as e:
        print(f"Error evaluating corpus data: {e}")
        raise


# Creates a PyTorch DataLoader for batching input IDs, attention masks, and labels.
def create_dataloader(input_ids, attention_mask, labels, batch_size=32):
    # with tensor and dataLoader we can obtain the loader
    try:
        dataset = TensorDataset(input_ids, attention_mask, labels)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        return dataloader
    # exception as the error
    except Exception as e:
        print(f"Error creating DataLoader: {e}")
        raise
