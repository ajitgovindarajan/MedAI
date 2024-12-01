"""
# Firebase_.py
This file allows to the leveraging of the firebase secure database that is utilized for many important applications to keep details
and account credentials safe.

This will give high level protection to the system that any users will add the peace of mind

"""
import firebase_admin
from firebase_admin import credentials, auth

# filepath that contains the.json file of the details
file_path ="C:/Users/agovi/OneDrive - The University of Colorado Denver/Documents/Fall 2024/CSCI 6970 Course Project/Code/Firebase"

# Initialize Firebase app with your service account key JSON file
cred = credentials.Certificate(file_path)
firebase_admin.initialize_app(cred)