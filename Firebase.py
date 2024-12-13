"""
# Firebase_.py
This file allows to the leveraging of the firebase secure database that is utilized for many important applications to keep details
and account credentials safe.

This will give high level protection to the system that any users will add the peace of mind

"""
import os
from firebase_admin import auth, credentials, initialize_app


firebase_initialized = False
# filepath that contains the.json file of the details
file_path ="C:/Users/agovi/OneDrive - The University of Colorado Denver/Documents/Fall 2024/CSCI 6970 Course Project/Code/Firebase/sample1.json"
'''
# Initialize Firebase app with your service account key JSON file
cred = credentials.Certificate(file_path)
firebase_admin.initialize_app(cred)
'''

# initialize the Firebase database. If the file is missing, then the notice
# will be provided to the user
def initialize_firebase():
    global firebase_initialized
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Service account JSON file not found at {file_path}.")
        
        cred = credentials.Certificate(file_path)
        initialize_app(cred)
        firebase_initialized = True
        print("Firebase initialized successfully!")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please generate a service account key from the Firebase Console:")
        print("1. Go to your Firebase project settings.")
        print("2. Navigate to 'Service Accounts'.")
        print("3. Click 'Generate New Private Key' and download the JSON file.")
        print(f"4. Place the file in the directory: {os.path.dirname(file_path)}")
    except Exception as e:
        print(f"Unexpected error during Firebase initialization: {e}")

# User Management Functions
# this is whre we make and store the details of a new account
# user
def register_user_in_firebase(email=None, phone=None, password=None):
    if not firebase_initialized:
        print("Firebase not initialized. Cannot register user.")
        return False
    try:
        user = auth.create_user(
            email=email,
            phone_number=phone,
            password=password,
        )
        print(f"Firebase: User {user.uid} created successfully.")
        return True
    except Exception as e:
        print(f"Firebase registration error: {e}")
        return False

# add the MFA side with the verification
def authenticate_user_in_firebase(email):
    if not firebase_initialized:
        print("Firebase not initialized. Cannot authenticate user.")
        return None
    try:
        user = auth.get_user_by_email(email)
        print(f"Firebase: User {user.uid} authenticated successfully.")
        return user
    except Exception as e:
        print(f"Firebase authentication error: {e}")
        return None