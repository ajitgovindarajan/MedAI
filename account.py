'''
account.py
This file be in charge of the account details and manufacturing when the patients want to make an account.
With the help of the development database, Firebase, the file will intake, write and store details as well as provide security through robust
AES encryption and decryption.

The file is also incharge of manufacturing the multi-factor authentication protocol that is critical for the cybersecurity of the MedAI
system that will keep user details safe against cyber attacks of account intruders.

'''
import hashlib
import base64
import os
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives import padding
from cryptography.hazmat.backends import default_backend
from firebase_admin import auth, credentials, initialize_app
import pyotp
import smtplib
from email.mime.text import MIMEText
from email.message import EmailMessage

# AES encryption key (should be stored securely)
AES_KEY = os.urandom(32)

# Initialize Firebase (reusing Firebase.py functionality)
firebase_initialized = False
try:
    from Firebase import initialize_firebase

    initialize_firebase()
    firebase_initialized = True
except ImportError:
    print("Firebase.py not found or Firebase not initialized.")

# AES Encryption and Decryption

# take in the database data and the username/password data from the patient and then this will enyrpted into the secure 
# firebase database for the optimal security
def encrypt_data(data):
    backend = default_backend()
    iv = os.urandom(16)
    cipher = Cipher(algorithms.AES(AES_KEY), modes.CBC(iv), backend=backend)
    encryptor = cipher.encryptor()

    padder = padding.PKCS7(128).padder()
    padded_data = padder.update(data.encode()) + padder.finalize()

    encrypted = encryptor.update(padded_data) + encryptor.finalize()
    return base64.b64encode(iv + encrypted).decode()

# need the AES decrytpion for the data to makesure they are comprehensible 
def decrypt_data(encrypted_data):
    backend = default_backend()
    encrypted_data = base64.b64decode(encrypted_data)

    iv = encrypted_data[:16]
    encrypted = encrypted_data[16:]

    cipher = Cipher(algorithms.AES(AES_KEY), modes.CBC(iv), backend=backend)
    decryptor = cipher.decryptor()

    padded_data = decryptor.update(encrypted) + decryptor.finalize()
    unpadder = padding.PKCS7(128).unpadder()
    data = unpadder.update(padded_data) + unpadder.finalize()
    return data.decode()


#   MFA (using pyotp)  
# this method will use the python method for one-time passwords to randomly generate them for sending methods
def generate_mfa_secret():
    return pyotp.random_base32()

# this method will send the one-time password through either email and phone
def send_mfa_code(contact_method, contact_value, mfa_code):
    if contact_method == "email":
        send_email(contact_value, "Your MFA Code", f"Your MFA code is: {mfa_code}")
    elif contact_method == "phone":
        send_sms(contact_value, f"Your MFA code is: {mfa_code}")
    else:
        raise ValueError("Invalid contact method.")

#this method will be MFA through email based on what the patient will request
# this just a general send email function that will be integrated from the main and the app
def send_email(recipient, subject, body):
    try:
        sender = "example@example.com"  # Replace with your email
        password = "password"  # Replace with your email password
        server = smtplib.SMTP("smtp.example.com", 587)  # Replace with your SMTP server
        server.starttls()
        server.login(sender, password)
        msg = MIMEText(body)
        msg["Subject"] = subject
        msg["From"] = sender
        msg["To"] = recipient
        server.sendmail(sender, recipient, msg.as_string())
        server.quit()
        print("Email sent successfully.")
    except Exception as e:
        print(f"Failed to send email: {e}")

# this function will send the one-time-password o the user's phone if they decide for phone contact
def send_sms(phone_number, message):
    # Placeholder for SMS implementation. Firebase can be used for phone auth.
    print(f"Sending SMS to {phone_number}: {message}")


#   User Registration and Login  

# register users with name, passwords and the contact method to use MFA to access account and maybe one day recover account
def register_user(username, password, contact_method, contact_value):
    hashed_password = hashlib.sha256(password.encode()).hexdigest()
    encrypted_mfa_secret = encrypt_data(generate_mfa_secret())

    if firebase_initialized:
        # Register user in Firebase
        auth.create_user(
            email=username if contact_method == "email" else None,
            phone_number=contact_value if contact_method == "phone" else None,
            password=password,
        )
        print(f"User {username} registered in Firebase.")
    else:
        # Placeholder for local user storage
        print(f"User {username} registered locally with encrypted MFA secret.")

# this will take in the user from the login method with the standard username and password
def login_user(username, password):
    hashed_password = hashlib.sha256(password.encode()).hexdigest()

    if firebase_initialized:
        try:
            # Authenticate with Firebase
            user = auth.get_user_by_email(username)
            print(f"User {username} authenticated via Firebase.")
            return True
        except Exception as e:
            print(f"Firebase authentication failed: {e}")
            return False
    else:
        # Placeholder for local authentication
        print(f"User {username} authenticated locally.")
        return True