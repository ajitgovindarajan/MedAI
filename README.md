\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*

* Name      : Ajit Govindarajan
* Student ID: 107861904
* Class     :  CSCI 5800-H02: Natural Language Processing and GenAI & MS Course Project CSCI 6970

\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*


Read Me


\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*

* Description of the program

\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*

1. This program will will generate a disease or ailment diagnosis based on the user or patient input 


\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*

* Source files

\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*

Name:  main.py

Main program. This file initialize the transformer model from the Transform.py and also train and keep the neural network and also use the pandas, PyTorch libraries to find the metrics with the machine learning models.
Once the model part is finished, the register and login parts of the account capabilities, both with registering and login parts.This file will set the stage prior to the app.py file to make sure the models are evalauted and ran properly.

Name:  app.py

This file will the main runner file for the the project. This runs the streamlit application for the MedAI symptom diagnosis system This will start with importng all the necessary libraries to start within the a anaconda environment that runs 3.10.15 and the libraries in requirements.txt. 
Next the tokenizer model and the dataset paths will be extracted for the best use of the data with symptomsand diagnosis. Also the fairness threshold will also be set for the best possible model output.
Then the methods to input the symptoms from the user, run the symptom checker through the models and then ending with the feature analysis and model running in the end

Name: account.py

This file be in charge of the account details and manufacturing when the patients want to make an account.
With the help of the development database, Firebase, the file will intake, write and store details as well as provide security through robust
AES encryption and decryption. The file is also incharge of manufacturing the multi-factor authentication protocol that is critical for the cybersecurity of the MedAI system that will keep user details safe against cyber attacks of account intruders.

Name: DataLoader.py

This file will be the way that data is extracted from the datasets whether they are XAI datasets and symptom datasets.The files will be combines from the path and will split into 20-80 split with the seeding of the random state. This file will be heavily based in the PyTroch library with Tensor dataset and pandas as well to ensure the best data preparation for the models that need to trained

Name: Firebase.py

This file allows to the leveraging of the firebase secure database that is utilized for many important applications to keep details and account credentials safe.This will give high level protection to the system that any users will add the peace of mind. This will be the utility file to ensure security in MedAI.

Name: Neural.py

This file initialize and create the Neural network model through PyTorch and partner with the DataLoader file to ensure the best model with the capability of training and testing, forwarding libraries to find the metrics with the machine learning models. This will be the center of the generative AI model that will output the solution to the patient.

Name: omniExpAI.py

This file will be the explanable AI hub of the project. With the assiatnce of omniXAI in the python 3.10.15 version, it will intake the models of the symptom classifiers and take the use of the tabular classfier method to explain the data and the decision by the model to the user. Start with setting the scaler, uploading the model, setting the tabular explainers, initializing the explainaers for the different techniques.
Then when the methods are initialized, the explaining will happen with the methods that is requested from the app.py and then the integration will occur with app.py with the button to train and provide the patient with an explanation at request.

Name: Transform.py

This file will start with the Natural Language Processing part of this project will run the language model which can take the symptoms from the patient input using the class. The main method will be run through the Transformers library with the uncased Bert classfier with mutiple hidden layers for the optimal language model to help the patients. The next methods will involved NLP with stanza of StanfordNLP with those evauations and metrics from the model.The final part will process the input from the data to see the type of sentences that need to be formed.

Name: MedAIEnhanced_Logo_Code.py

This file is a utility file that create the moving logo picture for the streamlit application and the and final reports as well as the presentation later. With the help of the matplotlib capabilities allows for a logo that gives some creatvity to the MedAI system. Also this file can give some insights into how certain results could be presented through pictures and not all through text.

Name: Logo.py

This file is a utility file that create the static logo picture for the streamlit application and the and final reports as well
as the presentation later.
With the help of the matplotlib capabilities allows for a logo that gives some creatvity to the MedAI system.

Name: MedAI.yml

This file is the virtual conda enviroment that will best run the system. The running directions is listed below as the environment provides the tailored environment compared to going through the requirements.txt

Name: requirements.txt

This file contains the python version and libraries that are requires to run this program

Name: MedAI_Logo

This file is the main logo to MedAI made from the cross-enhanced file that allows for the glowing logo on the dark webpage

Name: MedAI_Creative_AI_Neural_Network_Logo

This file is the moving logo for the system in the app page which will provide some illustration

Name: MedAI_Enhanced_Logo_Static

This file is the static logo for the system in the app page which will provide some illustration

\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*

* Folders that need to be made

\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*

Firebase

This folder will have the account credentials as the database of those as well

trained_models

This folder will hav both the trained models of the transformer models and the neural network in the form of .pth files.

Also with in this folder, folders named neural and transformer need to be produced and fixed in path variables


\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*

* Circumstances of programs

\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*

The program runs successfully.

The program was developed and tested on The program was developed and tested Anaconda Python developer with Python 3.10.15. It was compiled, run, and tested on Anaconda python environment


\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*

* How to build and run the program

\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*

1. Uncompress the Project file.  The file is compressed.

To uncompress it use the following commands

% unzip [MedAI]

Now you should see a directory with the files:

account.py

app.py

DataLoader.py

Firebase.py

Logo.py

main.py

MedAI_Cross_Enhanced_Logo_Code.py

Neural.py

omniExpAI.py

Transform

Readme.md

Firebase folder

trained_models folder

requirements.txt

Data folder

MedAI.yml

1. Build the program.

Download the program files to your local machine

Make sure to change the paths of the datasets and the trained_models Change to the directory that contains the datafiles in app.py as well as the saved models and the Firebase

Compile the program by:

Then you will need the anaconda environment to run the code also known as MedAI.yaml

In the anaconda base terminal type: conda activate MedAI
which will make the environment to run the program

Also to save an enviromnent: activate the environment and the type : conda env export > environmentname.yml
and it will save 

Open a terminal with that environment

1. Run the program by:

heading to the file directory with all of the program files with cd command: cd /workspace/MedAI

# replace workspace with the alloted path on your machine

then type 'Streamlit run app.py'
