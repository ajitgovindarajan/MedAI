# MedAI
Medical symptom diagnosis system for final MS project and NLP Coure


\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*

* Name      : Ajit Govindarajan
* Student ID: 107861904
* Class     :  CSCI 5800-H02: Natural Language Processing and GenAI & MS Course Project CSCI 6970

\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*


Read Me


\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*

* Description of the program

\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*

1. Generate 100,000 random integers between 0 - 1,000,000.

Then save them to a text file where each line has 5 numbers per line.

1. Read the numbers back into a plain old array of integers.
1. Use insertion sort to sort the array
1. Asks the user to enter a number between 0 - 1,000,000.

The program uses the binary search algorithm to determine

if the specified number is in the array or not.  It also

displays the search step in details

1. Maintain a loop asking if the user wants to play again or not

after a search successfully completes.  Thetest set includes

the following integer numbers.

{-100, 0, 123456, 777777, 459845, 1000000, 1000001}


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

Name: MedAI_Cross_Enhanced_Logo_Code.py

This file is a utility file that create the moving logo picture for the streamlit application and the and final reports as well as the presentation later. With the help of the matplotlib capabilities allows for a logo that gives some creatvity to the MedAI system. Also this file can give some insights into how certain results could be presented through pictures and not all through text.

Name: Logo.py

This file is a utility file that create the static logo picture for the streamlit application and the and final reports as well
as the presentation later.
With the help of the matplotlib capabilities allows for a logo that gives some creatvity to the MedAI system.

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

% cd [MedAI]

Make sure to change the paths of the datasets and the trained_models Change to the directory that contains the datafiles in app.py as well as the saved models and the Firebase

Compile the program by:

Then you will need the anaconda environment to run the code also known as MedAI.yaml

In the anaconda base terminal type: conda activate MedAI
which will make the environment to run the program

Also to save an enviromnent: activate the environment and the type : conda env export > environmentname.yml
and it will save 

Open a terminal with that environment

1. Run the program by:

heading to the file directory with all of the program files with cd command

the type 'Streamlit run app.py'
