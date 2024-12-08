<img src="[https://user-images.githubusercontent.com/20041231/211718743-d6604ff7-8828-422b-9b60-ec156cdaf054.JPG](https://github.com/the-princekrrazz/SMS-spam-classifier/blob/main/Demo.png)"></img>
This is a Spam Detection App built using Logistic Regression and the NLTK (Natural Language Toolkit) library. The app classifies text messages as either spam or ham (non-spam) based on a pre-trained model.

Features
Spam Detection: Classifies messages as spam or ham using machine learning.
Logistic Regression: A simple yet effective machine learning model used for classification.
NLTK for Text Preprocessing: Utilizes the NLTK library for text cleaning and feature extraction.

Technologies Used
Python: Programming language.
NLTK: Natural Language Processing toolkit for text preprocessing.
Scikit-learn: For building and evaluating the Logistic Regression model.
Pandas: For data handling and manipulation.


Installation
Follow the steps below to set up the Spam Detection App on your local machine.

Step 1: Clone the Repository
bash
Copy code
git clone https://github.com/the_princekrrazz/Email-Spam-classifier.git
cd spam-detection-app
Step 2: Install Dependencies
Make sure you have Python 3.x installed on your system. Then, install the required libraries using pip:

bash
Copy code
pip install -r requirements.txt
requirements.txt includes the following dependencies:

nltk
scikit-learn
pandas
numpy
Step 3: Download NLTK Resources
This app uses NLTK resources for text preprocessing. Run the following script to download the necessary NLTK data:

python
Copy code
import nltk
nltk.download('stopwords')
nltk.download('punkt')
How to Use
Train the Model: The first step is to train the logistic regression model on a dataset of labeled messages (spam and ham).
Classify Messages: Once the model is trained, you can input new messages to predict whether they are spam or ham.
Training the Model
To train the model, use the train_model.py script. The training data must contain labeled messages where the label is either "spam" or "ham". A sample dataset is included in the repository.

bash
Copy code
python train_model.py
This script will:

Load the dataset.
Preprocess the text data (tokenization, removing stopwords, etc.).
Train the Logistic Regression model.
Save the trained model to a file (spam_classifier.pkl).
Classifying New Messages
To classify new messages, use the predict.py script. You can pass messages directly via the command line or modify the script for other types of input.

bash
Copy code
python predict.py "Congratulations! You've won a prize, call us now."
This will output whether the message is "spam" or "ham".

Example Workflow
Train the model with labeled spam/ham messages.
Classify a message:
Input: "Congrats, you've won a free ticket!"

Output: spam

Input: "Hey, how about meeting for lunch tomorrow?"

Output: ham
