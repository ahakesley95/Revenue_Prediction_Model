## Project Title
KV6002 Team Project and Professionalism - 'Revenue Prediction model' individual subsystem

## Project Description
This 'model' directory has been developed as an individual subsystem for KV6002 Team Project 
and Professionalism at Northumbria University.

## Setting up
For the purposes of this project, the Flask server is run locally (127.0.0.1:5000). In order 
to use the file, a number of packages are required and can be added via pip or Anaconda using 
the '*_requirements.txt' files. Once the packages are installed, run 'app.py'.
This will then start the Flask app in a development server. 

## Use of the prediction API
The API has three endpoints: authenticate, train and predict. 

### Authenticate
The 'authenticate' endpoint must first be used in order to generate an access token, which 
is required by the latter endpoints. It accepts a valid email address and password as parameters.
The access token will expire after 60 minutes.

### Train
The 'train' endpoint, which accepts a collection of JSON objects. A machine learning model
is created and trained on the collection, and a measure of the model's performance is returned.

### Predict
The 'predict' endpoint accepts a single JSON object of data relevant to a specific date. The
trained machine learning model uses the data as input and generates a prediction, which is then returned.

Examples of proper and improper use of the API can be found at https://documenter.getpostman.com/view/18259998/Uyr5oeeF.


## Troubleshooting
- If an error occurs prompting you to check the versions of Python and Numpy and you are using Visual Studio, click Ctrl+Shift+P > Terminal: Select Default Profile > Command Prompt, and re-run.
- SQLAlchemy "could not find database file" error - make sure that the active directory is the entire project i.e. 'front' and 'back', rather than just the 'model' directory.










