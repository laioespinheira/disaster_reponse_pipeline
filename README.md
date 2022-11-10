# Disaster Response Pipeline Project

### Project Motivation

Now more than ever the use of technology and in this case machine learning techniques are being used to assist us humans in every possible way. This is no different when a massive disaster happens whether it is natural or not. People need assistance right away, every minute counts. 

This project has the goal of filtering messages when a disaster happens to understand who needs help the most. This way we can help the local team identify quicker people in critical situations.

### The solution

To build this, first I had access to a lot of disaster messages and their categories. There're a total of 36 labels for different types of messages. 

Using this data I was able to train and test a classifier model that will help the local team identify what the person is in need depending on their message.

### Project Structure

	- app/: This is the folder where the flask app is stored
	- data/: csv files, database file, process_data.py(used to load and clean the data) and classifier.pkl
	- models/:  train_classifier.py, python script used to create the model for the app

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
