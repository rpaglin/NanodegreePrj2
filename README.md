# NanodegreePrj2
# About The Project
The project is intended as a training exercise and is related to machine learning classification of messages exchanged during an emergency 

## Prerequisites

The software uses the following libraries: 
- numpy
- pandas
- plotly
- sklearn
- nltk
- sqlalchemy
- pickle
- re 
- json
- flask

## Folder organization

The software is organized into three different folders, named 'data', 'models' and 'app'

Folder 'data' contains the two initial data sources, 'disaster_messages.csv' and 'disaster_categories.csv'. It also contains the python module 'process_data.py' which reads csv files, clean and merge info and finally stores the outcome into a table of an sql db in the same data directory. Db is named DisasterRespons.db

Folder 'models' include a python script 'train_classifier.py' which reads message date from the db and implement a message classifier through a natural langualge manipulation pipeline. Clasification is based om random forest nltk module

Folder 'app' include a py module for web visualization of results using flask. It also contains a subfolder with html templates

## Usage

The code was developed and tested within udacity workspace. Web app also works in that workspcae. 
Complete usage of the software consists of three steps. Procedure should be the following one:

Step1 (ETL). In folder data: python process_data.py disaster_messages.csv disaster_categories.csv DisasterResponse.db 

Step2 (Classifier). In folder models: python train_classifier.py ../data/DisasterResponse.db classifier.pkl 

Step3 (Flask web app). In folder app: python run.py

<img width=“964” alt=“java 8 and prio java 8  array review example” src=“https://github.com/rpaglin/NanodegreePrj2/blob/main/pictures/dataset_cat.png”>

## Note

The train_classifier module includes a boolean variable 'optimize', which (when True) triggers execution of gridsearch

## Note

The train_classifier module includes a boolean variable 'optimize', which (when True) triggers execution of gridsearch

## Note

The train_classifier module includes a boolean variable 'optimize', which (when True) triggers execution of gridsearch

## License

Free to use

## Contact

roberto.paglino@gmail.com

## Project Link: 

https://github.com/rpaglin/NanodegreePrj2
