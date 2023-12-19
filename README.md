# Capstone project ML Zoomcamp 2023: Student Performance Prediction

## Description of problem and data

In this project, I examined how the student's performance (test scores) is affected by other variables such as Gender, Ethnicity, Parental level of education, Lunch and Test preparation course.

The link for the dataset:  https://www.kaggle.com/datasets/spscientist/students-performance-in-exams?datasetId=74977
You can find the dataset in my GitHub repository also: (https://github.com/rassel25/Student-Performance Prediction/tree/main/data)

The dataset contains the following columns:
- gender : sex of students -> (Male/female)
- race/ethnicity : ethnicity of students -> (Group A, B,C, D,E)
- parental level of education : parents' final education ->(bachelor's degree,some college,master's degree,associate's degree,high school)
- lunch : having lunch before test (standard or free/reduced)
- test preparation course : complete or not complete before test
- math score
- reading score
- writing score

The target variable is taken as math score column so it makes the project as regression problem.

## Description of project

The steps that is followed are:

- Understanding the Problem Statement
- Data Collection
- Data Cleaning
- Exploratory data analysis (EDA)
- Data Pre-Processing
- Model Training and Hyperparameter Tuning
- Model Evaluation
- Testing

## Data Cleaning

The steps that is followed are:

- Check Missing value
- Check Duplicates
- Check data type
- Check the number of unique values of each column
- Check statistics of data set
- Check various categories present in the different categorical column

## Exploratory data analysis (EDA)

The steps that is followed are:

- Univariate Analysis
- Bivariate Analysis
- Multivariate Analysis

## Data Pre-Processing

The steps that is followed are:

- Feature Engineering: total score and average score
- Converting Categorical variable to Numerical variable
- Train - Test split

## Model Training and Hyperparameter Tuning

Total 8 models have been used to train our dataset and Randomisedcv is used to hypertune the models and pick the best model.
The models are:

- Linear Regression
- Support Vector Regressor
- KNeighbors Regressor
- AdaBoost Regressor
- Decision Tree Regressor
- XGBoost Regressor
- CatBoost Regressor
- Random Forest Regressor

## Model Evaluation

Model performance for Training set
- Root Mean Squared Error: 3.6252
- Mean Absolute Error: 2.8984
- R2 Score: 0.9417
- Mean Squared Error: 13.1420
----------------------------------
Model performance for Test set
- Root Mean Squared Error: 3.5927
- Mean Absolute Error: 2.8200
- R2 Score: 0.9470
- Mean Squared Error: 12.9073

## Description of files in Github Repository

- Data: stud.csv

- Jupyter Notebook: notebook.ipynb with
    - Understanding the Problem Statement
    - Data Collection
    - Data Cleaning
    - Exploratory data analysis (EDA)
    - Data Pre-Processing
    - Model Training and Hyperparameter Tuning
    - Model Evaluation
    - Testing


- Script: train.py - in here the final model is build

- model.bin: The final model with its pipeline and parameters are saved by pickle to model.bin 

- predict.py. contains
  - Loading the model
  - Serving it via flask web service

- Files with dependencies: I used Poetry for dependency and management. All the dependencies are in pyproject.toml and poetry.lock files.

- Dockerfile: for containerise the project

- test_predict.py: to test the docker file

- Deployment: Used AWS Beanstalk to deploy the dockerized file

## Description of how to use the model

## Docker

- isoloate the environment from the host machine
- You can find docker image here https://hub.docker.com/_/python
- I have chosen python:3.11 to match my python version
- Build the docker image:  docker build -t student-prediction . 
- Run the docker image: docker run -it --rm -p 9696:9696 student-prediction   
- Test the docker image: python predict_test.py

## Deploy to AWS Elastic Beanstalk

- create an aws account
- install eb cli as dev dependency pipenv install awsebcli --dev
- go to virtual environment pipenv shell
- initial the eb init -p "Docker running on 64bit Amazon Linux 2" parkinson-disease
- ls -a to check whether there is .elasticbeanstalk folder
- ls .elasticbeanstalk/ to check the doc inside the folder config.yml
- run locally to test eb local run --port 9696
- in another terminal run python predict_test.py to test
- implement in the cloud: create a cloud environment -> eb create parkinson-disease-env
- copy the service link to predict_test.py, update our url
  
![aws](https://github.com/rassel25/Parkinson-Disease-Detection/assets/36706178/aa21508a-65ce-490f-ada8-f3f91e966093)

I have terminated AWS Beanstalk service to avoid generating extra fees.
