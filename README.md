
# Linear Regression for House Pricing Machine Learning Project

This machine learning project is part of an independent study by Mr Riboulet Ronan.\
The aim of this project is to predict the price of a property using features contained in a dataset.\
The data set comes from Kaggle with this given URL: 
- https://www.kaggle.com/datasets/yasserh/housing-prices-dataset/data\
The feature exploring and the EDA are not a big part of this project which is more focused on its dockerization and its deployment on the cloud platform Heroku with GitHub actions






## Run the App Locally

Clone the project

```bash
  git clone https://github.com/RonanRibouletPython/Linear_Regression_House_Pricing.git
```

Go to the project directory

```bash
  cd Linear_Regression_House_Pricing
```

Create the python environment

```bash
  conda create -p venv python==3.12 -y
```

Download the required librairies

```bash
  pip install -r requirements.txt
```

Run the application

```bash
  python .\app.py
```



## Lessons Learned

During this project I reinforced my knowledge of basic feature engineering, EDA and model evaluation to solve linear regression problems.\
I also learned how to build a web application using the Flask API and how to use Postman to test it.
Later, I learned how to deploy a machine learning application using Github and Heroku.\
And I wanted to go further and explore the CI/CD deployment pipeline, so I created a Docker container and used Github actions to deploy the container to the Heroku cloud platform. 

