# CPSC322-Final-Project
## Cole Thorpen & Payton Burks

This project uses classifiers such as kNN, random forest, decision trees, etc. to predict drug effectiveness by patient.

project.ipynb contains the bulk of 'results' and output. Actual code for the classifiers is stored in mysklearn. A test file, flask deployment files, and utility functions for the jupyter notebook are stored in main as well.

The datasets can be found in input_data.

Dataset Source: [UCI ML Drug Review Dataset](https://www.kaggle.com/datasets/jessicali9530/kuc-hackathon-winter-2018?select=drugsComTest_raw.csv)

### Deployment
The Heroku deployment can be found [here](https://drug-effectiveness.herokuapp.com)  

Example Predictions:
- https://drug-effectiveness.herokuapp.com/predict/Drug=benzonatate&Age=0-2&Condition=Cough&Season=summer&EaseofUse=E&Satisfaction=E&Sex=Female
- https://drug-effectiveness.herokuapp.com/predict?Drug=coumadin&Age=25-34&Condition=Blood%20Clot&Season=fall&EaseofUse=NE&Satisfaction=NE&Sex=Female
