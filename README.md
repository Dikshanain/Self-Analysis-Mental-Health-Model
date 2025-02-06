# Self-Analysis-Mental-Health-Model
 A Self-Analysis Mental Health Model that predicts possible mental health conditions based on user-provided symptoms.

## Overview

This repository contains the code and necessary files for a machine learning-based model designed to predict mental health conditions based on user-provided data. The model predicts mental health conditions such as **Depression**, **Anxiety**, and **Bipolar Disorder** using various user inputs, including age, gender, work interference, family history of mental illness, and access to mental health resources.

The project includes:
1. An **inference script** (`predict_mental_health.py`) to make predictions based on user input.
2. A **CLI interface** for testing and interacting with the model (included in the inference script).
3. **Dataset preprocessing steps** and model training code (included in the inference script).
4. **Dataset**: "Mental Health in Tech Survey" from Kaggle.

## Files Included

- `predict_mental_health.py`: The main script that loads the model, processes user input, makes predictions, and provides coping strategies.
- `mental_health_model.pkl`: The trained machine learning model (saved using `joblib`).
- `label_encoders.pkl`: The label encoders used for transforming categorical data during inference.
- `README.md`: Documentation file providing details about the project, dataset, and usage.

## Dataset

The dataset used for training and testing the model is the **Mental Health in Tech Survey** from Kaggle. This dataset contains information about employees in the tech industry and their mental health status, including factors like age, gender, work interference, and access to mental health support. You can access the dataset [here](https://www.kaggle.com/datasets/osmi/mental-health-in-tech-survey).

### Dataset Preprocessing

The following preprocessing steps were applied to the dataset:

1. **Data Cleaning & Preprocessing**: Ensured consistency and usability of the dataset by handling missing values and removing irrelevant columns (e.g., timestamps).

2. **Handle Missing Values**: Gaps in the data were addressed to improve model accuracy and prevent bias.

3. **Normalization**: Continuous features, such as age, were normalized to ensure better model performance and comparability of data.

4. **Exploratory Data Analysis (EDA)**: Relationships between symptoms and mental health conditions were explored to identify key insights.

5. **Feature Engineering**: Categorical variables were encoded using label encoders, while symptoms and conditions were encoded as input features and labels.

6. **Feature Selection**: The most impactful features for mental health prediction were identified, allowing the model to focus on the most relevant data for prediction.

## How to Run the Inference Script

To run the inference script and predict mental health conditions, follow these steps:

1. Clone or download this repository.
2. Install the necessary libraries:
   ```bash
   pip install -r requirements.txt
3. Run the `predict_mental_health.py` Script
   Clone or download this repository to your local machine:
   ```bash
   git clone <repository_url>
 4. Input Prompts
    You will be prompted to input the following details:
    
    Age

    Gender (Male/Female/Other)
    
    Work interference (Often/Rarely/Never)
    
    Family history of mental illness (Yes/No)
    
    Mental health benefits (Yes/No)
    
    Access to care options (Yes/No)
    
    Preference for anonymity (Yes/No)
    
    Difficulty of taking mental health leave (Very Easy/Somewhat Easy/Somewhat Difficult/Very Difficult)
6. Output
   After providing the inputs, the model will predict a mental health condition (e.g., Depression, Anxiety, Bipolar Disorder) and suggest a coping strategy.

## How to Use the CLI

Follow the instructions on the terminal to input the necessary information.

The model will output:

Predicted mental health condition (e.g., Depression, Anxiety, Bipolar Disorder)
A coping strategy based on the predicted condition.



