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

### Model Selection Rationale

For the task of predicting mental health conditions, we evaluated three models:

1. **Logistic Regression**  
2. **Random Forest**  
3. **XGBoost**

These models were trained and tested using the `mental_health` dataset, which contains a variety of features related to mental health conditions, such as family history, work interference, mental health consequences, and more. The dataset was preprocessed, and relevant features were extracted.

#### Model Evaluation

Each model was evaluated based on several metrics:

- **Accuracy**: How often the model correctly predicts the mental health condition.
- **Precision**: The proportion of true positive predictions among all positive predictions.
- **Recall**: The proportion of true positive predictions among all actual positives.
- **F1-score**: A harmonic mean of precision and recall, providing a single metric for evaluating the model's performance.
- **ROC-AUC**: A measure of the model's ability to distinguish between classes, with higher values indicating better performance.

#### Results:

- **Logistic Regression**: Achieved perfect performance across all metrics (Accuracy: 1.0000, Precision: 1.0000, Recall: 1.0000, F1-score: 1.0000).
- **Random Forest**: Performed slightly lower with an F1-score of 0.9637 but had a high ROC-AUC of 1.0000.
- **XGBoost**: Achieved excellent performance with an F1-score of 0.9880 and high accuracy, precision, and recall, also with a perfect ROC-AUC score.

#### Best Model Selection

Although all models performed well, **Logistic Regression** achieved a perfect performance across all metrics. However, given the slight overfitting seen in Logistic Regression and the high stability of **XGBoost**, **XGBoost** was chosen as the final model for the mental health condition prediction task. The **XGBoost** model exhibited robust performance with an F1-score of 0.9880, precision of 0.9883, and recall of 0.9880, making it the most reliable choice.

#### Model Saving & SHAP Interpretation

The final model (XGBoost) was saved as `mental_health_model.pkl` for deployment. Additionally, the model was interpreted using SHAP (SHapley Additive exPlanations) to understand the influence of each feature on the model's predictions. The SHAP values were visualized to provide insights into which features (e.g., family history, work interference) most strongly influence mental health condition predictions.


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



