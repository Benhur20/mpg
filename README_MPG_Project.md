
# Miles per Gallon (MPG) Prediction Project

## Overview
This project aims to predict the miles per gallon (MPG) of vehicles based on various features. The dataset contains information about different car models, including their specifications, which are used to train regression models to estimate MPG. This project implements various regression models and compares their performance to identify the best model for MPG prediction.

## Dataset
The dataset used in this project is `auto-mpg.csv`, which includes the following columns:
- `mpg`: The target variable, representing miles per gallon.
- `cylinders`: Number of cylinders in the car's engine.
- `displacement`: Engine displacement (in cubic inches).
- `horsepower`: Engine horsepower.
- `weight`: Vehicle weight (in lbs).
- `acceleration`: Time taken to accelerate from 0 to 60 mph (in seconds).
- `model year`: Year of the car model.
- `origin`: Origin of the car (e.g., USA, Europe, Japan).

## Project Structure
- **`Miles_per_Gallon_Project.ipynb`**: Jupyter notebook containing all code for data loading, preprocessing, model training, and evaluation.
- **`auto-mpg.csv`**: Dataset file (not included here but assumed to be in the same directory).
- **`best_model.pkl`**: The saved model file of the best-performing model.

## Steps in the Notebook

### 1. Data Loading
The dataset is loaded using `pandas`, and an initial inspection is performed to understand its structure and check for any missing values.

### 2. Data Cleaning
- **Duplicates**: Removed any duplicate rows to ensure the dataset is unique.
- **Missing Values**: Dropped rows with missing values to ensure data quality.

### 3. Exploratory Data Analysis (EDA)
Data visualizations are created using `matplotlib` and `seaborn` to explore relationships and distributions of features, identify patterns, and understand feature correlations with MPG.

### 4. Data Preprocessing
- **Standardization**: Features are standardized using `StandardScaler` to ensure they have similar scales, which can improve model performance.

### 5. Model Training
Various regression models are trained and evaluated, including:
- **Linear Regression**
- **Decision Tree Regressor**
- **Support Vector Regressor (SVR)**
- **Random Forest Regressor**
- **K-Nearest Neighbors Regressor**
- **XGBoost Regressor**

Each model is evaluated on the test set using the following metrics:
- **Mean Squared Error (MSE)**: Measures the average squared difference between predicted and actual values. Lower values are better.
- **R-squared (R²)**: Measures how well the regression model explains the variability of the target variable. Higher values are better.

### 6. Model Evaluation
The models are compared based on MSE and R² scores to identify the best-performing model for MPG prediction. 

### 7. Model Saving
The best-performing model is saved using `joblib` as `best_model.pkl`, enabling future use without retraining.

## Requirements
Install the necessary Python libraries using the following command:
```bash
pip install numpy pandas joblib matplotlib seaborn scikit-learn xgboost
```

## How to Run
1. **Clone the Repository**: Clone or download the project files to your local machine.
2. **Open the Notebook**: Launch `Miles_per_Gallon_Project.ipynb` in Jupyter Notebook or Jupyter Lab.
3. **Run Each Cell**: Execute each cell in sequence to load the data, preprocess it, train the models, and view the evaluation results.
4. **Save or Load Model**: The best model is saved as `best_model.pkl` after training. You can load it in future projects without retraining.

## Results
The project concludes by identifying the best-performing model for MPG prediction, which is saved for potential deployment or further analysis.

## License
This project is open-source and available under the MIT License.
