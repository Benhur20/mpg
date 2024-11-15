import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

df=pd.read_csv('/auto-mpg.csv')
df.head()

df.describe()

column_types=df.dtypes;
print(column_types);

# Remove duplicates
df.drop_duplicates(inplace=True)
#handling missing values
df.dropna(inplace=True)

# converting categorical data to numerical data using one-hot encoding

origin = df.pop('origin')
df['USA'] = (origin == 1)*1.0
df['Europe'] = (origin == 2)*1.0
df['Japan'] = (origin == 3)*1.0
df.tail()

df.drop(columns=['car name'],inplace=True)

df.head()

# Replace '?' with NaN
df['horsepower'] = df['horsepower'].replace('?', np.nan)

# Convert 'horsepower' column to float
df['horsepower'] = df['horsepower'].astype(float)

# Fill missing values with the mean
mean_horsepower = df['horsepower'].mean()
df['horsepower'].fillna(mean_horsepower, inplace=True)

# Convert 'horsepower' column to integer
df['horsepower'] = df['horsepower'].astype(int)

df.dtypes

# Scatterplots of 'mpg' against each numerical feature
numerical_features = ['cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model year','USA','Europe','Japan']

for feature in numerical_features:
    plt.figure(figsize=(8, 6))
    plt.scatter(df[feature], df['mpg'], alpha=0.5)
    plt.title(f'Scatterplot of {feature.capitalize()} vs MPG')
    plt.xlabel(feature.capitalize())
    plt.ylabel('MPG')
    plt.grid(True)
    plt.show()

# Compute the correlation matrix
corr_matrix = df.corr()

# Plot the heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Correlation Heatmap')
plt.show()

# Pairplot for visualizing relationships between numerical features and target variable
sns.pairplot(df, vars=['mpg', 'cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model year'], diag_kind='kde')
plt.show()

# Boxplot for visualizing the distribution of 'origin' feature with respect to 'mpg'
plt.figure(figsize=(12, 10))
plt.ylim(0, 300)
sns.boxplot(data=df)
plt.show()

# Calculate IQR
Q1 = df['horsepower'].quantile(0.25)
Q3 = df['horsepower'].quantile(0.75)
IQR = Q3 - Q1

# Define upper and lower bounds
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Identify outliers
outliers = (df['horsepower'] < lower_bound) | (df['horsepower'] > upper_bound)

# Impute outliers with median value
median_horsepower = df['horsepower'].median()
df_imputed = df.copy()
df_imputed.loc[outliers, 'horsepower'] = median_horsepower
df=df_imputed

# Replot boxplot to check if outliers are imputed
plt.figure(figsize=(8, 6))
sns.boxplot(x='horsepower', data=df)
plt.title('Boxplot of Horsepower (Outliers Imputed)')
plt.xlabel('Horsepower')
plt.show()

# Repeating the process to remove the remaining outliers

# Calculate IQR
Q1 = df['horsepower'].quantile(0.25)
Q3 = df['horsepower'].quantile(0.75)
IQR = Q3 - Q1

# Define upper and lower bounds
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Identify outliers
outliers = (df['horsepower'] < lower_bound) | (df['horsepower'] > upper_bound)

# Impute outliers with median value
median_horsepower = df['horsepower'].median()
df_imputed = df.copy()
df_imputed.loc[outliers, 'horsepower'] = median_horsepower
df=df_imputed

# Replot boxplot to check if outliers are imputed
plt.figure(figsize=(8, 6))
sns.boxplot(x='horsepower', data=df)
plt.title('Boxplot of Horsepower (Outliers Imputed)')
plt.xlabel('Horsepower')
plt.show()

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


X=df.drop(columns=['mpg'])
y=df['mpg']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)

#linear regression
from sklearn.linear_model import LinearRegression

# Create and train the model
linear_regression_model = LinearRegression()
linear_regression_model.fit(X_train, y_train)

# Predict on test set
prediction_linearReg = linear_regression_model.predict(X_test)
mse_linear = mean_squared_error(y_test, prediction_linearReg)
r2_linear = r2_score(y_test, prediction_linearReg)
#model evaluation
print("Linear Regression Results:")
print(f"MSE of : {mse_linear}")
print(f"R-squared: {r2_linear}\n")

#Decision Tree
from sklearn.tree import DecisionTreeRegressor

# Create and train the model
decision_tree_model = DecisionTreeRegressor()
decision_tree_model.fit(X_train, y_train)

# Predict on test set
y_pred_tree = decision_tree_model.predict(X_test)

#model_evaluation
mse_DecisionTree = mean_squared_error(y_test, y_pred_tree)
r2_DecisionTree = r2_score(y_test, y_pred_tree)
print("Decision Tree Results:")
print(f"MSE of : {mse_DecisionTree}")
print(f"R-squared: {r2_DecisionTree}\n")



# SVR
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create and train the model
svm_regressor = SVR(kernel='rbf')
svm_regressor.fit(X_train_scaled, y_train)

# Predict on test set
y_pred_svm = svm_regressor.predict(X_test_scaled)

mse_svm = mean_squared_error(y_test, y_pred_svm)
r2_svm = r2_score(y_test, y_pred_svm)
print("Support Vector Machines (SVM):")
print(f"MSE: {mse_svm}")
print(f"R-squared: {r2_svm}\n")


#Random Forest
from sklearn.ensemble import RandomForestRegressor

# Create and train the model
random_forest = RandomForestRegressor()
random_forest.fit(X_train, y_train)

# Predict on test set
y_pred_forest = random_forest.predict(X_test)

mse_forest = mean_squared_error(y_test, y_pred_forest)
r2_forest = r2_score(y_test, y_pred_forest)
print("Random Forest:")
print(f"MSE: {mse_forest}")
print(f"R-squared: {r2_forest}\n")

#KNN
from sklearn.neighbors import KNeighborsRegressor

# Create and train the model
knn_regressor = KNeighborsRegressor()
knn_regressor.fit(X_train, y_train)

# Predict on test set
y_pred_knn = knn_regressor.predict(X_test)

# Evaluate KNN
mse_knn = mean_squared_error(y_test, y_pred_knn)
r2_knn = r2_score(y_test, y_pred_knn)
print("K-Nearest Neighbors (KNN):")
print(f"MSE: {mse_knn}")
print(f"R-squared: {r2_knn}")

#creating new clean data frame
clean_df=pd.DataFrame()
clean_df=df
output_file = "clean_dataset.csv"
clean_df.to_csv(output_file, index=False)

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib

# Load the cleaned dataset
data = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/clean_dataset.csv')

# Separate features (X) and target variable (y)
X = data.drop(columns=['mpg'])
y = data['mpg']

# Initialize and train the final RandomForestRegressor model
final_model = RandomForestRegressor()
final_model.fit(X, y)

# Save the final model using joblib
joblib.dump(final_model, 'final_model.pkl')

import joblib
our_final_model=joblib.load('/content/drive/MyDrive/Colab Notebooks/final_model.pkl')

print("Enter the vehicle parameters:")
cylinders = int(input("Number of cylinders: "))
displacement = float(input("Displacement (in cubic inches): "))
horsepower = float(input("Horsepower: "))
weight = float(input("Weight (in pounds): "))
acceleration = float(input("Acceleration (0-60 mph in seconds): "))
model_year = int(input("Model year (19XX format just write the last 2 digits of the year): "))
origin = int(input("Origin (1 for USA, 2 for Europe, 3 for Japan): "))
USA=0
Europe=0
Japan=0
if(origin==1):
  USA=1

elif(origin==2):
  Europe=1

else:
  Japan=1


predicted_mpg = our_final_model.predict([[cylinders,displacement,horsepower,weight,acceleration,model_year,USA,Europe,Japan]])
print(" Predicted MPG ="+str(predicted_mpg))