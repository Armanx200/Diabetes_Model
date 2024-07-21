import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

# Import and preprocess data
df = pd.read_csv('diabetes_dataset.csv')

# One-hot encode categorical variables
df = pd.get_dummies(df, columns=['gender', 'location', 'smoking_history'], drop_first=True)

# Split data into features and target
X = df.drop('diabetes', axis=1)
y = df['diabetes']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the logistic regression model with increased iterations
model = LogisticRegression(max_iter=5000, solver='lbfgs')
model.fit(X_train, y_train)

# Make predictions and evaluate the model
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Check and handle multicollinearity
X = X.apply(pd.to_numeric, errors='coerce')
y = pd.to_numeric(y, errors='coerce')

# Drop rows with NaNs
X = X.dropna()
y = y[X.index]

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Convert scaled data to DataFrame and add constant for VIF calculation
X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)
X_scaled_df = add_constant(X_scaled_df)

# Calculate VIF
vif_data = pd.DataFrame()
vif_data["feature"] = X_scaled_df.columns
vif_data["VIF"] = [variance_inflation_factor(X_scaled_df.values, i) for i in range(X_scaled_df.shape[1])]
print(vif_data)

# Drop features with high VIF
high_vif_features = vif_data[vif_data["VIF"] > 10]["feature"]
X_scaled_df = X_scaled_df.drop(columns=high_vif_features)

# Fit the logistic regression model with regularization (L2)
model = LogisticRegression(solver='liblinear')
model.fit(X_scaled_df, y)

# Print model coefficients and evaluate
print("Model coefficients:", model.coef_)
y_pred = model.predict(X_scaled_df)
print(classification_report(y, y_pred))

# Create subplots for visualization
fig, axs = plt.subplots(2, 2, figsize=(18, 12))
fig.suptitle('Diabetes Analysis', fontsize=16)

# Plot correlation matrix
axs[0, 0].set_title('Correlation Matrix')
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', ax=axs[0, 0])

# Gender analysis
gender_columns = [col for col in df.columns if col.startswith('gender_')]
if len(gender_columns) > 1:
    df['gender'] = df[gender_columns].idxmax(axis=1).str.replace('gender_', '')
    df['gender_diabetes'] = df['gender'] + '_' + df['diabetes'].astype(str)
    gender_diabetes_distribution = df.groupby(['gender', 'diabetes']).size().unstack()
    gender_diabetes_distribution.plot(kind='bar', stacked=True, ax=axs[0, 1])
    axs[0, 1].set_title('Distribution of Diabetes by Gender')
    axs[0, 1].set_xlabel('Gender')
    axs[0, 1].set_ylabel('Count')

# Location analysis
location_cols = [col for col in df.columns if col.startswith('location_')]
if location_cols:
    location_diabetes_distribution = df[location_cols].copy()
    location_diabetes_distribution['diabetes'] = df['diabetes']
    location_diabetes_distribution = location_diabetes_distribution.melt(id_vars='diabetes', var_name='location', value_name='count')
    location_diabetes_distribution = location_diabetes_distribution[location_diabetes_distribution['count'] == 1]
    location_diabetes_distribution = location_diabetes_distribution.groupby(['location', 'diabetes']).size().unstack(fill_value=0)
    location_diabetes_distribution.plot(kind='bar', stacked=True, ax=axs[1, 0])
    axs[1, 0].set_title('Distribution of Diabetes by Location')
    axs[1, 0].set_xlabel('Location')
    axs[1, 0].set_ylabel('Count')

# Comorbidity analysis
comorbidities = ['hypertension', 'heart_disease']
for idx, condition in enumerate(comorbidities):
    if condition in df.columns:
        axs[1, 1].set_title(f'Distribution of Diabetes by {condition.capitalize()}')
        condition_diabetes_distribution = df.groupby(condition)['diabetes'].value_counts().unstack()
        condition_diabetes_distribution.plot(kind='bar', stacked=True, ax=axs[1, 1])
        axs[1, 1].set_xlabel(condition.capitalize())
        axs[1, 1].set_ylabel('Count')
        break  # Only plot the first condition

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
