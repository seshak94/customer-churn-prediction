import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.metrics import confusion_matrix

# Load dataset (update path to your CSV file)
df = pd.read_csv('customer_churn.csv')  # Replace with actual path

# Convert TotalCharges to numeric, handle errors
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'].fillna(df['TotalCharges'].mean(), inplace=True)

# A) Data Manipulation
# a. Total number of male customers
male_customers = df[df['gender'] == 'Male'].shape[0]
print(f"Total number of male customers: {male_customers}")

# b. Total number of customers with DSL Internet Service
dsl_customers = df[df['InternetService'] == 'DSL'].shape[0]
print(f"Total number of customers with DSL Internet Service: {dsl_customers}")

# c. Female senior citizens with Mailed check payment method
new_customer = df[(df['gender'] == 'Female') & 
                 (df['SeniorCitizen'] == 1) & 
                 (df['PaymentMethod'] == 'Mailed check')]
print(f"Number of female senior citizens with Mailed check: {new_customer.shape[0]}")
new_customer.to_csv('new_customer_senior_female_mailed.csv', index=False)

# d. Customers with tenure < 10 months or TotalCharges < 500
new_customer = df[(df['tenure'] < 10) | (df['TotalCharges'] < 500)]
print(f"Number of customers with tenure < 10 months or TotalCharges < 500: {new_customer.shape[0]}")
new_customer.to_csv('new_customer_tenure_totalcharges.csv', index=False)

# B) Data Visualization
# a. Pie-chart for Churn distribution
churn_counts = df['Churn'].value_counts()
plt.figure(figsize=(6, 6))
plt.pie(churn_counts, labels=['No Churn', 'Churn'], autopct='%1.1f%%', colors=['#66b3ff', '#ff9999'])
plt.title('Churn Distribution')
plt.savefig('churn_pie_chart.png')
plt.close()

# b. Bar-plot for Internet Service distribution
plt.figure(figsize=(8, 6))
sns.countplot(x='InternetService', data=df, palette='Blues')
plt.title('Distribution of Internet Service')
plt.xlabel('Internet Service')
plt.ylabel('Count')
plt.savefig('internet_service_bar_plot.png')
plt.close()

# C) Model Building
# Prepare data for models
scaler = StandardScaler()

# Model 1: Using 'tenure' as feature
X = df[['tenure']].values
y = df['Churn'].values
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

model1 = Sequential([
    Dense(12, activation='relu', input_shape=(1,)),
    Dense(8, activation='relu'),
    Dense(1, activation='sigmoid')
])
model1.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
history1 = model1.fit(X_train, y_train, epochs=150, batch_size=32, validation_split=0.2, verbose=0)
y_pred1 = (model1.predict(X_test) > 0.5).astype(int)
cm1 = confusion_matrix(y_test, y_pred1)
print("\nModel 1 Confusion Matrix:")
print(cm1)

plt.figure(figsize=(8, 6))
plt.plot(history1.history['accuracy'], label='Training Accuracy')
plt.plot(history1.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model 1: Accuracy vs Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('model1_accuracy_epochs.png')
plt.close()

# Model 2: Using 'tenure' with dropout layers
model2 = Sequential([
    Dense(12, activation='relu', input_shape=(1,)),
    Dropout(0.3),
    Dense(8, activation='relu'),
    Dropout(0.2),
    Dense(1, activation='sigmoid')
])
model2.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
history2 = model2.fit(X_train, y_train, epochs=150, batch_size=32, validation_split=0.2, verbose=0)
y_pred2 = (model2.predict(X_test) > 0.5).astype(int)
cm2 = confusion_matrix(y_test, y_pred2)
print("\nModel 2 Confusion Matrix:")
print(cm2)

plt.figure(figsize=(8, 6))
plt.plot(history2.history['accuracy'], label='Training Accuracy')
plt.plot(history2.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model 2: Accuracy vs Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('model2_accuracy_epochs.png')
plt.close()

# Model 3: Using 'tenure', 'MonthlyCharges', 'TotalCharges'
X = df[['tenure', 'MonthlyCharges', 'TotalCharges']].values
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

model3 = Sequential([
    Dense(12, activation='relu', input_shape=(3,)),
    Dense(8, activation='relu'),
    Dense(1, activation='sigmoid')
])
model3.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
history3 = model3.fit(X_train, y_train, epochs=150, batch_size=32, validation_split=0.2, verbose=0)
y_pred3 = (model3.predict(X_test) > 0.5).astype(int)
cm3 = confusion_matrix(y_test, y_pred3)
print("\nModel 3 Confusion Matrix:")
print(cm3)

plt.figure(figsize=(8, 6))
plt.plot(history3.history['accuracy'], label='Training Accuracy')
plt.plot(history3.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model 3: Accuracy vs Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('model3_accuracy_epochs.png')
plt.close()

# Save models
model1.save('model1_churn.h5')
model2.save('model2_churn.h5')
model3.save('model3_churn.h5')