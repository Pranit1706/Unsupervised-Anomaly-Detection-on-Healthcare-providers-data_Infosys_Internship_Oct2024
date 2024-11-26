import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.decomposition import PCA
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score, f1_score

df = pd.read_csv('Updated_HealthCare.csv')
cols = ['Number of Services', 'Number of Medicare Beneficiaries', 
                  'Average Medicare Allowed Amount', 'Average Submitted Charge Amount', 
                  'Average Medicare Payment Amount', 'Number of Distinct Medicare Beneficiary/Per Day Services',
                  'Average Medicare Standardized Amount']

input_dim = df[cols].shape[1]  

input_layer = Input(shape=(input_dim,))
encoded = Dense(64, activation='relu')(input_layer)
encoded = Dense(32, activation='relu')(encoded) 

decoded = Dense(64, activation='relu')(encoded)
decoded = Dense(input_dim, activation='sigmoid')(decoded)

autoencoder = Model(inputs=input_layer, outputs=decoded, name = "autoencoder_model")
autoencoder.compile(optimizer='adam', loss='mean_squared_error')

print(autoencoder.summary()) 

X_train, X_test = train_test_split(df[cols].values, test_size=0.2, random_state=42)

history = autoencoder.fit(
    X_train, X_train,  
    epochs=10,  
    batch_size=32,  
    validation_data=(X_test, X_test),
    verbose=1
)

plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
X_train_reconstructed = autoencoder.predict(X_train)
X_test_reconstructed = autoencoder.predict(X_test)

train_error = np.mean(np.square(X_train - X_train_reconstructed), axis=1)
test_error = np.mean(np.square(X_test - X_test_reconstructed), axis=1)

plt.figure(figsize=(12, 6))
plt.hist(train_error, bins=50, alpha=0.6, label='Train')
plt.hist(test_error, bins=50, alpha=0.6, label='Test')
plt.xlabel('Reconstruction Error')
plt.ylabel('Frequency')
plt.legend(loc='upper right')
plt.title('Reconstruction Error Distribution')
plt.show()

reconstruction_error = np.mean(np.abs(X_test - autoencoder.predict(X_test)), axis=1)

threshold = np.percentile(reconstruction_error, 95)

anomaly_labels = (reconstruction_error > threshold).astype(int)
plt.figure(figsize=(10, 6))
plt.hist(reconstruction_error, bins=50, alpha=0.7, label="Reconstruction Error")
plt.axvline(threshold, color='red', linestyle='--', label=f'Threshold (95th percentile)')
plt.xlabel('Reconstruction Error')
plt.ylabel('Frequency')
plt.legend()
plt.title('Reconstruction Error Distribution with Anomaly Threshold')
plt.show()

pca = PCA(n_components=2)
X_test_pca = pca.fit_transform(X_test)

plt.figure(figsize=(10, 6))
plt.scatter(X_test_pca[:, 0], X_test_pca[:, 1], c=anomaly_labels, cmap='coolwarm', label='Data Points')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.title('Anomaly Detection with Autoencoder (PCA Visualization)')
plt.show()
unique, counts = np.unique(anomaly_labels, return_counts=True)
print(f"Normal data points (0): {counts[0]}")
print(f"Anomalous data points (1): {counts[1]}")

anomaly_labels = (reconstruction_error > threshold).astype(int)
anomaly_labels_df = pd.DataFrame(anomaly_labels, columns=["Anomaly"])

X_test_df = pd.DataFrame(X_test, columns=cols)
X_test_df["Anomaly"] = anomaly_labels_df["Anomaly"].values

anomalies = X_test_df[X_test_df["Anomaly"] == 1]

anomalies.head()
X = df[cols].values
reconstruction_error = np.mean(np.abs(X - autoencoder.predict(X)), axis=1)
threshold = np.percentile(reconstruction_error, 95)
y_true = np.where(reconstruction_error > threshold, 1, 0)

iso_forest = IsolationForest(contamination=0.06, random_state=42)  
iso_forest.fit(reconstruction_error.reshape(-1, 1))  

y_pred = iso_forest.predict(reconstruction_error.reshape(-1, 1))
y_pred = (y_pred == -1).astype(int)

print(confusion_matrix(y_true, y_pred))

precision = precision_score(y_true, y_pred)
print(f"Precision: {precision}")

recall = recall_score(y_true, y_pred)
print(f"Recall: {recall}")

accuracy = accuracy_score(y_true, y_pred)
print(f"Accuracy: {accuracy}")

f1 = f1_score(y_true, y_pred)
print(f"F1 Score: {f1}")

best_model = IsolationForest(
    n_estimators=250, 
    max_samples=0.689708907570723, 
    contamination=0.011181951395629604,
    random_state=42
)

best_model.fit(X)

y_pred_iforest = best_model.predict(X)
y_pred_iforest = (y_pred_iforest == -1).astype(int) 

conf_matrix = confusion_matrix(y_true, y_pred_iforest)
accuracy = accuracy_score(y_true, y_pred_iforest)
precision = precision_score(y_true, y_pred_iforest)
recall = recall_score(y_true, y_pred_iforest)
f1 = f1_score(y_true, y_pred_iforest)

print("Confusion Matrix:")
print(conf_matrix)
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")

best_model = IsolationForest(
    n_estimators=200, 
    max_samples=0.1, 
    contamination=0.1,
    random_state=42
)

best_model.fit(X)

y_pred_iforest = best_model.predict(X)
y_pred_iforest = (y_pred_iforest == -1).astype(int) 

conf_matrix = confusion_matrix(y_true, y_pred_iforest)
accuracy = accuracy_score(y_true, y_pred_iforest)
precision = precision_score(y_true, y_pred_iforest)
recall = recall_score(y_true, y_pred_iforest)
f1 = f1_score(y_true, y_pred_iforest)

print("Confusion Matrix:")
print(conf_matrix)
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")

best_model = IsolationForest(
    n_estimators=200, 
    max_samples=0.1, 
    contamination=0.03,
    random_state=42
)

best_model.fit(X)

y_pred_iforest = best_model.predict(X)
y_pred_iforest = (y_pred_iforest == -1).astype(int) 

conf_matrix = confusion_matrix(y_true, y_pred_iforest)
accuracy = accuracy_score(y_true, y_pred_iforest)
precision = precision_score(y_true, y_pred_iforest)
recall = recall_score(y_true, y_pred_iforest)
f1 = f1_score(y_true, y_pred_iforest)

print("Confusion Matrix:")
print(conf_matrix)
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")

def check_anomaly_reshaped(row_number, df, autoencoder, threshold, cols):
    try:
        row = df[cols].iloc[row_number].values.reshape(1, -1)  
        reconstructed = autoencoder.predict(row)

        reconstruction_error = np.mean((row - reconstructed) ** 2)

        if reconstruction_error > threshold:
            return f"Row {row_number} is an **anomaly**. Reconstruction error: {reconstruction_error:.4f}"
        else:
            return f"Row {row_number} is **normal**. Reconstruction error: {reconstruction_error:.4f}"
    except IndexError:
        return f"Invalid row number. Please select a row between 0 and {len(df) - 1}."
    except Exception as e:
        return f"An error occurred: {str(e)}"

if __name__ == "__main__":
    threshold = 0.02  

    cols = ["Number of Services", "Number of Medicare Beneficiaries", 
            "Average Medicare Allowed Amount", "Average Submitted Charge Amount", 
            "Average Medicare Payment Amount", "Number of Distinct Medicare Beneficiary/Per Day Services", 
            "Average Medicare Standardized Amount"]

    # Prompt user to input a row number
    try:
        row_num = int(input(f"Enter the row number to check (0 to {len(df) - 1}): "))
        result = check_anomaly_reshaped(row_num, df, autoencoder, threshold, cols)
        print(result)
    except ValueError:
        print("Please enter a valid integer for the row number.")
