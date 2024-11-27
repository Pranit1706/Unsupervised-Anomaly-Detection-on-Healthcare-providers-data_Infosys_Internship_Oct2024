import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import GaussianNoise
from sklearn.model_selection import train_test_split

df = pd.read_csv('Updated_HealthCare.csv')

cols = ['Number of Services', 'Number of Medicare Beneficiaries', 
                  'Average Medicare Allowed Amount', 'Average Submitted Charge Amount', 
                  'Average Medicare Payment Amount', 'Number of Distinct Medicare Beneficiary/Per Day Services',
                  'Average Medicare Standardized Amount']
X = df[cols].values 

noise_factor = 0.2
X_noisy = X + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=X.shape)
X_noisy = np.clip(X_noisy, 0., 1.) 

X_train, X_test, X_train_noisy, X_test_noisy = train_test_split(X, X_noisy, test_size=0.2, random_state=42)

input_dim = X.shape[1]

input_layer = Input(shape=(input_dim,))
noisy_input = GaussianNoise(0.2)(input_layer)  

encoded = Dense(128, activation='relu')(noisy_input)
encoded = Dense(64, activation='relu')(encoded)
encoded = Dense(32, activation='relu')(encoded)

decoded = Dense(64, activation='relu')(encoded)
decoded = Dense(128, activation='relu')(decoded)
decoded = Dense(input_dim, activation='sigmoid')(decoded)  

autoencoder = Model(inputs=input_layer, outputs=decoded)
autoencoder.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
print(autoencoder.summary()) 

history = autoencoder.fit(X_train_noisy, X_train,
                          epochs=15,
                          batch_size=32,
                          shuffle=True,
                          validation_data=(X_test_noisy, X_test))

plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

reconstruction_error = np.mean(np.abs(X - autoencoder.predict(X)), axis=1)
threshold = np.percentile(reconstruction_error, 95)

anomalous_indices = np.where(reconstruction_error > threshold)[0]
anomalous_dataset = df.iloc[anomalous_indices].copy()

anomalous_dataset['Reconstruction Error'] = reconstruction_error[anomalous_indices]

print("Anomalous Dataset:")
anomalous_dataset.head()

X = df[cols].values
reconstruction_error = np.mean(np.abs(X - autoencoder.predict(X)), axis=1)
threshold = np.percentile(reconstruction_error, 95)
y_true = np.where(reconstruction_error > threshold, 1, 0)

from sklearn.ensemble import IsolationForest

iso_forest = IsolationForest(contamination=0.06, random_state=42)  
iso_forest.fit(reconstruction_error.reshape(-1, 1))  

y_pred = iso_forest.predict(reconstruction_error.reshape(-1, 1))
y_pred = (y_pred == -1).astype(int)

from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score, f1_score

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
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

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
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

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
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

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

    cols = ["Number of Services", "Number of Medicare Beneficiaries", 
            "Average Medicare Allowed Amount", "Average Submitted Charge Amount", 
            "Average Medicare Payment Amount", "Number of Distinct Medicare Beneficiary/Per Day Services", 
            "Average Medicare Standardized Amount"]

    try:
        row_num = int(input(f"Enter the row number to check (0 to {len(df) - 1}): "))
        result = check_anomaly_reshaped(row_num, df, autoencoder, threshold, cols)
        print(result)
    except ValueError:
        print("Please enter a valid integer for the row number.")

