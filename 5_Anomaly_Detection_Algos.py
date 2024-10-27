import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
import category_encoders as ce

# Load the dataset
file_path = "C:/Users/shiva/Desktop/PROJECT 1/Cleaned_Healthcare_Providers.csv"
df = pd.read_csv(file_path)

# Select numeric columns for analysis
numeric_columns = [
    'Number of Services', 'Number of Medicare Beneficiaries',
    'Number of Distinct Medicare Beneficiary/Per Day Services',
    'Average Medicare Allowed Amount', 'Average Submitted Charge Amount',
    'Average Medicare Payment Amount', 'Average Medicare Standardized Amount'
]

# Standardize numeric columns
scaler = StandardScaler()
df[numeric_columns] = scaler.fit_transform(df[numeric_columns])

# Apply Binary Encoding for categorical features
binary_encoding_columns = ['Provider Type', 'Credentials of the Provider', 'Gender of the Provider', 'Entity Type of the Provider']
binary_encoder = ce.BinaryEncoder(cols=binary_encoding_columns)
df_encoded = binary_encoder.fit_transform(df)

# Further encode any remaining categorical columns
label_encoding_columns = ['Medicare Participation Indicator', 'Place of Service', 'HCPCS Code', 'HCPCS Drug Indicator']
for col in label_encoding_columns:
    df_encoded[col] = df_encoded[col].astype('category').cat.codes

# Drop any non-numeric columns and NaN values
df_encoded = df_encoded.apply(pd.to_numeric, errors='coerce').dropna(axis=1)

# Standardize the cleaned dataset
scaled_df = scaler.fit_transform(df_encoded)

# Sample the data for faster t-SNE and PCA processing
sample_size = 5000  # Adjust based on your dataset size
df_sampled = df_encoded.sample(n=sample_size, random_state=42)
scaled_sampled_df = scaler.fit_transform(df_sampled)

# Apply t-SNE with optimized parameters
tsne = TSNE(n_components=2, perplexity=20, max_iter=300, method='barnes_hut', random_state=42)
tsne_components = tsne.fit_transform(scaled_sampled_df)
tsne_df = pd.DataFrame(tsne_components, columns=['t-SNE1', 't-SNE2'])

# Apply PCA to the sampled data
pca = PCA(n_components=2, random_state=42)
pca_components = pca.fit_transform(scaled_sampled_df)
pca_df = pd.DataFrame(pca_components, columns=['PCA1', 'PCA2'])

# Perform anomaly detection algorithms on the sampled data
iso_forest = IsolationForest(contamination=0.05, random_state=42)
iso_labels = iso_forest.fit_predict(scaled_sampled_df)
tsne_df['Isolation_Forest'] = pca_df['Isolation_Forest'] = np.where(iso_labels == -1, 'Anomaly', 'Normal')

svm_model = OneClassSVM(kernel='rbf', gamma=0.001, nu=0.03)
svm_labels = svm_model.fit_predict(scaled_sampled_df)
tsne_df['OneClassSVM'] = pca_df['OneClassSVM'] = np.where(svm_labels == -1, 'Anomaly', 'Normal')

lof = LocalOutlierFactor(n_neighbors=20, contamination=0.05)
lof_labels = lof.fit_predict(scaled_sampled_df)
tsne_df['LocalOutlierFactor'] = pca_df['LocalOutlierFactor'] = np.where(lof_labels == -1, 'Anomaly', 'Normal')

dbscan = DBSCAN(eps=1.5, min_samples=5)
dbscan_labels = dbscan.fit_predict(scaled_sampled_df)
tsne_df['DBSCAN'] = pca_df['DBSCAN'] = np.where(dbscan_labels == -1, 'Anomaly', 'Normal')

# Visualization of Anomaly Detection Results in t-SNE and PCA Space
fig, axes = plt.subplots(4, 2, figsize=(18, 24), sharex=False, sharey=False)
fig.suptitle("Optimized Anomaly Detection in t-SNE and PCA Spaces", fontsize=18)

# t-SNE Visualizations
sns.scatterplot(x='t-SNE1', y='t-SNE2', hue='Isolation_Forest', style='Isolation_Forest', markers={'Normal': 'o', 'Anomaly': 'X'},
                palette={'Normal': 'blue', 'Anomaly': 'red'}, data=tsne_df, ax=axes[0, 0])
axes[0, 0].set_title("Isolation Forest (t-SNE)")

sns.scatterplot(x='t-SNE1', y='t-SNE2', hue='OneClassSVM', style='OneClassSVM', markers={'Normal': 'o', 'Anomaly': 'X'},
                palette={'Normal': 'green', 'Anomaly': 'red'}, data=tsne_df, ax=axes[1, 0])
axes[1, 0].set_title("One-Class SVM (t-SNE)")

sns.scatterplot(x='t-SNE1', y='t-SNE2', hue='LocalOutlierFactor', style='LocalOutlierFactor', markers={'Normal': 'o', 'Anomaly': 'X'},
                palette={'Normal': 'purple', 'Anomaly': 'red'}, data=tsne_df, ax=axes[2, 0])
axes[2, 0].set_title("Local Outlier Factor (t-SNE)")

sns.scatterplot(x='t-SNE1', y='t-SNE2', hue='DBSCAN', style='DBSCAN', markers={'Normal': 'o', 'Anomaly': 'X'},
                palette={'Normal': 'orange', 'Anomaly': 'red'}, data=tsne_df, ax=axes[3, 0])
axes[3, 0].set_title("DBSCAN (t-SNE)")

# PCA Visualizations
sns.scatterplot(x='PCA1', y='PCA2', hue='Isolation_Forest', style='Isolation_Forest', markers={'Normal': 'o', 'Anomaly': 'X'},
                palette={'Normal': 'blue', 'Anomaly': 'red'}, data=pca_df, ax=axes[0, 1])
axes[0, 1].set_title("Isolation Forest (PCA)")

sns.scatterplot(x='PCA1', y='PCA2', hue='OneClassSVM', style='OneClassSVM', markers={'Normal': 'o', 'Anomaly': 'X'},
                palette={'Normal': 'green', 'Anomaly': 'red'}, data=pca_df, ax=axes[1, 1])
axes[1, 1].set_title("One-Class SVM (PCA)")

sns.scatterplot(x='PCA1', y='PCA2', hue='LocalOutlierFactor', style='LocalOutlierFactor', markers={'Normal': 'o', 'Anomaly': 'X'},
                palette={'Normal': 'purple', 'Anomaly': 'red'}, data=pca_df, ax=axes[2, 1])
axes[2, 1].set_title("Local Outlier Factor (PCA)")

sns.scatterplot(x='PCA1', y='PCA2', hue='DBSCAN', style='DBSCAN', markers={'Normal': 'o', 'Anomaly': 'X'},
                palette={'Normal': 'orange', 'Anomaly': 'red'}, data=pca_df, ax=axes[3, 1])
axes[3, 1].set_title("DBSCAN (PCA)")

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()
