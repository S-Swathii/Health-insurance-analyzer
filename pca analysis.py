import pandas as pd
from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.preprocessing import StandardScaler
import tkinter as tk
from tkinter import messagebox
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageTk

# Load your dataset
df = pd.read_csv('/Users/i.seviantojensima/Desktop/Jensi/Sem 4/pa/health insurance.csv')


numerical_features = ['age', 'bmi', 'children', 'charges']

# Preprocessing
# Standardize numerical features
scaler = StandardScaler()
df[numerical_features] = scaler.fit_transform(df[numerical_features])

# Function to format and display messages in a dialog box
def show_message(title, message):
    messagebox.showinfo(title, message)

# Function to perform PCA
def perform_pca():
    pca = PCA(n_components=None)
    principal_components = pca.fit_transform(df[numerical_features])

    # Create DataFrame for loadings
    loadings_df = pd.DataFrame(pca.components_, columns=numerical_features)

    # Explained variance ratio
    explained_variance_ratio = pca.explained_variance_ratio_

    pca_message = "PCA Loadings:\n" + str(loadings_df) + "\n\n"
    pca_message += "Explained variance ratio for each component:\n"
    for i, ratio in enumerate(explained_variance_ratio, 1):
        pca_message += f"Principal Component {i}: {ratio:.4f}\n"
    
    pca_message += "\nPCA Analysis Inference:\n"
    pca_message += "PCA effectively reduces the dimensionality of the dataset, with the first principal component capturing the largest variance, primarily influenced by charges and age.\n"
    pca_message += "The second principal component is dominated by children, indicating a separate dimension of variability.\n"
    pca_message += "Together, the first two components explain a significant portion of the variance, suggesting that charges, age, and children are key variables in the dataset."

    show_message("PCA Results", pca_message)
    plot_pca(principal_components, explained_variance_ratio)

# Function to plot PCA results
def plot_pca(principal_components, explained_variance_ratio):
    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    plt.scatter(principal_components[:, 0], principal_components[:, 1], c=df['charges'], cmap='viridis', alpha=0.7)
    plt.colorbar()
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('PCA Result')

    plt.subplot(1, 2, 2)
    plt.bar(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio)
    plt.xlabel('Principal Components')
    plt.ylabel('Variance Ratio')
    plt.title('Explained Variance Ratio')

    plt.suptitle('PCA Analysis')
    plt.show()

# Function to perform Factor Analysis
def perform_fa():
    n_factors = len(numerical_features)  
    fa = FactorAnalysis(n_components=n_factors, random_state=0)
    factor_components = fa.fit_transform(df[numerical_features])

    # DataFrame for factor loadings
    factor_loadings_df = pd.DataFrame(fa.components_, columns=numerical_features)

    # variance
    explained_variance = factor_loadings_df.var(axis=1)

    fa_message = "Factor Analysis Loadings:\n" + str(factor_loadings_df) + "\n\n"
    fa_message += "Explained variance for each factor:\n"
    for i, var in enumerate(explained_variance, 1):
        fa_message += f"Factor {i}: {var:.4f}\n"
    
    fa_message += "\nFactor Analysis Inference:\n"
    fa_message += "Factor Analysis reveals that most of the variance is captured by a single factor, heavily influenced by charges.\n"
    fa_message += "The subsequent factors do not contribute significantly to the explained variance, indicating that a single underlying factor (likely related to insurance charges) is predominant."

    show_message("Factor Analysis Results", fa_message)
    plot_fa(factor_components, factor_loadings_df)

# Function to plot Factor Analysis results
def plot_fa(factor_components, factor_loadings_df):
    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    plt.scatter(factor_components[:, 0], factor_components[:, 1], c=df['charges'], cmap='viridis', alpha=0.7)
    plt.colorbar()
    plt.xlabel('Factor 1')
    plt.ylabel('Factor 2')
    plt.title('Factor Analysis Result')

    plt.subplot(1, 2, 2)
    plt.bar(range(1, len(factor_loadings_df) + 1), factor_loadings_df.var(axis=1))
    plt.xlabel('Factors')
    plt.ylabel('Variance')
    plt.title('Factor Variance')

    plt.suptitle('Factor Analysis')
    plt.show()

# Function to show final comparison
def show_comparison():
    comparison_message = "Comparison:\n"
    comparison_message += "Both PCA and Factor Analysis identify charges as a major contributing factor.\n"
    comparison_message += "PCA provides a clearer picture of the individual contributions of age, bmi, and children through the different principal components.\n"
    comparison_message += "Factor Analysis simplifies the structure to one dominant factor, which might be useful for identifying a single underlying trait in the dataset."

    show_message("Comparison", comparison_message)

# Create the main window
root = tk.Tk()
root.title("PCA and Factor Analysis")

# Load the background image
background_image = Image.open("/Users/i.seviantojensima/Desktop/Jensi/Sem 4/pa/pca.jpg")
background_photo = ImageTk.PhotoImage(background_image)

# Create a label to display the background image
background_label = tk.Label(root, image=background_photo)
background_label.place(x=0, y=0, relwidth=1, relheight=1)

# Create buttons for PCA and FA
pca_button = tk.Button(root, text="Perform PCA", command=perform_pca)
fa_button = tk.Button(root, text="Perform Factor Analysis", command=perform_fa)
comparison_button = tk.Button(root, text="Show Comparison", command=show_comparison)

# Place buttons on the window
pca_button.place(relx=0.5, rely=0.3, anchor=tk.CENTER)
fa_button.place(relx=0.5, rely=0.5, anchor=tk.CENTER)
comparison_button.place(relx=0.5, rely=0.7, anchor=tk.CENTER)

# Run the application
root.mainloop()


