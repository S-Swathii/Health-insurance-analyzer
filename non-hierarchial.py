'''import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
import seaborn as sns
import tkinter as tk
from tkinter import messagebox

class HealthInsuranceAnalysisApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Health Insurance Data Analysis")
        self.root.geometry("800x600")  # height for video
        self.root.configure(background='white')  # background color

        # Perform clustering and plot the graph
        self.perform_non_hierarchical_clustering()

   
        # Load the dataset
        csv_path = "/Users/i.seviantojensima/Desktop/Sem 4/pa/health insurance.csv"  # Update the file path
        if not os.path.isfile(csv_path):
            messagebox.showerror("Error", "CSV file not found!")
            return

        data = pd.read_csv(csv_path)

        # Select columns for clustering
        columns_for_clustering = ['age', 'bmi']

        # Check if 'bmi' column exists
        if 'bmi' not in data.columns:
            messagebox.showerror("Error", "'bmi' column not found in the dataset. Clustering cannot be performed.")
            return

        # Handle missing values (replace NaNs with column means)
        data[columns_for_clustering] = data[columns_for_clustering].fillna(data[columns_for_clustering].mean())

        # Standardize numerical features
        scaler = StandardScaler()
        data[columns_for_clustering] = scaler.fit_transform(data[columns_for_clustering])

        # Perform hierarchical clustering
        model = AgglomerativeClustering(n_clusters=3, linkage='ward')
        clusters = model.fit_predict(data[columns_for_clustering])

        # Add cluster labels to the dataset
        data['cluster'] = clusters

        # Map cluster labels to meaningful names
        cluster_mapping = {0: 'Underweight: BMI less than 18.5', 
                           1: 'Normal Weight: BMI between 18.5 and 24.9', 
                           2: 'Overweight: BMI between 25 and 49'}
        data['cluster'] = data['cluster'].map(cluster_mapping)

        # Visualize the clusters
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x='age', y='bmi', hue='cluster', data=data, palette='viridis', legend='full')
        plt.title('Non-Hierarchical Clustering')
        plt.xlabel('Age')
        plt.ylabel('BMI')
        plt.xlim(-2, 2)  # Limit the x-axis (age) to -2 to 2
        plt.ylim(-2, 2)  # Limit the y-axis (BMI) to -2 to 2
        plt.show()

    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    app = HealthInsuranceAnalysisApp()
    app.run()'''

import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
import seaborn as sns
import tkinter as tk
from tkinter import messagebox

def perform_non_hierarchical_clustering():
    # Load the dataset
    csv_path = "/Users/i.seviantojensima/Desktop/Sem 4/pa/health insurance.csv"
    if not os.path.isfile(csv_path):
        messagebox.showerror("Error", "CSV file not found!")
        return

    data = pd.read_csv(csv_path)

    # Select columns for clustering
    columns_for_clustering = ['age', 'bmi']

    # Check if 'bmi' column exists
    if 'bmi' not in data.columns:
        messagebox.showerror("Error", "'bmi' column not found in the dataset. Clustering cannot be performed.")
        return

    # Handle missing values
    data[columns_for_clustering] = data[columns_for_clustering].fillna(data[columns_for_clustering].mean())

    # Standardize numerical features
    scaler = StandardScaler()
    data[columns_for_clustering] = scaler.fit_transform(data[columns_for_clustering])

    # Perform non-hierarchial clustering
    model = AgglomerativeClustering(n_clusters=3, linkage='ward')
    clusters = model.fit_predict(data[columns_for_clustering])

    # Add cluster labels to the dataset
    data['cluster'] = clusters

    # Map cluster labels to meaningful names
    cluster_mapping = {0: 'Underweight: BMI less than 18.5', 
                       1: 'Normal Weight: BMI between 18.5 and 24.9', 
                       2: 'Overweight: BMI between 25 and 49'}
    data['cluster'] = data['cluster'].map(cluster_mapping)

    # Visualize the clusters
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='age', y='bmi', hue='cluster', data=data, palette='viridis', legend='full')
    plt.title('Non-Hierarchical Clustering')
    plt.xlabel('Age')
    plt.ylabel('BMI')
    plt.xlim(-2, 2)  # Limit the x-axis (age) to -2 to 2
    plt.ylim(-2, 2)  # Limit the y-axis (BMI) to -2 to 2
    plt.show()

if __name__ == "__main__":
    perform_non_hierarchical_clustering()

