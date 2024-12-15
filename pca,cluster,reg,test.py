import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk

from sklearn.metrics import accuracy_score, r2_score

from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.feature_selection import f_oneway
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt
import cv2
import subprocess
import numpy as np
import os  

class HealthInsuranceAnalysisApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Health Insurance Data Analysis")
        self.root.geometry("800x600")  # height for video
        self.root.configure(background='white')  # background color

        self.current_test = None  # Initialize current test variable

        # Initialize the video player
        self.video_source = "/Users/i.seviantojensima/Desktop/Jensi/Sem 4/pa/front.mp4"
        self.video_player = VideoPlayerApp(self.root, self.video_source)
        self.video_player.pack(fill=tk.BOTH, expand=True)

        # Add buttons for data analysis
        self.regression_button = tk.Button(self.root, text="Regression", command=self.open_regression_options)
        self.regression_button.place(relx=0.5, rely=0.4, anchor=tk.CENTER)

        self.test_button = tk.Button(self.root, text="Test", command=self.open_test_options)
        self.test_button.place(relx=0.5, rely=0.5, anchor=tk.CENTER)

        self.pca_fa_button = tk.Button(self.root, text="PCA and Factor Analysis", command=self.open_pca_fa_options)
        self.pca_fa_button.place(relx=0.5, rely=0.6, anchor=tk.CENTER)

        # Add button for clustering
        self.clustering_button = tk.Button(self.root, text="Clustering", command=self.perform_clustering)
        self.clustering_button.place(relx=0.5, rely=0.7, anchor=tk.CENTER)

    def perform_clustering(self):
        # Load the dataset
        csv_path = "/Users/i.seviantojensima/Desktop/Jensi/Sem 4/pa/health insurance.csv"  
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

        # Encode categorical variables
        label_encoder = LabelEncoder()
        data['sex'] = label_encoder.fit_transform(data['sex'])
        data['smoker'] = label_encoder.fit_transform(data['smoker'])
        data['region'] = label_encoder.fit_transform(data['region'])

        # Standardize numerical features
        scaler = StandardScaler()
        data[columns_for_clustering] = scaler.fit_transform(data[columns_for_clustering])

        # Compute the linkage matrix
        Z = linkage(data[columns_for_clustering], method='ward')

        # Plot the dendrogram
        plt.figure(figsize=(10, 6))
        dendrogram(Z)
        plt.title('Hierarchical Clustering Dendrogram')
        plt.xlabel('Samples')
        plt.ylabel('Distance')
        plt.show()


    def open_regression_options(self):
        regression_window = tk.Toplevel(self.root)
        regression_window.title("Regression Options")
        regression_window.attributes('-fullscreen', True)

        # Load and display image as background
        regression_image_path = "/Users/i.seviantojensima/Desktop/Jensi/Sem 4/pa/regression f.jpg"
        regression_image = Image.open(regression_image_path)
        regression_image = regression_image.resize((self.root.winfo_screenwidth(), self.root.winfo_screenheight()))
        regression_photo = ImageTk.PhotoImage(regression_image)

        # Create a label to display image
        regression_image_label = tk.Label(regression_window, image=regression_photo)
        regression_image_label.image = regression_photo
        regression_image_label.place(relx=0.5, rely=0.5, anchor=tk.CENTER)  # Center image

        # Add buttons for regression options
        logistic_regression_button = tk.Button(regression_window, text="Logistic Regression", command=self.logistic_regression_clicked)
        logistic_regression_button.place(relx=0.5, rely=0.5, anchor=tk.CENTER)

        linear_regression_button = tk.Button(regression_window, text="Multiple Linear Regression", command=self.linear_regression_clicked)
        linear_regression_button.place(relx=0.5, rely=0.6, anchor=tk.CENTER)

    def open_test_options(self):
        test_window = tk.Toplevel(self.root)
        test_window.title("Test Options")
        test_window.attributes('-fullscreen', True)

        # Load and display image as background
        test_image_path = "/Users/i.seviantojensima/Desktop/Jensi/Sem 4/pa/test.jpg"
        test_image = Image.open(test_image_path)
        test_image = test_image.resize((self.root.winfo_screenwidth(), self.root.winfo_screenheight()))
        test_photo = ImageTk.PhotoImage(test_image)

        # Create a label to display image
        test_image_label = tk.Label(test_window, image=test_photo)
        test_image_label.image = test_photo
        test_image_label.place(relx=0.5, rely=0.5, anchor=tk.CENTER)

        # Add buttons for test options
        ftest_button = tk.Button(test_window, text="Perform F-test", command=self.ftest_clicked)
        ftest_button.place(relx=0.5, rely=0.5, anchor=tk.CENTER)

        anova_button = tk.Button(test_window, text="Perform ANOVA", command=self.anova_clicked)
        anova_button.place(relx=0.5, rely=0.6, anchor=tk.CENTER)

    def open_pca_fa_options(self):
        pca_fa_window = tk.Toplevel(self.root)
        pca_fa_window.title("PCA and Factor Analysis Options")
        pca_fa_window.attributes('-fullscreen', True)

        # Load and display image as background
        pca_fa_image_path = "/Users/i.seviantojensima/Desktop/Jensi/Sem 4/pa/pca.jpg"
        pca_fa_image = Image.open(pca_fa_image_path)
        
        pca_fa_photo = ImageTk.PhotoImage(pca_fa_image)

        # Create a label to display image
        pca_fa_image_label = tk.Label(pca_fa_window, image=pca_fa_photo)
        pca_fa_image_label.image = pca_fa_photo
        pca_fa_image_label.place(relx=0.5, rely=0.5, anchor=tk.CENTER)  # Center image

        # Add buttons for PCA and FA options
        pca_button = tk.Button(pca_fa_window, text="Perform PCA", command=self.perform_pca)
        pca_button.place(relx=0.5, rely=0.4, anchor=tk.CENTER)

        fa_button = tk.Button(pca_fa_window, text="Perform Factor Analysis", command=self.perform_fa)
        fa_button.place(relx=0.5, rely=0.5, anchor=tk.CENTER)

        comparison_button = tk.Button(pca_fa_window, text="Show Comparison", command=self.show_comparison)
        comparison_button.place(relx=0.5, rely=0.6, anchor=tk.CENTER)

    def read_csv_data(self, csv_path):
        try:
            df = pd.read_csv(csv_path)
            return df
        except FileNotFoundError:
            messagebox.showerror("Error", "CSV file not found!")
            return None

    def perform_logistic_regression(self):
        csv_path = "/Users/i.seviantojensima/Desktop/Jensi/Sem 4/pa/health insurance.csv"
        data = self.read_csv_data(csv_path)
        if data is None:
            return

        # Convert categorical variables into dummy variables
        data = pd.get_dummies(data, columns=['sex', 'smoker', 'region'], drop_first=True)

        # Split data into features (X) and target variable (y)
        X = data.drop(columns=['charges'])
        y = data['charges']

        # Convert charges into categories for logistic regression
        discretizer = KBinsDiscretizer(n_bins=2, encode='ordinal', strategy='uniform')
        y_discrete = discretizer.fit_transform(y.values.reshape(-1, 1)).flatten()
        X_train_logistic, X_test_logistic, y_train_logistic, y_test_logistic = train_test_split(X, y_discrete, test_size=0.2, random_state=42)

        logistic_model = LogisticRegression(max_iter=1000)
        logistic_model.fit(X_train_logistic, y_train_logistic)
        y_pred_logistic = logistic_model.predict(X_test_logistic)
        accuracy_logistic = accuracy_score(y_test_logistic, y_pred_logistic)

        # Compute summary and inference
        summary = f"Accuracy (Logistic Regression): {accuracy_logistic}\n"
        if accuracy_logistic >= 0.8:
            inference = "The logistic regression model has high accuracy, indicating a good fit to the data."
        elif accuracy_logistic >= 0.6:
            inference = "The logistic regression model has moderate accuracy."
        else:
            inference = "The logistic regression model has low accuracy, indicating poor performance."

        messagebox.showinfo("Logistic Regression Result", summary + inference)

        # Plot logistic regression results
        plt.scatter(range(len(y_test_logistic)), y_test_logistic, color='black', label='Actual')
        plt.plot(range(len(y_test_logistic)), y_pred_logistic, color='red', linewidth=2, label='Predicted')
        plt.xlabel('Sample Index')
        plt.ylabel('Charge')
        plt.title('Logistic Regression Results')
        plt.legend()
        plt.show()

    def perform_linear_regression(self):
        csv_path = "/Users/i.seviantojensima/Desktop/Jensi/Sem 4/pa/health insurance.csv"
        data = self.read_csv_data(csv_path)
        if data is None:
            return

        # Convert categorical variables into dummy variables
        data = pd.get_dummies(data, columns=['sex', 'smoker', 'region'], drop_first=True)

        # Split data into features (X) and target variable (y)
        X = data.drop(columns=['charges'])
        y = data['charges']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        mlr_model = LinearRegression()
        mlr_model.fit(X_train, y_train)
        y_pred = mlr_model.predict(X_test)
        rmse = mean_squared_error(y_test, y_pred, squared=False)  # Note the use of squared=False to get RMSE
        r2 = r2_score(y_test, y_pred)

        # Compute summary and inference
        summary = f"RMSE: {rmse}\nR-squared: {r2}\n"
        if r2 >= 0.8:
            inference = "The multiple linear regression model has high R-squared, indicating a good fit to the data."
        elif r2 >= 0.6:
            inference = "The multiple linear regression model has moderate R-squared."
        else:
            inference = "The multiple linear regression model has low R-squared, indicating poor performance."

        messagebox.showinfo("Multiple Linear Regression Result", summary + inference)

        # Plot regression results
        plt.scatter(range(len(y_test)), y_test, color='black', label='Actual')
        plt.plot(range(len(y_test)), y_pred, color='blue', linewidth=2, label='Predicted')
        plt.xlabel('Sample Index')
        plt.ylabel('Charge')
        plt.title('Multiple Linear Regression Results')
        plt.legend()
        plt.show()

    def perform_pca(self):
        csv_path = "/Users/i.seviantojensima/Desktop/Jensi/Sem 4/pa/health insurance.csv"
        data = self.read_csv_data(csv_path)
        if data is None:
            return

        # Convert categorical variables into dummy variables
        data = pd.get_dummies(data, columns=['sex', 'smoker', 'region'], drop_first=True)

        # Standardize the data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(data)

        # Apply PCA
        pca = PCA(n_components=2)
        principal_components = pca.fit_transform(X_scaled)

        # Plot PCA results
        plt.scatter(principal_components[:, 0], principal_components[:, 1])
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.title('PCA Results')
        plt.show()

    def perform_fa(self):
        csv_path = "/Users/i.seviantojensima/Desktop/Jensi/Sem 4/pa/health insurance.csv"
        data = self.read_csv_data(csv_path)
        if data is None:
            return

        # Convert categorical variables into dummy variables
        data = pd.get_dummies(data, columns=['sex', 'smoker', 'region'], drop_first=True)

        # Standardize the data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(data)

        # Apply Factor Analysis
        fa = FactorAnalysis(n_components=2)
        factors = fa.fit_transform(X_scaled)

        # Plot Factor Analysis results
        plt.scatter(factors[:, 0], factors[:, 1])
        plt.xlabel('Factor 1')
        plt.ylabel('Factor 2')
        plt.title('Factor Analysis Results')
        plt.show()

    def show_comparison(self):
        csv_path = "/Users/i.seviantojensima/Desktop/Jensi/Sem 4/pa/health insurance.csv"
        data = self.read_csv_data(csv_path)
        if data is None:
            return

        # Convert categorical variables into dummy variables
        data = pd.get_dummies(data, columns=['sex', 'smoker', 'region'], drop_first=True)

        # Standardize the data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(data)

        # Apply PCA and Factor Analysis
        pca = PCA(n_components=2)
        fa = FactorAnalysis(n_components=2)
        principal_components = pca.fit_transform(X_scaled)
        factors = fa.fit_transform(X_scaled)

        # Plot PCA results
        plt.scatter(principal_components[:, 0], principal_components[:, 1], label='PCA', alpha=0.5)
        # Plot Factor Analysis results
        plt.scatter(factors[:, 0], factors[:, 1], label='Factor Analysis', alpha=0.5)
        plt.xlabel('Component/Factor 1')
        plt.ylabel('Component/Factor 2')
        plt.title('PCA and Factor Analysis Comparison')
        plt.legend()
        plt.show()

    def logistic_regression_clicked(self):
        self.perform_logistic_regression()

    def linear_regression_clicked(self):
        self.perform_linear_regression()

    def ftest_clicked(self):
        csv_path = "/Users/i.seviantojensima/Desktop/Jensi/Sem 4/pa/health insurance.csv"
        data = self.read_csv_data(csv_path)
        if data is None:
            return

        f_stat, p_value = f_oneway(data['age'], data['bmi'], data['children'], data['charges'])
        result = f"F-statistic: {f_stat}\nP-value: {p_value}"
        messagebox.showinfo("F-test Result", result)

    def anova_clicked(self):
        csv_path = "/Users/i.seviantojensima/Desktop/Jensi/Sem 4/pa/health insurance.csv"
        data = self.read_csv_data(csv_path)
        if data is None:
            return

        f_stat, p_value = f_oneway(data['age'], data['bmi'], data['children'], data['charges'])
        result = f"F-statistic: {f_stat}\nP-value: {p_value}"
        messagebox.showinfo("ANOVA Result", result)

    def run(self):
        self.root.mainloop()

class VideoPlayerApp(tk.Frame):
    def __init__(self, parent, video_source):
        super().__init__(parent)
        self.parent = parent
        self.video_source = video_source
        self.cap = cv2.VideoCapture(video_source)
        self.width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

        self.canvas = tk.Canvas(self, width=self.width, height=self.height, bg='black')
        self.canvas.pack()

        self.delay = 15
        self.update_frame()

    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            self.photo = self.convert_frame_to_image(frame)
            self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)
        self.after(self.delay, self.update_frame)

    def convert_frame_to_image(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame)
        return ImageTk.PhotoImage(image)

if __name__ == "__main__":
    app = HealthInsuranceAnalysisApp()
    app.run()
