'''import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, accuracy_score, r2_score
from sklearn.preprocessing import KBinsDiscretizer
from scipy.stats import f_oneway, ttest_ind
import matplotlib.pyplot as plt
import cv2

class HealthInsuranceAnalysisApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Health Insurance Data Analysis")
        self.root.geometry("800x600")  # Increased height for video
        self.root.configure(background='white')    # Set background color to white

        # Create a frame for the video and buttons
        self.video_frame = tk.Frame(self.root, bg='white')
        self.video_frame.pack(fill=tk.BOTH, expand=True)

        # Initialize the video player
        self.video_source = "/Users/i.seviantojensima/Desktop/pa.mp4"  
        self.video_player = VideoPlayerApp(self.video_frame, self.video_source)
        self.video_player.pack(side=tk.LEFT, padx=10, pady=10, fill=tk.BOTH, expand=True)

        # Create a frame for buttons
        self.button_frame = tk.Frame(self.video_frame, bg='white')
        self.button_frame.pack(side=tk.RIGHT, padx=10, pady=10)

        # Add buttons for data analysis
        self.regression_button = tk.Button(self.button_frame, text="Regression", command=self.open_regression_options)
        self.regression_button.pack(pady=5)

        self.test_button = tk.Button(self.button_frame, text="Test", command=self.open_test_options)
        self.test_button.pack(pady=5)

        self.current_test = None  # Initialize current test variable

        # Center the button frame vertically and horizontally
        self.button_frame.place(relx=0.5, rely=0.5, anchor=tk.CENTER)

    def open_regression_options(self):
        regression_window = tk.Toplevel(self.root)
        regression_window.title("Regression Options")
        regression_window.geometry("200x150")

        self.logistic_regression_button = tk.Button(regression_window, text="Logistic Regression", command=self.logistic_regression_clicked)
        self.logistic_regression_button.pack(pady=5)

        self.linear_regression_button = tk.Button(regression_window, text="Linear Regression", command=self.linear_regression_clicked)
        self.linear_regression_button.pack(pady=5)

    def open_test_options(self):
        test_window = tk.Toplevel(self.root)
        test_window.title("Test Options")
        test_window.geometry("200x150")

        self.anova_button = tk.Button(test_window, text="Perform ANOVA", command=self.anova_clicked)
        self.anova_button.pack(pady=5)

        self.ttest_button = tk.Button(test_window, text="Perform T-test", command=self.ttest_clicked)
        self.ttest_button.pack(pady=5)

        self.ftest_button = tk.Button(test_window, text="Perform F-test", command=self.ftest_clicked)
        self.ftest_button.pack(pady=5)

    def read_csv_data(self, csv_path):
        try:
            df = pd.read_csv(csv_path)
            return df
        except FileNotFoundError:
            messagebox.showerror("Error", "CSV file not found!")
            return None

    def perform_logistic_regression(self):
        csv_path = '/Users/i.seviantojensima/Desktop/Sem 4/pa/health insurance.csv'
        data = self.read_csv_data(csv_path)
        if data is None:
            return

        # Convert categorical variables into dummy variables
        data = pd.get_dummies(data, columns=['sex', 'smoker', 'region'], drop_first=True)

        # Split data into features (X) and target variable (y)
        X = data.drop(columns=['charges'])
        y = data['charges']

        # Convert charges into categories for logistic regression
        discretizer = KBinsDiscretizer(n_bins=2, encode='ordinal', strategy='uniform', subsample=None)
        y_discrete = discretizer.fit_transform(y.values.reshape(-1, 1)).flatten()
        X_train_logistic, X_test_logistic, y_train_logistic, y_test_logistic = train_test_split(X, y_discrete, test_size=0.2, random_state=42)

        logistic_model = LogisticRegression(max_iter=1000)
        logistic_model.fit(X_train_logistic, y_train_logistic)
        y_pred_logistic = logistic_model.predict(X_test_logistic)
        accuracy_logistic = accuracy_score(y_test_logistic, y_pred_logistic)
        messagebox.showinfo("Logistic Regression Result", f"Accuracy (Logistic Regression): {accuracy_logistic}")

        # Plot logistic regression results
        plt.scatter(range(len(y_test_logistic)), y_test_logistic, color='black', label='Actual')
        plt.plot(range(len(y_test_logistic)), y_pred_logistic, color='red', linewidth=2, label='Predicted')
        plt.xlabel('Sample Index')
        plt.ylabel('Charge')
        plt.title('Logistic Regression Results')
        plt.legend()
        plt.show()

    def perform_linear_regression(self):
        csv_path = '/Users/i.seviantojensima/Desktop/Sem 4/pa/health insurance.csv'
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
        y_pred_mlr = mlr_model.predict(X_test)
        mse_mlr = mean_squared_error(y_test, y_pred_mlr)
        r2_mlr = r2_score(y_test, y_pred_mlr)
        messagebox.showinfo("Multiple Linear Regression Result", f"Mean Squared Error (Multiple Linear Regression): {mse_mlr}\nR-squared (Multiple Linear Regression): {r2_mlr}")

        # Plot linear regression results
        plt.scatter(range(len(y_test)), y_test, color='black', label='Actual')
        plt.plot(range(len(y_test)), y_pred_mlr, color='red', linewidth=2, label='Predicted')
        plt.xlabel('Sample Index')
        plt.ylabel('Charge')
        plt.title('Linear Regression Results')
        plt.legend()
        plt.show()

    def perform_anova(self):
        csv_path = '/Users/i.seviantojensima/Desktop/Sem 4/pa/health insurance.csv'
        data = self.read_csv_data(csv_path)
        if data is None:
            return

        anova_result = f_oneway(data['charges'], data['age'])
        messagebox.showinfo("ANOVA Result", f"ANOVA p-value for charges and age: {anova_result.pvalue:.4f}")

        # Plot ANOVA results
        plt.boxplot([data['charges'], data['age']], labels=['Charges', 'Age'])
        plt.ylabel('Value')
        plt.title('ANOVA Results')
        plt.show()

    def perform_ttest(self):
        csv_path = '/Users/i.seviantojensima/Desktop/Sem 4/pa/health insurance.csv'
        data = self.read_csv_data(csv_path)
        if data is None:
            return

        smokers = data[data['smoker'] == 'yes']['charges']
        non_smokers = data[data['smoker'] == 'no']['charges']
        ttest_result = ttest_ind(smokers, non_smokers)
        messagebox.showinfo("T-Test Result", f"T-test p-value for charges between smokers and non-smokers: {ttest_result.pvalue:.4f}")

        # Plot T-test results
        plt.boxplot([smokers, non_smokers], labels=['Smokers', 'Non-Smokers'])
        plt.ylabel('Charge')
        plt.title('T-Test Results')
        plt.show()

    def perform_ftest(self):
        csv_path = '/Users/i.seviantojensima/Desktop/Sem 4/pa/health insurance.csv'
        data = self.read_csv_data(csv_path)
        if data is None:
            return

        males = data[data['sex'] == 'male']['charges']
        females = data[data['sex'] == 'female']['charges']
        f_statistic, f_p_value = f_oneway(males, females)
        messagebox.showinfo("F-Test Result", f"F-test p-value for charges between male and female: {f_p_value:.4f}")

        # Plot F-test results
        plt.boxplot([males, females], labels=['Males', 'Females'])
        plt.ylabel('Charge')
        plt.title('F-Test Results')
        plt.show()

    def logistic_regression_clicked(self):
        self.perform_logistic_regression()

    def linear_regression_clicked(self):
        self.perform_linear_regression()

    def anova_clicked(self):
        self.perform_anova()

    def ttest_clicked(self):
        self.perform_ttest()

    def ftest_clicked(self):
        self.perform_ftest()

    def run(self):
        self.root.mainloop()

class VideoPlayerApp(tk.Frame):
    def __init__(self, master, video_source):
        tk.Frame.__init__(self, master)
        self.master = master
        self.video_source = video_source

        # Initialize video player
        self.video = cv2.VideoCapture(self.video_source)
        self.width = self.master.winfo_screenwidth()  # Get screen width
        self.height = self.master.winfo_screenheight()  # Get screen height

        self.canvas = tk.Canvas(self.master, width=self.width, height=self.height)
        self.canvas.pack(fill=tk.BOTH, expand=True)

        self.delay = 15
        self.update()

    def update(self):
        ret, frame = self.video.read()
        if ret:
            frame = cv2.resize(frame, (self.width, self.height))
            self.photo = self.convert_to_tkimage(frame)
            self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)
        self.master.after(self.delay, self.update)

    def convert_to_tkimage(self, frame):
        return ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))

def main():
    app = HealthInsuranceAnalysisApp()
    app.run()

if __name__ == "__main__":
    main()


import pandas as pd
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import cv2
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import os.path

class HealthInsuranceAnalysisApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Health Insurance Data Analysis")
        self.root.geometry("800x600")
        self.root.configure(background='white')

        self.current_test = None

        # Initialize the video player
        self.video_source = "/Users/i.seviantojensima/Desktop/Sem 4/pa/front.mp4"
        self.video_player = VideoPlayerApp(self.root, self.video_source)
        self.video_player.pack(fill=tk.BOTH, expand=True)

        # Add buttons for data analysis
        self.regression_button = tk.Button(self.root, text="Regression", command=self.open_regression_options)
        self.regression_button.place(relx=0.5, rely=0.5, anchor=tk.CENTER)

        self.test_button = tk.Button(self.root, text="Test", command=self.open_test_options)
        self.test_button.place(relx=0.5, rely=0.6, anchor=tk.CENTER)

        self.cluster_button = tk.Button(self.root, text="Perform Clustering", command=self.perform_clustering)
        self.cluster_button.place(relx=0.5, rely=0.7, anchor=tk.CENTER)

    def open_regression_options(self):
        # Implement this method if needed
        pass

    def open_test_options(self):
        # Implement this method if needed
        pass

    
    
    def perform_clustering(self):
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

        # Encode categorical variables
        label_encoder = LabelEncoder()
        data['sex'] = label_encoder.fit_transform(data['sex'])
        data['smoker'] = label_encoder.fit_transform(data['smoker'])
        data['region'] = label_encoder.fit_transform(data['region'])

        # Standardize numerical features
        scaler = StandardScaler()
        data[columns_for_clustering] = scaler.fit_transform(data[columns_for_clustering])

        # Perform hierarchical clustering
        model = AgglomerativeClustering(n_clusters=3, linkage='ward')
        clusters = model.fit_predict(data[columns_for_clustering])

        # Add cluster labels to the dataset
        data['cluster'] = clusters

        # Map cluster labels to meaningful names
        data['cluster'] = data['cluster'].map({-1: 'Obesity: BMI 50 or greater', 
                                               0: 'Underweight: BMI less than 18.5', 
                                               1: 'Normal Weight: BMI between 18.5 and 24.9', 
                                               2: 'Overweight: BMI between 25 and 49'})

        # Visualize the clusters
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x='age', y='bmi', hue='cluster', data=data, palette='viridis', legend='full')
        plt.title('Non-Hierarchical Clustering')
        plt.xlabel('Age')
        plt.ylabel('BMI')
        plt.show()

    def run(self):
        self.root.mainloop()


class VideoPlayerApp(tk.Frame):
    def __init__(self, master, video_source):
        tk.Frame.__init__(self, master)
        self.master = master
        self.video_source = video_source

        # video player
        self.video = cv2.VideoCapture(self.video_source)
        self.width = self.master.winfo_screenwidth()
        self.height = self.master.winfo_screenheight()

        self.canvas = tk.Canvas(self.master, width=self.width, height=self.height)
        self.canvas.pack(fill=tk.BOTH, expand=True)

        self.delay = 15
        self.update()

    def update(self):
        ret, frame = self.video.read()
        if ret:
            frame = cv2.resize(frame, (self.width, self.height))
            self.photo = self.convert_to_tkimage(frame)
            self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)
        self.master.after(self.delay, self.update)


    def convert_to_tkimage(self, frame):
        return ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))

if __name__ == "__main__":
    app = HealthInsuranceAnalysisApp()
    app.run()'''

import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from scipy.cluster.hierarchy import dendrogram, linkage
import tkinter.messagebox as messagebox

def perform_clustering():
    # Load the dataset
    csv_path = "health insurance.csv"  # Update the file path
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

def run():
    perform_clustering()


run()


def run(self):
    self.root.mainloop()

