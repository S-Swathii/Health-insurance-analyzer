import tkinter as tk
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
        self.root.geometry("800x600")  #height for video
        self.root.configure(background='white')    #background color

        self.current_test = None  # Initialize current test variable

        # Initialize the video player
        self.video_source = "/Users/i.seviantojensima/Desktop/Sem 4/pa/front.mp4"  
        self.video_player = VideoPlayerApp(self.root, self.video_source)
        self.video_player.pack(fill=tk.BOTH, expand=True)

        # Add buttons for data analysis
        self.regression_button = tk.Button(self.root, text="Regression", command=self.open_regression_options)
        self.regression_button.place(relx=0.5, rely=0.5, anchor=tk.CENTER)

        self.test_button = tk.Button(self.root, text="Test", command=self.open_test_options)
        self.test_button.place(relx=0.5, rely=0.6, anchor=tk.CENTER)


    def open_regression_options(self):
        regression_window = tk.Toplevel(self.root)
        regression_window.title("Regression Options")
        regression_window.attributes('-fullscreen', True)

        # Load and display image as background
        regression_image_path = "/Users/i.seviantojensima/Desktop/regression f.jpg"  
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
        test_image_path = "/Users/i.seviantojensima/Desktop/test.jpg"  
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
        
        # Compute summary and inference
        summary = f"Mean Squared Error (Multiple Linear Regression): {mse_mlr}\nR-squared (Multiple Linear Regression): {r2_mlr}\n"
        inference = ""
        if r2_mlr >= 0.7:
            inference = "The multiple linear regression model explains a substantial amount of the variance in the target variable."
        elif r2_mlr >= 0.4:
            inference = "The multiple linear regression model explains a moderate amount of the variance in the target variable."
        else:
            inference = "The multiple linear regression model explains a low amount of the variance in the target variable."
        
        messagebox.showinfo("Multiple Linear Regression Result", summary + inference)

        # Plot linear regression results
        plt.scatter(range(len(y_test)), y_test, color='black', label='Actual')
        plt.plot(range(len(y_test)), y_pred_mlr, color='red', linewidth=2, label='Predicted')
        plt.xlabel('Sample Index')
        plt.ylabel('Charge')
        plt.title('Multiple Linear Regression Results')
        plt.legend()
        plt.show()

    def perform_anova(self):
        csv_path = '/Users/i.seviantojensima/Desktop/Sem 4/pa/health insurance.csv'
        data = self.read_csv_data(csv_path)
        if data is None:
            return

        anova_result = f_oneway(data['charges'], data['age'])
        
        # Compute summary and inference
        summary = f"ANOVA p-value for charges and age: {anova_result.pvalue:.4f}\n"
        if anova_result.pvalue < 0.05:
            inference = "There is a statistically significant difference between charges and age."
        else:
            inference = "There is no statistically significant difference between charges and age."
        
        messagebox.showinfo("ANOVA Result", summary + inference)

        # Plot ANOVA results
        plt.boxplot([data['charges'], data['age']], labels=['Charges', 'Age'])
        plt.ylabel('Value')
        plt.title('ANOVA Results')
        plt.show()

    def perform_ftest(self):
        csv_path = '/Users/i.seviantojensima/Desktop/Sem 4/pa/health insurance.csv'
        data = self.read_csv_data(csv_path)
        if data is None:
            return

        males = data[data['sex'] == 'male']['charges']
        females = data[data['sex'] == 'female']['charges']
        f_statistic, f_p_value = f_oneway(males, females)
        
        # Compute summary and inference
        summary = f"F-test p-value for charges between male and female: {f_p_value:.4f}\n"
        if f_p_value < 0.05:
            inference = "There is a statistically significant difference in charges between males and females."
        else:
            inference = "There is no statistically significant difference in charges between males and females."
        
        messagebox.showinfo("F-Test Result", summary + inference)

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

    def ftest_clicked(self):
        self.perform_ftest()

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

def main():
    app = HealthInsuranceAnalysisApp()
    app.run()

if __name__ == "__main__":
    main()
