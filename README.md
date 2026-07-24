# 🚗 Saudi Used Car Price Prediction (Syarah Dataset)

## 📌 Project Overview
This project focuses on conducting an in-depth **Exploratory Data Analysis (EDA)** and building a **Machine Learning Regression Model** to predict used car prices in Saudi Arabia. The dataset is scraped from the popular local Saudi platform, **Syarah**. The project covers the full end-to-end pipeline: from data cleaning and exploration to predictive modeling.

Developed as a project for the Data Science and AI Bootcamp at **Tuwaiq Academy**.

---

## 📊 Dataset & Limitations
* **Source:** Real-world used car listings from the **Syarah** app (scraped dataset available on Kaggle).
* **Limitations:** The dataset does not include an accident history column (whether the car has been in previous accidents), which is a key pricing factor. The model relies on available attributes like manufacturing year, mileage, engine size, brand, and trim to achieve optimal prediction accuracy.

---

## 🔍 Exploratory Data Analysis & Key Insights
During the EDA phase, the project explores the following key market questions:

* **Popular Used Cars:** Which car models are the most listed and in-demand in Saudi Arabia?
* **Top 5 Brands:** What are the top 5 leading automotive manufacturers in the used car market?
* **Vehicle Origins:** What percentage of the market is represented by vehicle origins (e.g., GCC/Saudi specs, US, Japanese, European)?
* **Color Preferences:** What is the most popular vehicle exterior color?
* **Engines & Trims:** What are the most common engine sizes and trim choices (e.g., Standard, Semi-Full, Full Option)?

---

## 🤖 Machine Learning Objectives
After uncovering market insights, the predictive modeling phase addresses:

* **Feature Importance:** Which features have the strongest impact on car pricing (e.g., Year, Mileage, Engine Size, Brand, Region)?
* **Algorithm Evaluation:** Which machine learning regression algorithm provides the best prediction performance based on metrics like $R^2$, $MAE$, and $RMSE$?

---

## 🛠️ Project Workflow

### 1. Data Preprocessing & Cleaning
* Handled missing values, zero-value prices (e.g., negotiable listings), and duplicate records.
* Cleaned and standardized categorical attributes (Brand, Model, Color, Origin, Region).
* Encoded categorical features and scaled numerical variables for model readiness.

### 2. Exploratory Data Analysis & Visualization
* Created charts and plots using `Matplotlib` and `Seaborn` to answer all research questions and visualize feature distributions.

### 3. Machine Learning & Modeling
* Split the dataset into training and testing sets.
* Trained and evaluated multiple regression algorithms (e.g., Linear Regression, Random Forest Regressor).
* Compared model performance using evaluation metrics ($R^2$, $MAE$, $RMSE$) to select the best predictor.

---

## 💻 Tech Stack & Tools
* **Programming Language:** Python
* **Data Processing & Analysis:** Pandas, NumPy
* **Data Visualization:** Matplotlib, Seaborn
* **Machine Learning:** Scikit-Learn
* **Environment:** Google Colab / Jupyter Notebook
