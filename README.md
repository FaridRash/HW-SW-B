# Interactive Student Performance Dashboard & Prediction

## 📌 Project Overview
This project is an end-to-end machine learning web application built with **Streamlit** to analyze and predict student academic performance. Using the **xAPI-Edu-Data** dataset, the app identifies key behavioral factors (such as class participation and resource visitation) that influence student grades. It combines unsupervised learning for student segmentation with supervised learning for grade prediction.

## 🛠 Tech Stack & Tools
* **Framework:** Streamlit (for web app interface)
* **Data Analysis:** Pandas, NumPy
* **Visualization:** Plotly, Seaborn, Matplotlib
* **Machine Learning:** Scikit-learn (Decision Tree, Random Forest, K-Means, PCA)
* **Statistical Analysis:** Statsmodels (VIF for feature selection)

## ⚙️ Key Features & Methodology
1.  **Interactive Data Exploration (EDA):**
    * Real-time visualization of feature distributions (Histograms, Boxplots) and correlations.
    * Automated handling of missing values and outlier detection/capping.
2.  **Feature Engineering:**
    * Applied **Variance Inflation Factor (VIF)** analysis to remove multicollinear features.
    * Encoded categorical variables using One-Hot and Label Encoding.
3.  **Unsupervised Learning (Clustering):**
    * Implemented **K-Means Clustering** to group students based on behavioral patterns.
    * Utilized **PCA (Principal Component Analysis)** for dimensionality reduction and 2D visualization of clusters.
4.  **Predictive Modeling:**
    * Trained **Decision Tree** and **Random Forest** classifiers to predict student grades (Class L, M, H).
    * Evaluated models using Accuracy, Classification Reports, and Confusion Matrices.

## 🚀 How to Run
```bash
pip install -r requirements.txt
streamlit run app.py
