# 🍄 Mushroom Classification using Gradient Boosting

This repository contains a Machine Learning project that uses the
**Gradient Boosting Classifier** to predict whether a mushroom is
**edible or poisonous** based on its physical characteristics.

---

## 📌 Project Overview
Mushroom classification is a classic supervised learning problem.
In this project, Gradient Boosting is applied to improve prediction
accuracy by combining multiple weak learners into a strong ensemble model.

The trained model is saved and used for prediction through a Python application.

---

## 📂 Dataset
- Mushroom dataset with categorical features
- Target variable: **Edible / Poisonous**
- Dataset is preprocessed using encoding techniques

---

## ⚙️ Methodology
1. Data loading and preprocessing  
2. Encoding categorical variables  
3. Train-test split  
4. Model training using Gradient Boosting  
5. Model evaluation  
6. Model serialization for reuse  

---

## 🧠 Algorithm Used
- **Gradient Boosting Classifier**

---

## 📊 Results
The Gradient Boosting model achieved high classification accuracy
in distinguishing edible and poisonous mushrooms.
The ensemble approach helped reduce bias and variance,
resulting in reliable predictions.

---

## ▶️ How to Run

1. Clone the repository  
2. Navigate to the project folder  
3. Install required libraries:
   ```
   pip install -r requirements.txt
   ```
4. Run the project:
   ```
   python gradient_boosting_mushroom.py
   ```

---

## 🛠 Technologies Used
- Python  
- NumPy  
- Pandas  
- Scikit-learn  
- Pickle  
- Jupyter Notebook  

---

## 📁 Repository Structure
