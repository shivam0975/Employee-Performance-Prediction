# 🧠 Employee Performance Prediction using Machine Learning

A machine learning-powered Flask web application to predict employee performance in a garment production setting based on various work-related and behavioral metrics.

---

## 📌 Project Overview

The **Employee Performance Prediction** system uses supervised machine learning algorithms to predict productivity scores for employees based on input parameters like team size, idle time, overtime, and more. This project aims to provide insights that assist in:

- 🎯 **Talent Retention**
- 📈 **Performance Enhancement**
- 🛠️ **Resource Allocation**

The system empowers HR teams and managers to make proactive decisions backed by data.

---

## 🚀 Project Objectives

By completing this project, you will:

- Understand core machine learning concepts and data pipelines.
- Gain experience with data pre-processing and feature engineering.
- Learn to train, evaluate, and deploy ML models.
- Build and integrate a Flask-based UI for real-time predictions.

---

## 💻 Tech Stack

| Area         | Technologies Used                              |
|--------------|------------------------------------------------|
| Language     | Python                                         |
| ML Libraries | Scikit-learn, XGBoost                          |
| Visualization| Matplotlib, Seaborn                           |
| Web Framework| Flask                                          |
| Frontend     | HTML, CSS                                      |

---

## 🧠 Use Cases

### 📌 Scenario 1: Talent Retention
Identify high-performing employees at risk of attrition and act proactively.

### 📌 Scenario 2: Performance Improvement
Detect employees needing training/support based on prediction trends.

### 📌 Scenario 3: Resource Allocation
Assign projects based on predicted productivity to optimize workforce deployment.

---

## 📦 Features

- 📝 Form-based UI for data input
- 📊 ML model integration for real-time prediction
- 🌐 Intuitive, responsive frontend using HTML/CSS
- 🧠 Supports multiple algorithms (e.g., Random Forest, XGBoost)

---

## 🛠️ Setup Instructions

### ✅ Prerequisites

- Python 3.8+
- pip
- Virtual environment (optional but recommended)

### ⚙️ Installation

```bash
# Clone the repository
git clone https://github.com/your-username/employee-performance-predictor.git
cd employee-performance-predictor

# (Optional) Create virtual environment
python -m venv venv
source venv/bin/activate     # Linux/macOS
venv\Scripts\activate        # Windows

# Install required packages
pip install -r requirements.txt

# Run the app
python app.py
```

## ✅ Input Parameters

| Feature                | Description                                              |
|------------------------|----------------------------------------------------------|
| `quarter`              | Fiscal quarter (Quarter1, Quarter2, etc.)               |
| `department`           | Department of work (e.g., sewing, finishing)            |
| `day`                  | Day of the week (e.g., Monday, Tuesday, etc.)           |
| `team`                 | Team number assigned to the task                        |
| `targeted_productivity`| Target productivity score for the task                  |
| `smv`                  | Standard Minute Value – estimated time for the task     |
| `over_time`            | Overtime work in minutes                                |
| `incentive`            | Incentive awarded for the task                          |
| `idle_time`            | Minutes the team was idle                               |
| `idle_men`             | Number of workers idle during the task                  |
| `no_of_style_change`   | Number of style changes during task execution           |
| `no_of_workers`        | Number of workers on the task                           |
| `month`                | Month in which the task was performed                   |

---

## ⚙️ ML Model Training Pipeline

1. **Data Loading**
   - Load the dataset using `pandas`.
   - Inspect and clean the data.

2. **Data Preprocessing**
   - Handle missing/null values.
   - Encode categorical features (`quarter`, `department`, `day`).
   - Normalize or scale numerical features as needed.

3. **Feature Engineering**
   - Extract relevant features.
   - Drop irrelevant columns.

4. **Train-Test Split**
   - Split the dataset using `train_test_split()` from Scikit-learn.

5. **Model Training**
   - Train models like `RandomForest`, `XGBoost`, and `DecisionTree`.
   - Tune hyperparameters for best performance.

6. **Model Evaluation**
   - Evaluate models using metrics like:
     - Mean Absolute Error (MAE)
     - Root Mean Square Error (RMSE)
     - R² Score

7. **Model Saving**
   - Save the best-performing model using `pickle` or `joblib`.

8. **Integration**
   - Integrate the trained model with the Flask app for predictions.

---

## 📚 Learning Resources

- [Google Machine Learning Crash Course](https://developers.google.com/machine-learning/crash-course)
- [IBM SmartBridge AI Training Course](https://skills.yourlearning.ibm.com/activity/PLAN-E624C2604060?ngo-id=0302&utm_campaign=aca-smartbridge-T250K-APSCHE-event#1)
- [Supervised Learning (Javatpoint)](https://www.javatpoint.com/supervised-machine-learning)
- [Unsupervised Learning (Javatpoint)](https://www.javatpoint.com/unsupervised-machine-learning)
- [Decision Trees](https://www.javatpoint.com/machine-learning-decision-tree-classification-algorithm)
- [Random Forests](https://www.javatpoint.com/machine-learning-random-forest-algorithm)
- [K-Nearest Neighbors (KNN)](https://www.javatpoint.com/k-nearest-neighbor-algorithm-for-machine-learning)
- [XGBoost Math & Intuition](https://www.analyticsvidhya.com/blog/2018/09/an-end-to-end-guide-to-understand-the-math-behind-xgboost/)
- [Model Evaluation Metrics](https://www.analyticsvidhya.com/blog/2019/08/11-important-model-evaluation-error-metrics/)
- [Flask Basics YouTube Tutorial](https://www.youtube.com/watch?v=lj4I_CvBnt0)

---

## 🙏 Special Thanks

A special thanks to:

- [SkillWallet by SmartInternz](https://skillwallet.smartinternz.com/)  
  For offering curated project-based learning opportunities and certifications in emerging tech fields.



