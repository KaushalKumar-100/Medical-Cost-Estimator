# 🛡️ Insurance Cost Predictor

> **Predict your annual insurance premium instantly using Machine Learning.**
> Built with **Streamlit**, **Scikit-learn**, and **Random Forest Regression**.

---

## 🚀 Live Demo

🔗 **Web App:**
```text
https://medical-cost-estimator-100.streamlit.app/
```

---


# ✨ Features

* 🤖 AI-powered insurance premium prediction
* 📊 Interactive Streamlit dashboard
* 🌍 Real-time currency conversion using Exchange Rate API
* 💰 Convert predicted premium into multiple currencies
* ⚡ Fast predictions with cached machine learning models
* 🎨 Modern responsive user interface with custom HTML & CSS
* 🔄 End-to-end Machine Learning Pipeline

---

# 🧠 Machine Learning Pipeline

This project follows a complete end-to-end Machine Learning workflow.

## 📌 Data Preprocessing

* Handling missing values
* Feature engineering
* Categorical encoding
* Numerical preprocessing
* **ColumnTransformer**
* **Scikit-learn Pipeline**

---

## 🤖 Models Evaluated

Several regression algorithms were trained and evaluated before selecting the final model.

| Model                     | Status       |
| ------------------------- | ------------ |
| ✅ Linear Regression       | Evaluated    |
| ✅ Ridge Regression        | Evaluated    |
| ✅ Lasso Regression        | Evaluated    |
| ✅ Decision Tree Regressor | Evaluated    |
| ⭐ Random Forest Regressor | **Selected** |
| ✅ XGBoost Regressor       | Evaluated    |

---

## 📈 Model Evaluation

To compare model performance, the following metrics were used:

* Root Mean Squared Error (RMSE)
* Mean Absolute Error (MAE)
* R² Score

---

## 🔍 Model Selection Strategy

The best model was selected using:

* ✅ K-Fold Cross Validation
* ✅ GridSearchCV
* ✅ Hyperparameter Tuning
* ✅ Performance Comparison Across Multiple Models

---

## 🌲 Final Model

The deployed model is an optimized:

### **Random Forest Regressor**

The final model was trained after:

* Feature preprocessing
* Pipeline integration
* Hyperparameter tuning
* Cross-validation
* Model comparison

---

# 📂 Project Structure

```text
Insurance-Predictor/
│
├── app.py
├── log_model.pkl
├── retrain_model.pkl
├── currency.pkl
├── requirements.txt
├── README.md
├── assets/
│   ├── screenshots/
│   └── logo.png
└── .venv/
```

---

# 📦 Model Files

## `log_model.pkl`

Main deployed prediction model.

Contains:

* Preprocessing Pipeline
* ColumnTransformer
* Optimized Random Forest Regressor

---

## `retrain_model.pkl`

Retrained version of the model used for experimentation and model comparison.

---

## `currency.pkl`

Stores the list of supported currencies used in the application.

The selected currency is combined with a live Exchange Rate API to convert the predicted insurance premium from **INR** into the user's preferred currency.

---

# 🌍 Currency Conversion

After generating the insurance premium in **Indian Rupees (INR)**, the application automatically fetches the latest exchange rates and converts the prediction into the selected currency.

Supported examples:

* 🇮🇳 INR
* 🇺🇸 USD
* 🇪🇺 EUR
* 🇬🇧 GBP
* 🇯🇵 JPY
* and many more...

---

# 📊 Input Features

The prediction is based on the following user inputs:

* 👤 Age
* 🚻 Gender
* ⚖️ BMI
* 👨‍👩‍👧 Number of Dependants
* 🚬 Smoking Status
* 🌎 Region

---

# 📈 Output

The application provides:

* 💰 Estimated Annual Insurance Premium (INR)
* 🌍 Converted Premium in Selected Currency

---

# 🛠️ Tech Stack

### Programming Language

* Python

### Machine Learning

* Scikit-learn
* Random Forest Regressor
* XGBoost
* Pipeline
* ColumnTransformer
* GridSearchCV
* K-Fold Cross Validation

### Data Analysis

* Pandas
* NumPy

### Visualization

* Matplotlib
* Seaborn

### Web Framework

* Streamlit

### Utilities

* Pickle
* Requests

---

# ▶️ Run the Application

```bash
streamlit run app.py
```

---

# 📦 Dependencies

Install all required packages:

```bash
pip install -r requirements.txt
```

---

# 🚀 Future Improvements

* 📄 Download prediction report as PDF
* 🔐 User authentication
* 📊 Prediction history
* 📱 Mobile-friendly interface
* ☁️ Cloud deployment
* 🤖 Support for additional ML models
* 📈 Model monitoring and automated retraining

---


---

# 👨‍💻 Author

**Kaushal Kumar**

B.Tech Computer Science Engineering
Machine Learning • Artificial Intelligence • Data Science
