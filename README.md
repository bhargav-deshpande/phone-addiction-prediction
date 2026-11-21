# 📱 Teen Phone Addiction Prediction

### Machine Learning Regression Model with Streamlit Deployment

## 📌 Project Overview

Teen Phone Addiction Prediction is a Machine Learning project designed to estimate the addiction level of teenagers based on their smartphone usage patterns, daily activities, sleep habits and behavioural factors.

The project uses regression algorithms to analyze different factors associated with smartphone usage and predict an addiction level based on the provided inputs.

An interactive Streamlit application allows users to enter relevant information and obtain a predicted addiction level.

## 🎯 Problem Statement

Excessive smartphone usage among teenagers can be associated with changes in sleep patterns, academic performance, physical activity and social interactions.

This project explores the relationship between smartphone usage habits and behavioural factors using Machine Learning regression techniques.

## ⚙️ Machine Learning Workflow

1. **Data Exploration:** Examined dataset structure, statistical information and missing values.
2. **Data Preprocessing:** Separated input features and target variable.
3. **Feature Encoding:** Applied One-Hot Encoding to categorical features.
4. **Feature Scaling:** Used StandardScaler for numerical features in Linear Regression.
5. **Train-Test Split:** Divided the dataset into 80% training and 20% testing data.
6. **Model Training:** Trained and evaluated multiple regression algorithms.
7. **Model Evaluation:** Compared models using R² Score, Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE).
8. **Deployment:** Integrated the saved prediction model into a Streamlit application.

## 🤖 Machine Learning Algorithms

The following regression algorithms were implemented:

- Linear Regression
- Random Forest Regressor
- XGBoost Regressor

## 📊 Features Used

The prediction considers multiple factors, including:

- Age and Gender
- Daily Phone Usage Hours
- Sleep Hours
- Phone Checks Per Day
- Number of Apps Used Daily
- Social Media Usage
- Gaming and Educational Usage
- Weekend Phone Usage
- Screen Time Before Bed
- Exercise Hours
- Social Interactions
- Academic Performance
- Family Communication
- Anxiety Level
- Depression Level
- Self-Esteem
- Parental Control

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- XGBoost
- Joblib
- Streamlit

## 📂 Project Structure

```text
phone-addiction-prediction/
│
├── app.py
├── Addiction_level.py
├── addiction_level.pkl
├── columns.pkl
├── teen_phone_addiction_dataset.csv
├── .gitignore
└── README.md
```

## 🚀 Installation and Setup

**1. Clone the repository**

```bash
git clone https://github.com/bhargav-deshpande/phone-addiction-prediction.git
```

**2. Navigate to the project directory**

```bash
cd phone-addiction-prediction
```

**3. Install the required libraries**

```bash
pip install streamlit pandas numpy matplotlib seaborn scikit-learn xgboost joblib
```

**4. Run the Streamlit application**

```bash
streamlit run app.py
```

**5. Open the application**

The application will open in your browser, usually at:

```text
http://localhost:8501
```

## 💻 Application

The Streamlit application allows users to:

- Enter personal and smartphone usage details.
- Provide behavioural and lifestyle information.
- Generate an addiction-level prediction.
- View the predicted result through an interactive interface.

## 📚 Learning Outcomes

Through this project, I gained practical experience in:

- Data preprocessing and exploratory data analysis.
- Feature encoding and scaling.
- Implementing regression algorithms.
- Comparing Machine Learning models.
- Saving and loading trained models using Joblib.
- Deploying a Machine Learning model using Streamlit.

## ⚠️ Disclaimer

This project is an educational Machine Learning application. Its predictions are estimates based on the dataset and should not be treated as a clinical diagnosis or medical assessment.

## 👨‍💻 Author

**Bhargav Deshpande**

GitHub: [bhargav-deshpande](https://github.com/bhargav-deshpande)
