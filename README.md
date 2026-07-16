# Heart Disease Prediction App

A professional machine learning web application built with Streamlit to estimate the risk of heart disease based on patient clinical and demographic information.

The app uses a trained **Tuned Random Forest** model and provides an interactive dashboard for entering patient data and viewing prediction results.

---

## Project Overview

This project is designed as an end-to-end machine learning portfolio project.  
It includes model development, optimization, model serialization, and deployment through a Streamlit web application.

The application allows users to input clinical features such as age, sex, chest pain type, blood pressure, cholesterol level, maximum heart rate, and exercise-related indicators. Based on these inputs, the model predicts the likelihood of heart disease.

> Disclaimer: This application is for educational and informational purposes only and should not be used as a substitute for professional medical advice, diagnosis, or treatment.

---

## App Features

- Interactive Streamlit dashboard
- Custom-designed user interface
- Patient clinical input form
- Heart disease risk prediction
- Probability visualization using a gauge chart
- Diagnostic output cards
- Sidebar project information
- Cached model loading for better performance

---

## Machine Learning Model

The final model used in this application is a:
```text
Tuned Random Forest Classifier

The trained model is loaded from a serialized `.pkl` file using `joblib`.

Model file:

text
final_random_forest.pkl

The application uses `st.cache_resource` to load the model efficiently and avoid reloading it on every interaction.

---

## Input Features

The model uses the following patient information as input:

| Feature | Description |
|--------|-------------|
| `age` | Patient age |
| `sex` | Patient sex |
| `cp` | Chest pain type |
| `trestbps` | Resting blood pressure |
| `chol` | Serum cholesterol |
| `fbs` | Fasting blood sugar > 120 mg/dl |
| `restecg` | Resting electrocardiographic results |
| `thalach` | Maximum heart rate achieved |
| `exang` | Exercise-induced angina |
| `oldpeak` | ST depression induced by exercise |
| `slope` | Slope of the peak exercise ST segment |

---

## Tech Stack

- Python
- Streamlit
- Scikit-learn
- Pandas
- Joblib
- Plotly

---

## Project Structure

text
heart-disease-prediction/
│
├── app.py
├── final_random_forest.pkl
├── requirements.txt
├── README.md
└── .gitignore

Depending on your local structure, the model file may also be placed inside a `models/` directory.

---

## Installation

Clone the repository:

bash
git clone https://github.com/your-username/heart-disease-prediction.git
cd heart-disease-prediction

Create and activate a virtual environment:

bash
python -m venv venv

On Windows:

bash
venv\Scripts\activate

On macOS/Linux:

bash
source venv/bin/activate

Install dependencies:

bash
pip install -r requirements.txt

---

## Run the App Locally

Use the following command:

bash
streamlit run app.py

Then open the local URL shown in the terminal.

Usually:

text
http://localhost:8501

---

## Deployment

This app can be deployed using Streamlit Community Cloud.

General deployment steps:

1. Push the project to GitHub.
2. Go to [Streamlit Community Cloud](https://streamlit.io/cloud).
3. Connect your GitHub repository.
4. Select the repository and branch.
5. Set the main file path as:

text
app.py

6. Deploy the app.

---

## Requirements

Example `requirements.txt`:

txt
streamlit
scikit-learn
pandas
numpy
joblib
plotly

If the model was trained with a specific version of `scikit-learn`, it is recommended to pin that version to avoid compatibility issues when loading the `.pkl` file.

Example:

txt
scikit-learn==1.5.1

---

## Important Notes

- Make sure the model path inside `app.py` matches the actual location of `final_random_forest.pkl`.
- Avoid using absolute local paths such as `C:/Users/...`.
- Use relative paths for deployment compatibility.
- If using a `models/` folder, keep the model loading path consistent.

---

## Disclaimer

This project is intended for educational and portfolio purposes.  
It is not a medical diagnostic tool and should not be used for clinical decision-making.

---

## Author

Developed by **Your Name**

Machine Learning / Data Science Portfolio Project


---

# نکته مهم درباره README

در این README چند جای قابل شخصی‌سازی هست:

## 1. نام repository

این خط را بعداً عوض می‌کنیم:

```bash
git clone https://github.com/your-username/heart-disease-prediction.git
