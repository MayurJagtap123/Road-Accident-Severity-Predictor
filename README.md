# Accident Severity Prediction Based on Road and Environmental Conditions

This project aims to predict the severity of road accidents using machine learning techniques based on features like road type, junction details, light conditions, weather, and vehicle type. The model can help authorities understand high-risk factors and improve road safety.

---

## 🚀 Features

- Predicts accident severity as **Slight**, **Serious**, or **Fetal**.
- Trained using the **XGBoost** classifier for high performance.
- Includes data preprocessing, label encoding, and model training.
- Saves model and encoders using **Joblib** for future use.
- Includes multiple **data visualizations** to explore trends and insights.

---

## 🛠 Technologies Used

- Python
- Pandas & NumPy
- Scikit-learn
- XGBoost
- Joblib
- Matplotlib & Seaborn

---

## 📂 Dataset

- File: `Road Accident Data.csv`
- Columns used:
  - `Junction_Control`
  - `Junction_Detail`
  - `Light_Conditions`
  - `Road_Surface_Conditions`
  - `Road_Type`
  - `Weather_Conditions`
  - `Vehicle_Type`
  - `Accident_Severity` (Target)

---

## 📊 Visualizations

1. **Bar Chart** – Count of accidents by severity
2. **Pie Chart** – Proportion of each severity type
3. **Count Plots** – Accidents under different weather or lighting
4. **Heatmap** – Correlation between numerical features
5. **Box Plot** – Severity vs Light or Road conditions

These visuals help understand which conditions contribute to severe accidents.

---

## 📈 Model Training

- **Model Used:** XGBoost Classifier
- **Evaluation Metric:** Accuracy
- **Train-Test Split:** 80% - 20%
- **Target Encoding:** LabelEncoder
- **Input Encoding:** One-Hot Encoding

---

## ✅ Results

- Achieved high accuracy on the test data.
- Model is saved as `xgboost_classification_model.pkl`.
- Columns and encoders saved for deployment.

---

## 📦 Output Files

- `xgboost_classification_model.pkl` – Trained model
- `label_encoder.pkl` – Encoded target class mapping
- `model_columns.pkl` – Feature columns for input mapping

---

## 📌 Conclusion

This project demonstrates how machine learning can be used to forecast accident severity based on various environmental and road-related features. It can serve as a foundation for building intelligent road safety systems.

---

## 📄 Author

**Mayur Jagtap**  
Computer Engineering | Data Science Enthusiast

---