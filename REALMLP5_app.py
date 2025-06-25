import streamlit as st

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('student_test_scores.csv')

X = df[['study_hours']]
y = df['test_score']

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

Real_poly = PolynomialFeatures(degree = 2)
X_real_poly = Real_poly.fit_transform(X)

#retrain model on all data
retrain = LinearRegression()
retrain.fit(X_real_poly, y)

polypred = retrain.predict(X_real_poly)


from sklearn.metrics import r2_score, mean_absolute_error

r2 = r2_score(y, polypred)
mae = mean_absolute_error(y,polypred) 

print("R² Score:", round(r2, 4))
print("Mean Absolute Error (MAE):", round(mae, 2))

st.markdown(
    "<h3 style='color:black;'>📊 Model Performance</h3>",
    unsafe_allow_html=True
)

st.markdown(
    f"""
    <div style='color: black; font-size: 18px;'>
        🧠 <strong>R² Score:</strong> {round(r2, 4)} <br>
        📏 <strong>MAE:</strong> {round(mae, 2)}
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <style>
    .stApp {
        background-image: url("https://i.imgur.com/rsunhtC.jpeg");
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }
    </style>
    """,
    unsafe_allow_html=True
)


st.markdown(
    "<h1 style='color:black;'>📘 Study Hours Predictor</h1>",
    unsafe_allow_html=True
)


st.markdown(
    """
    <style>
    .stNumberInput label {
        color: black !important;
        font-weight: bold;
    }
    </style>
    """,
    unsafe_allow_html=True
)


hours = st.number_input("Enter Study Hours", min_value=0.0, step=0.1)

# Example: Load your model and data
# model = trained LinearRegression()
# poly = trained PolynomialFeatures()

# User input

if st.button("Predict Score"):
    # Prepare input and prediction
    X_input = np.array([[hours]])
    X_input_poly = Real_poly.transform(X_input)
    score = retrain.predict(X_input_poly)[0]

    # Display styled prediction
    st.markdown(
        f"""
        <div style='
            background-color: black;
            color: white;
            padding: 12px 18px;
            border-radius: 8px;
            font-size: 18px;
            font-weight: bold;
            margin-top: 10px;
        '>
            🎯 Predicted Test Score: {score:.2f}
        </div>
        """,
        unsafe_allow_html=True
    )

    # Custom feedback message
    if hours >= 5.01:
        st.warning("🧠 You've studied a lot — consider taking a break!")
    elif hours <= 1:
        st.info("📚 A bit more study time might help improve your score!")
    else:
        st.success("✅ Great balance — keep it up!")

    # Plot regression line and user's point
    X_range = np.linspace(0, 10, 100).reshape(-1, 1)
    y_range = retrain.predict(Real_poly.transform(X_range))
    
    plt.style.use('dark_background')
    plt.figure(figsize=(8, 5))
    plt.plot(X_range, y_range, label='Regression Line', color='blue')
    plt.scatter(hours, score, color='red', s=100, label='Your Prediction')
    plt.xlabel("Study Hours")
    plt.ylabel("Predicted Score")
    plt.title("Study Hours vs Predicted Test Score")
    plt.legend()
    plt.grid(True)
    st.pyplot(plt)
   
df_results = pd.DataFrame({
    'study_hours': X['study_hours'],           
    'Actual Salary': y,
    'Predicted Salary': polypred.astype(int)
})
df_results


