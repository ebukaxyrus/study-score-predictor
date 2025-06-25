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
    


    with plt.style.context('dark_background'):
        
        fig, ax = plt.subplots(figsize=(8, 5), facecolor='black')  # Make figure background black
        ax.set_facecolor('black')  # Make plot area background black
        
        ax.plot(X_range, y_range, label='Regression Line', color='cyan')
        ax.scatter(hours, score, color='yellow', s=100, label='Your Prediction')
        
        ax.set_xlabel("Study Hours", color='white')
        ax.set_ylabel("Predicted Score", color='white')
        ax.set_title("Study Hours vs Predicted Test Score", color='white')
        
        ax.tick_params(colors='white')  # Make axis ticks white
        ax.grid(True, color='gray')
        ax.legend()
        
        st.pyplot(fig)  # ✅ Use fig, not plt
        
   
df_results = pd.DataFrame({
    'Study Hours': X['study_hours'],           
    'Actual Score': y,
    'Predicted Score': polypred.astype(int)
})
df_results

# Step 1: Prepare X_range and prediction line
X_range = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
y_range = retrain.predict(Real_poly.transform(X_range))

# Step 2: Plot
with plt.style.context('dark_background'):
    fig, ax = plt.subplots(figsize=(8, 5), facecolor='black')
    ax.set_facecolor('black')

    # Scatter plot of actual data
    ax.scatter(X, y, color='yellow', s=60, label='Actual Data')

    # Polynomial regression line
    ax.plot(X_range, y_range, color='cyan', linewidth=2.5, label='Polynomial Regression Line')

    ax.set_xlabel("Study Hours", color='white')
    ax.set_ylabel("Test Score", color='white')
    ax.set_title("Polynomial Regression Fit", color='white')
    ax.tick_params(colors='white')
    ax.grid(True, color='gray')
    ax.legend()

    st.pyplot(fig)



