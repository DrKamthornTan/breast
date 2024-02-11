import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# Load the CSV data
data = pd.read_csv("breast_cancer.csv")

# Extract the numeric columns for regression
numeric_columns = ["Age", "TumorSize", "SurvivalMonths"]
numeric_data = data[numeric_columns]

# Extract the categorical columns for plotting
categorical_columns = ["Tstage ", "NStage", "sixthStage", "differentiate", "Grade", "AStage"]

# Compute the median age
median_age = numeric_data["Age"].median()

# Train the initial regression model
X = numeric_data.drop("SurvivalMonths", axis=1)
y = numeric_data["SurvivalMonths"]
regression_model = LinearRegression()
regression_model.fit(X, y)

# Plot the initial regression line in black
plt.scatter(X["Age"], y, color="black", label="Data")
plt.plot(X["Age"], regression_model.predict(X), color="black", label="Initial Regression Line")

# Streamlit App
st.title("การคาดการณ์จำนวนเดือนที่รอด (Survival Months) ของมะเร็งเต้านม")
st.write("ฝึกจากชุดข้อมูลจำนวน 4,000 รายผู้ป่วยทั้งที่รอดและเสียชีวิต")

# User Inputs
age = st.slider("อายุ", int(numeric_data["Age"].min()), 100, int(median_age))
tumor_size = st.slider("ขนาดมะเร็ง (มม.)",0, int(numeric_data["TumorSize"].max()), int(numeric_data["TumorSize"].min()))

# Modify the features based on user input
modified_X = X.copy()
modified_X.loc[0, "Age"] = age
modified_X.loc[0, "TumorSize"] = tumor_size

# Predictions
initial_prediction = regression_model.predict(X)
modified_prediction = regression_model.predict(modified_X)

# Plot the second regression line in blue
plt.plot(X["Age"], initial_prediction, color="blue", label="Second Regression Line")

# Plot the third regression line in red or green
if modified_prediction[0] < initial_prediction[0]:
    plt.plot(modified_X["Age"], modified_prediction - 10, color="red", label="Third Regression Line")
else:
    plt.plot(modified_X["Age"], modified_prediction + 10, color="green", label="Third Regression Line")

# Plot settings
plt.xlabel("Age")
plt.ylabel("Survival Months")
plt.legend()

# Show the plot
st.pyplot(plt.gcf())

# Accuracy metrics
y_pred = regression_model.predict(X)
mse = mean_squared_error(y, y_pred)
r2 = r2_score(y, y_pred)

st.write(f"Mean Squared Error: {mse}")
st.write(f"R^2 Score: {r2}")

