import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# --- Page Configuration ---
st.set_page_config(page_title="Calories Burnt Predictor", layout="wide")

# --- 1. Load and Prep Data ---
@st.cache_data
def load_data():
    try:
        df = pd.read_csv(r"C:\Users\srava\Downloads\calories.csv")
        return df
    except FileNotFoundError:
        st.error("File 'calories.csv' not found. Please place it in the same directory.")
        return None

df = load_data()

if df is not None:
    # --- 2. Train Model (On the Fly) ---
    @st.cache_resource
    def train_model(data):
        # Features and Target
        X = data[["Gender", "Age", "Height", "Weight", "Duration", "Heart_Rate", "Body_Temp"]]
        y = data["Calories"]

        # Split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Preprocessing & Model Pipeline
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), ["Age", "Height", "Weight", "Duration", "Heart_Rate", "Body_Temp"]),
                ("cat", OneHotEncoder(), ["Gender"]),
            ]
        )

        model = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("regressor", LinearRegression())
        ])

        model.fit(X_train, y_train)
        score = model.score(X_test, y_test)
        return model, score

    model, accuracy = train_model(df)

    # --- 3. Sidebar UI (User Inputs) ---
    st.sidebar.header("🏃‍♂️ User Input Parameters")
    st.sidebar.write("Adjust the sliders below:")

    def user_input_features():
        gender = st.sidebar.selectbox("Gender", ("male", "female"))
        age = st.sidebar.slider("Age", 10, 100, 30)
        height = st.sidebar.slider("Height (cm)", 100.0, 250.0, 175.0)
        weight = st.sidebar.slider("Weight (kg)", 30.0, 150.0, 75.0)
        duration = st.sidebar.slider("Duration (min)", 1.0, 120.0, 30.0)
        heart_rate = st.sidebar.slider("Heart Rate (bpm)", 60.0, 200.0, 95.0)
        body_temp = st.sidebar.slider("Body Temp (°C)", 36.0, 42.0, 40.0)

        data = {
            'Gender': gender,
            'Age': age,
            'Height': height,
            'Weight': weight,
            'Duration': duration,
            'Heart_Rate': heart_rate,
            'Body_Temp': body_temp
        }
        return pd.DataFrame(data, index=[0])

    input_df = user_input_features()

    # --- 4. Main Dashboard ---
    st.title("🔥 Calories Burnt Prediction App")
    st.markdown("""
    This application predicts the **Calories Burnt** during exercise using a Machine Learning model 
    trained on your fitness dataset.
    """)

    # Create Columns for Layout
    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Your Parameters")
        st.write(input_df.T) # Transpose for better view

    with col2:
        st.subheader("Prediction Result")
        if st.button("🚀 Predict Calories"):
            prediction = model.predict(input_df)
            st.success(f"You burnt approximately **{prediction[0]:.2f} calories**!")
            st.metric(label="Model Accuracy", value=f"{accuracy*100:.1f}%")
        else:
            st.info("Adjust the sidebar sliders and click Predict!")

    st.divider()

    # --- 5. Data Visualization Section ---
    st.header("📊 Dataset Analysis & Insights")
    
    tab1, tab2, tab3 = st.tabs(["Correlations", "Duration vs Calories", "Distributions"])

    with tab1:
        st.subheader("Feature Correlations")
        fig, ax = plt.subplots(figsize=(8, 4))
        # Select numeric columns for correlation
        numeric_df = df.select_dtypes(include=['float64', 'int64'])
        sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
        st.pyplot(fig)
        st.caption("Strong correlation (red) between Duration, Heart Rate, and Calories.")

    with tab2:
        st.subheader("Duration vs Calories Burned")
        fig, ax = plt.subplots()
        sns.scatterplot(data=df, x="Duration", y="Calories", hue="Gender", alpha=0.6, ax=ax)
        st.pyplot(fig)
        st.caption("Longer duration linearly increases calories burned.")

    with tab3:
        st.subheader("Distribution of Calories in Dataset")
        fig, ax = plt.subplots()
        sns.histplot(df["Calories"], kde=True, color="purple", ax=ax)
        st.pyplot(fig)

else:
    st.warning("Awaiting 'calories.csv' file upload...")