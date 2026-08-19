import streamlit as st
import pandas as pd
import joblib

# ------------------------------------------------------------
# PAGE CONFIGURATION
# ------------------------------------------------------------

st.set_page_config(
    page_title="EMIPredict AI",
    page_icon="💰",
    layout="wide"
)

# ------------------------------------------------------------
# LOAD MODEL AND PREPROCESSOR
# ------------------------------------------------------------

@st.cache_resource
def load_model():
    model = joblib.load("emi_logistic_regression_model.pkl")
    preprocessor = joblib.load("emi_preprocessor.pkl")
    return model, preprocessor


model, preprocessor = load_model()

# ------------------------------------------------------------
# TITLE
# ------------------------------------------------------------

st.title("💰 EMIPredict AI")
st.subheader("Intelligent Financial Risk Assessment Platform")

st.write(
    "Enter the applicant's financial and personal information "
    "to predict EMI eligibility."
)

st.divider()

# ------------------------------------------------------------
# INPUT FORM
# ------------------------------------------------------------

with st.form("emi_prediction_form"):

    st.subheader("👤 Personal Information")

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input(
            "Age",
            min_value=18,
            max_value=100,
            value=30
        )

    with col2:
        gender = st.selectbox(
            "Gender",
            ["Male", "Female"]
        )

    with col3:
        marital_status = st.selectbox(
            "Marital Status",
            ["Married", "Single"]
        )

    col1, col2 = st.columns(2)

    with col1:
        education = st.selectbox(
            "Education",
            ["High School", "Graduate", "Post Graduate"]
        )

    with col2:
        employment_type = st.selectbox(
            "Employment Type",
            ["Salaried", "Self-employed"]
        )

    st.divider()

    st.subheader("💼 Employment & Housing")

    col1, col2, col3 = st.columns(3)

    with col1:
        years_of_employment = st.number_input(
            "Years of Employment",
            min_value=0.0,
            max_value=50.0,
            value=5.0
        )

    with col2:
        company_type = st.selectbox(
            "Company Type",
            ["Private", "Government", "Public", "Other"]
        )

    with col3:
        house_type = st.selectbox(
            "House Type",
            ["Owned", "Rented", "Other"]
        )

    col1, col2 = st.columns(2)

    with col1:
        dependents = st.number_input(
            "Number of Dependents",
            min_value=0,
            max_value=20,
            value=1
        )

    with col2:
        monthly_rent = st.number_input(
            "Monthly Rent",
            min_value=0.0,
            value=10000.0,
            step=500.0
        )

    st.divider()

    st.subheader("💵 Financial Information")

    col1, col2, col3 = st.columns(3)

    with col1:
        monthly_salary = st.number_input(
            "Monthly Salary",
            min_value=0.0,
            value=50000.0,
            step=1000.0
        )

    with col2:
        bank_balance = st.number_input(
            "Bank Balance",
            min_value=0.0,
            value=100000.0,
            step=5000.0
        )

    with col3:
        emergency_fund = st.number_input(
            "Emergency Fund",
            min_value=0.0,
            value=50000.0,
            step=5000.0
        )

    col1, col2, col3 = st.columns(3)

    with col1:
        school_fees = st.number_input(
            "School Fees",
            min_value=0.0,
            value=0.0,
            step=500.0
        )

    with col2:
        college_fees = st.number_input(
            "College Fees",
            min_value=0.0,
            value=0.0,
            step=500.0
        )

    with col3:
        travel_expenses = st.number_input(
            "Travel Expenses",
            min_value=0.0,
            value=3000.0,
            step=500.0
        )

    col1, col2 = st.columns(2)

    with col1:
        groceries_utilities = st.number_input(
            "Groceries & Utilities",
            min_value=0.0,
            value=8000.0,
            step=500.0
        )

    with col2:
        other_monthly_expenses = st.number_input(
            "Other Monthly Expenses",
            min_value=0.0,
            value=5000.0,
            step=500.0
        )

    st.divider()

    st.subheader("🏦 Loan & EMI Information")

    col1, col2, col3 = st.columns(3)

    with col1:
        existing_loans = st.selectbox(
            "Existing Loans",
            ["Yes", "No"]
        )

    with col2:
        current_emi_amount = st.number_input(
            "Current EMI Amount",
            min_value=0.0,
            value=5000.0,
            step=500.0
        )

    with col3:
        credit_score = st.number_input(
            "Credit Score",
            min_value=300.0,
            max_value=900.0,
            value=700.0,
            step=1.0
        )

    col1, col2, col3 = st.columns(3)

    with col1:
        emi_scenario = st.selectbox(
            "EMI Scenario",
            [
                "Personal Loan EMI",
                "Home Loan EMI",
                "Education Loan EMI",
                "Car Loan EMI",
                "Other"
            ]
        )

    with col2:
        requested_amount = st.number_input(
            "Requested Loan Amount",
            min_value=0.0,
            value=100000.0,
            step=5000.0
        )

    with col3:
        requested_tenure = st.number_input(
            "Requested Tenure (months)",
            min_value=1,
            max_value=360,
            value=24
        )

    st.divider()

    submit = st.form_submit_button(
        "🔍 Predict EMI Eligibility",
        use_container_width=True
    )

# ------------------------------------------------------------
# PREDICTION
# ------------------------------------------------------------

if submit:

    input_data = pd.DataFrame({
        "age": [age],
        "gender": [gender],
        "marital_status": [marital_status],
        "education": [education],
        "monthly_salary": [monthly_salary],
        "employment_type": [employment_type],
        "years_of_employment": [years_of_employment],
        "company_type": [company_type],
        "house_type": [house_type],
        "monthly_rent": [monthly_rent],
        "dependents": [dependents],
        "school_fees": [school_fees],
        "college_fees": [college_fees],
        "travel_expenses": [travel_expenses],
        "groceries_utilities": [groceries_utilities],
        "other_monthly_expenses": [other_monthly_expenses],
        "existing_loans": [existing_loans],
        "current_emi_amount": [current_emi_amount],
        "credit_score": [credit_score],
        "bank_balance": [bank_balance],
        "emergency_fund": [emergency_fund],
        "emi_scenario": [emi_scenario],
        "requested_amount": [requested_amount],
        "requested_tenure": [requested_tenure]
    })

    try:

        # Transform input using saved preprocessor
        transformed_data = preprocessor.transform(input_data)

        # Generate prediction
        prediction = model.predict(transformed_data)[0]

        # Generate probabilities
        probabilities = model.predict_proba(transformed_data)[0]

        classes = model.classes_

        probability_df = pd.DataFrame({
            "Eligibility": classes,
            "Probability": probabilities
        })

        # ----------------------------------------------------
        # DISPLAY RESULT
        # ----------------------------------------------------

        st.divider()

        st.subheader("📊 EMI Eligibility Prediction")

        if prediction == "Eligible":
            st.success(
                "✅ Prediction: ELIGIBLE"
            )

        elif prediction == "High_Risk":
            st.warning(
                "⚠️ Prediction: HIGH RISK"
            )

        else:
            st.error(
                "❌ Prediction: NOT ELIGIBLE"
            )

        # Confidence
        confidence = max(probabilities) * 100

        st.metric(
            "Prediction Confidence",
            f"{confidence:.2f}%"
        )

        st.subheader("Prediction Probabilities")

        for class_name, probability in zip(classes, probabilities):

            st.write(
                f"**{class_name}**: "
                f"{probability * 100:.2f}%"
            )

            st.progress(float(probability))

        st.subheader("Input Summary")

        st.dataframe(
            input_data,
            use_container_width=True
        )

    except Exception as e:

        st.error("Prediction failed.")

        st.exception(e)