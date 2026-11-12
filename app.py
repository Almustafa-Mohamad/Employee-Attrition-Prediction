import os
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# ==========================================
# 1. Page configuration
# ==========================================

st.set_page_config(
    page_title="Employee Attrition Prediction",
    page_icon="📊",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"

MODEL_PATH = "random_forest_smote.pkl"
PREPROCESSOR_PATH = "preprocessor.pkl"

FEATURE_NAMES = [
    "Age",
    "BusinessTravel",
    "DailyRate",
    "Department",
    "DistanceFromHome",
    "Education",
    "EducationField",
    "EnvironmentSatisfaction",
    "Gender",
    "HourlyRate",
    "JobInvolvement",
    "JobLevel",
    "JobRole",
    "JobSatisfaction",
    "MaritalStatus",
    "MonthlyIncome",
    "MonthlyRate",
    "NumCompaniesWorked",
    "OverTime",
    "PercentSalaryHike",
    "PerformanceRating",
    "RelationshipSatisfaction",
    "StockOptionLevel",
    "TotalWorkingYears",
    "TrainingTimesLastYear",
    "WorkLifeBalance",
    "YearsAtCompany",
    "YearsInCurrentRole",
    "YearsSinceLastPromotion",
    "YearsWithCurrManager",
    "AgeGroup",
    "TenureGroup"
]

# ==========================================
# 2. Load saved model and preprocessor
# ==========================================

@st.cache_resource
def load_artifacts():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}"
        )

    if not os.path.exists(PREPROCESSOR_PATH):
        raise FileNotFoundError(
            f"Preprocessor file not found: {PREPROCESSOR_PATH}"
        )

    model = joblib.load(MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)

    return model, preprocessor


try:
    model, preprocessor = load_artifacts()
except Exception as error:
    st.error("Could not load the saved model files.")
    st.exception(error)
    st.stop()

# ==========================================
# 3. Application title
# ==========================================

st.title("Employee Attrition Prediction")

st.subheader(
    "AI-powered Employee Prediction System"
)

st.write(
    "Enter employee information to estimate "
    "the likelihood of employee attrition."
)

st.divider()

# ==========================================
# 4. Employee input form
# ==========================================

with st.form("employee_prediction_form"):

    st.header("Employee Information")

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input(
            "Age",
            min_value=18,
            max_value=65,
            value=30
        )

        gender = st.selectbox(
            "Gender",
            ["Male", "Female"]
        )

        marital_status = st.selectbox(
            "Marital Status",
            ["Single", "Married", "Divorced"]
        )

        department = st.selectbox(
            "Department",
            ["Sales", "Research & Development", "Human Resources"]
        )

        business_travel = st.selectbox(
            "Business Travel",
            [
                "Non-Travel",
                "Travel_Rarely",
                "Travel_Frequently"
            ]
        )

        distance_from_home = st.number_input(
            "Distance From Home",
            min_value=1,
            max_value=29,
            value=10
        )

        education = st.selectbox(
            "Education",
            [1, 2, 3, 4, 5],
            index=2
        )

        education_field = st.selectbox(
            "Education Field",
            [
                "Life Sciences",
                "Medical",
                "Marketing",
                "Technical Degree",
                "Human Resources",
                "Other"
            ]
        )

    with col2:
        job_role = st.selectbox(
            "Job Role",
            [
                "Sales Executive",
                "Research Scientist",
                "Laboratory Technician",
                "Manufacturing Director",
                "Healthcare Representative",
                "Manager",
                "Sales Representative",
                "Research Director",
                "Human Resources"
            ]
        )

        job_level = st.selectbox(
            "Job Level",
            [1, 2, 3, 4, 5]
        )

        job_satisfaction = st.selectbox(
            "Job Satisfaction",
            [1, 2, 3, 4]
        )

        job_involvement = st.selectbox(
            "Job Involvement",
            [1, 2, 3, 4]
        )

        environment_satisfaction = st.selectbox(
            "Environment Satisfaction",
            [1, 2, 3, 4]
        )

        relationship_satisfaction = st.selectbox(
            "Relationship Satisfaction",
            [1, 2, 3, 4]
        )

        performance_rating = st.selectbox(
            "Performance Rating",
            [1, 2, 3, 4]
        )

        overtime = st.selectbox(
            "OverTime",
            ["No", "Yes"]
        )

    with col3:
        monthly_income = st.number_input(
            "Monthly Income",
            min_value=100,
            max_value=20000,
            value=5000
        )

        monthly_rate = st.number_input(
            "Monthly Rate",
            min_value=100,
            max_value=30000,
            value=15000
        )

        daily_rate = st.number_input(
            "Daily Rate",
            min_value=100,
            max_value=1500,
            value=800
        )

        hourly_rate = st.number_input(
            "Hourly Rate",
            min_value=30,
            max_value=100,
            value=65
        )

        percent_salary_hike = st.number_input(
            "Percent Salary Hike",
            min_value=0,
            max_value=25,
            value=15
        )

        stock_option_level = st.selectbox(
            "Stock Option Level",
            [0, 1, 2, 3]
        )

    st.divider()

    st.header("Work Experience")

    col4, col5, col6 = st.columns(3)

    with col4:
        total_working_years = st.number_input(
            "Total Working Years",
            min_value=0,
            max_value=40,
            value=8
        )

        years_at_company = st.number_input(
            "Years At Company",
            min_value=0,
            max_value=40,
            value=5
        )

        years_in_current_role = st.number_input(
            "Years In Current Role",
            min_value=0,
            max_value=18,
            value=3
        )

    with col5:
        years_since_last_promotion = st.number_input(
            "Years Since Last Promotion",
            min_value=0,
            max_value=15,
            value=2
        )

        years_with_curr_manager = st.number_input(
            "Years With Current Manager",
            min_value=0,
            max_value=17,
            value=3
        )

        num_companies_worked = st.number_input(
            "Number of Companies Worked",
            min_value=0,
            max_value=9,
            value=2
        )

    with col6:
        training_times_last_year = st.number_input(
            "Training Times Last Year",
            min_value=0,
            max_value=6,
            value=3
        )

        work_life_balance = st.selectbox(
            "Work Life Balance",
            [1, 2, 3, 4]
        )

    submitted = st.form_submit_button(
        "Predict Employee Attrition",
        use_container_width=True
    )

# ==========================================
# 5. Prediction
# ==========================================

if submitted:

    try:
        # Create engineered features
        if age <= 25:
            age_group = "Young"
        elif age <= 35:
            age_group = "Adult"
        elif age <= 45:
            age_group = "Middle Age"
        elif age <= 55:
            age_group = "Senior"
        else:
            age_group = "Older"

        if years_at_company <= 2:
            tenure_group = "New Employee"
        elif years_at_company <= 5:
            tenure_group = "Short Tenure"
        elif years_at_company <= 10:
            tenure_group = "Medium Tenure"
        else:
            tenure_group = "Long Tenure"

        # Build input data
        employee_data = {
            "Age": age,
            "BusinessTravel": business_travel,
            "DailyRate": daily_rate,
            "Department": department,
            "DistanceFromHome": distance_from_home,
            "Education": education,
            "EducationField": education_field,
            "EnvironmentSatisfaction": environment_satisfaction,
            "Gender": gender,
            "HourlyRate": hourly_rate,
            "JobInvolvement": job_involvement,
            "JobLevel": job_level,
            "JobRole": job_role,
            "JobSatisfaction": job_satisfaction,
            "MaritalStatus": marital_status,
            "MonthlyIncome": monthly_income,
            "MonthlyRate": monthly_rate,
            "NumCompaniesWorked": num_companies_worked,
            "OverTime": overtime,
            "PercentSalaryHike": percent_salary_hike,
            "PerformanceRating": performance_rating,
            "RelationshipSatisfaction": relationship_satisfaction,
            "StockOptionLevel": stock_option_level,
            "TotalWorkingYears": total_working_years,
            "TrainingTimesLastYear": training_times_last_year,
            "WorkLifeBalance": work_life_balance,
            "YearsAtCompany": years_at_company,
            "YearsInCurrentRole": years_in_current_role,
            "YearsSinceLastPromotion": years_since_last_promotion,
            "YearsWithCurrManager": years_with_curr_manager,
            "AgeGroup": age_group,
            "TenureGroup": tenure_group
        }

        input_data = pd.DataFrame([employee_data])

        # Verify expected columns
        if hasattr(preprocessor, "feature_names_in_"):
            expected_columns = list(
                preprocessor.feature_names_in_
            )
        else:
            expected_columns = FEATURE_NAMES

        missing_columns = [
            column
            for column in expected_columns
            if column not in input_data.columns
        ]

        if missing_columns:
            st.error(
                "Some required columns are missing:"
            )
            st.write(missing_columns)
            st.stop()

        # Keep exact expected column order
        input_data = input_data[expected_columns]

        # Transform input
        processed_data = preprocessor.transform(input_data)

        # Predict
        prediction = model.predict(processed_data)[0]

        st.divider()
        st.header("Prediction Result")

        # Display prediction
        if str(prediction).lower() in ["1", "yes", "true"]:
            st.error(
                "High Risk: The model predicts "
                "that this employee may leave the company."
            )
        elif str(prediction).lower() in ["0", "no", "false"]:
            st.success(
                "Low Risk: The model predicts "
                "that this employee is likely to stay."
            )
        else:
            st.info(
                f"Model prediction: {prediction}"
            )

        # Show probability when available
        if hasattr(model, "predict_proba"):

            probabilities = model.predict_proba(
                processed_data
            )[0]

            classes = list(model.classes_)

            positive_index = None

            for index, class_value in enumerate(classes):
                if str(class_value).lower() in [
                    "1", "yes", "true"
                ]:
                    positive_index = index
                    break

            if positive_index is not None:
                leave_probability = probabilities[
                    positive_index
                ]

                st.metric(
                    "Estimated Probability of Leaving",
                    f"{leave_probability:.1%}"
                )

                st.progress(
                    float(leave_probability)
                )

    except Exception as error:
        st.error("Prediction could not be completed.")
        st.exception(error)

# ==========================================
# 6. Footer
# ==========================================

st.divider()

st.caption(
    "This model is for decision support only. "
    "It should not be the sole basis for employment decisions."
)

# ==========================================
# 7. Quick test with sample employees
# ==========================================

st.divider()
st.header("Quick Model Testing")

if st.button("Run 3 Sample Employee Tests"):

    sample_employees = [
        {
            "Age": 45,
            "BusinessTravel": "Travel_Rarely",
            "DailyRate": 1200,
            "Department": "Research & Development",
            "DistanceFromHome": 5,
            "Education": 4,
            "EducationField": "Medical",
            "EnvironmentSatisfaction": 4,
            "Gender": "Female",
            "HourlyRate": 80,
            "JobInvolvement": 4,
            "JobLevel": 3,
            "JobRole": "Research Scientist",
            "JobSatisfaction": 4,
            "MaritalStatus": "Married",
            "MonthlyIncome": 8000,
            "MonthlyRate": 18000,
            "NumCompaniesWorked": 1,
            "OverTime": "No",
            "PercentSalaryHike": 20,
            "PerformanceRating": 4,
            "RelationshipSatisfaction": 4,
            "StockOptionLevel": 2,
            "TotalWorkingYears": 20,
            "TrainingTimesLastYear": 4,
            "WorkLifeBalance": 4,
            "YearsAtCompany": 15,
            "YearsInCurrentRole": 8,
            "YearsSinceLastPromotion": 2,
            "YearsWithCurrManager": 8,
            "AgeGroup": "Middle Age",
            "TenureGroup": "Long Tenure"
        },
        {
            "Age": 24,
            "BusinessTravel": "Travel_Frequently",
            "DailyRate": 300,
            "Department": "Sales",
            "DistanceFromHome": 25,
            "Education": 2,
            "EducationField": "Marketing",
            "EnvironmentSatisfaction": 1,
            "Gender": "Male",
            "HourlyRate": 40,
            "JobInvolvement": 1,
            "JobLevel": 1,
            "JobRole": "Sales Representative",
            "JobSatisfaction": 1,
            "MaritalStatus": "Single",
            "MonthlyIncome": 2000,
            "MonthlyRate": 10000,
            "NumCompaniesWorked": 4,
            "OverTime": "Yes",
            "PercentSalaryHike": 11,
            "PerformanceRating": 3,
            "RelationshipSatisfaction": 1,
            "StockOptionLevel": 0,
            "TotalWorkingYears": 2,
            "TrainingTimesLastYear": 1,
            "WorkLifeBalance": 1,
            "YearsAtCompany": 1,
            "YearsInCurrentRole": 0,
            "YearsSinceLastPromotion": 0,
            "YearsWithCurrManager": 0,
            "AgeGroup": "Young",
            "TenureGroup": "New Employee"
        },
        {
            "Age": 35,
            "BusinessTravel": "Travel_Rarely",
            "DailyRate": 700,
            "Department": "Human Resources",
            "DistanceFromHome": 10,
            "Education": 3,
            "EducationField": "Human Resources",
            "EnvironmentSatisfaction": 2,
            "Gender": "Female",
            "HourlyRate": 60,
            "JobInvolvement": 2,
            "JobLevel": 2,
            "JobRole": "Human Resources",
            "JobSatisfaction": 2,
            "MaritalStatus": "Divorced",
            "MonthlyIncome": 4500,
            "MonthlyRate": 15000,
            "NumCompaniesWorked": 2,
            "OverTime": "Yes",
            "PercentSalaryHike": 13,
            "PerformanceRating": 3,
            "RelationshipSatisfaction": 2,
            "StockOptionLevel": 1,
            "TotalWorkingYears": 10,
            "TrainingTimesLastYear": 2,
            "WorkLifeBalance": 2,
            "YearsAtCompany": 4,
            "YearsInCurrentRole": 2,
            "YearsSinceLastPromotion": 1,
            "YearsWithCurrManager": 2,
            "AgeGroup": "Adult",
            "TenureGroup": "Short Tenure"
        }
    ]

    results = []

    try:
        for i, employee in enumerate(sample_employees, start=1):

            sample_df = pd.DataFrame([employee])
            sample_df = sample_df[FEATURE_NAMES]

            processed = preprocessor.transform(sample_df)
            prediction = model.predict(processed)[0]

            result = {
                "Employee": f"Employee {i}",
                "Prediction": str(prediction)
            }

            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(processed)[0]
                classes = list(model.classes_)

                positive_index = next(
                    (
                        j for j, value in enumerate(classes)
                        if str(value).lower() in ["1", "yes", "true"]
                    ),
                    None
                )

                if positive_index is not None:
                    result["Leaving Probability"] = (
                        f"{probabilities[positive_index]:.1%}"
                    )

            results.append(result)

        st.success("All sample tests completed!")
        st.dataframe(
            pd.DataFrame(results),
            use_container_width=True
        )

    except Exception as error:
        st.error("Sample testing failed.")
        st.exception(error)
