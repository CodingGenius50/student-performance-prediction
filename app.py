
import os
import joblib
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go


BASE = os.path.dirname(os.path.abspath(__file__))
ASSETS = os.path.join(BASE, "assets")
GRAPH_DIR = os.path.join(ASSETS, "graphs")

st.set_page_config(
    page_title="EduPredict AI",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------- DESIGN ----------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}
.stApp { background: #f5f7fc; }
[data-testid="stSidebar"] { background: #101d3a; }
[data-testid="stSidebar"] * { color: white !important; }
.hero {
    padding: 35px;
    border-radius: 18px;
    color: white;
    background: linear-gradient(120deg,#102c62,#2563eb);
    margin-bottom: 25px;
}
.hero h1 {font-size: 35px; font-weight: 800;}
.hero p {font-size: 16px; color: #e0eaff;}
.metric-card {
    background: white;
    padding: 22px;
    border-radius: 14px;
    border: 1px solid #e7ebf3;
}
.metric-label {color:#64748b; font-size:13px;}
.metric-value {font-size:27px; font-weight:800; color:#173b79;}
.section-title {color:#173b79; font-weight:800;}
</style>
""", unsafe_allow_html=True)

# ---------- LOAD DATA ----------
@st.cache_data
def load_data():
    return pd.read_csv(os.path.join(ASSETS, "student_data.csv"))

@st.cache_resource
def load_model():
    model = joblib.load(os.path.join(ASSETS, "best_student_model.pkl"))
    columns = joblib.load(os.path.join(ASSETS, "model_columns.pkl"))
    return model, columns

def metric_card(label, value):
    st.markdown(
        f"""<div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        </div>""",
        unsafe_allow_html=True
    )

# ---------- CHECK FILES ----------
required = [
    os.path.join(ASSETS, "student_data.csv"),
    os.path.join(ASSETS, "best_student_model.pkl"),
    os.path.join(ASSETS, "model_columns.pkl"),
    os.path.join(ASSETS, "model_comparison.csv"),
    os.path.join(ASSETS, "feature_importance.csv")
]

missing = [f for f in required if not os.path.exists(f)]

if missing:
    st.error("Required project files are missing:")
    for f in missing:
        st.write(f"- {f}")
    st.stop()

try:
    df = load_data()
    model, model_columns = load_model()
    comparison = pd.read_csv(required[3])
    importance = pd.read_csv(required[4])
except Exception as e:
    st.error(f"Could not load project files: {e}")
    st.stop()

# ---------- SIDEBAR ----------
st.sidebar.markdown("# 🎓 EduPredict AI")
st.sidebar.caption("STUDENT ACADEMIC INTELLIGENCE")
st.sidebar.divider()

page = st.sidebar.radio(
    "NAVIGATION",
    [
        "🏠 Dashboard",
        "📁 Dataset Explorer",
        "📊 Exploratory Data Analysis",
        "⚙️ Data Preprocessing",
        "🤖 Model Comparison",
        "📈 Model Evaluation",
        "🔍 Feature Importance",
        "🎯 Live Prediction",
        "📚 About Project"
    ]
)

st.sidebar.divider()
st.sidebar.caption("Machine Learning Academic Project")
st.sidebar.caption("Student Performance Prediction")

# ---------- DASHBOARD ----------
if page == "🏠 Dashboard":

    st.markdown("""
    <div class="hero">
    <h1>EduPredict AI</h1>
    <p>Student Academic Performance Prediction Using Machine Learning</p>
    <p>Explore student data, understand learning patterns and predict
    academic performance through machine learning.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("## Project Overview")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("Total Students", f"{len(df):,}")
    with c2:
        metric_card("Dataset Features", str(len(df.columns)))
    with c3:
        metric_card("ML Models", str(len(comparison)))
    with c4:
        metric_card("Target Variable", "G3 / 20")

    st.write("")
    left, right = st.columns(2)

    with left:
        st.markdown("### Grade Distribution")
        fig = px.histogram(
            df, x="G3", nbins=20,
            color_discrete_sequence=["#2563eb"],
            labels={"G3": "Final Grade"}
        )
        fig.update_layout(template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.markdown("### Model Performance")
        st.dataframe(comparison, use_container_width=True)

    st.markdown("### Project Workflow")
    st.write(
        "Dataset Collection → Exploratory Data Analysis → "
        "Preprocessing → Model Training → Evaluation → Prediction"
    )

# ---------- DATASET ----------
elif page == "📁 Dataset Explorer":

    st.title("Dataset Explorer")
    st.write("Explore the UCI Student Performance Mathematics dataset.")

    c1, c2, c3 = st.columns(3)
    c1.metric("Records", len(df))
    c2.metric("Features", len(df.columns))
    c3.metric("Missing Values", int(df.isnull().sum().sum()))

    st.markdown("### Dataset Preview")
    st.dataframe(df, use_container_width=True)

    st.markdown("### Statistical Summary")
    st.dataframe(df.describe(include="all").T, use_container_width=True)

    st.markdown("### Dataset Information")
    st.write("Target variable: G3 (Final Mathematics Grade)")
    st.write("Dataset source: UCI Machine Learning Repository")
    st.write("Duplicate records:", int(df.duplicated().sum()))

# ---------- EDA ----------
elif page == "📊 Exploratory Data Analysis":

    st.title("Exploratory Data Analysis")
    st.write("Visual exploration of student academic performance.")

    if "G3" in df.columns:
        fig = px.histogram(
            df, x="G3", nbins=20,
            title="Final Grade Distribution",
            color_discrete_sequence=["#2563eb"]
        )
        st.plotly_chart(fig, use_container_width=True)

    if "studytime" in df.columns:
        fig = px.box(
            df, x="studytime", y="G3",
            title="Weekly Study Time vs Final Grade",
            color="studytime"
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "Studytime is a weekly category: "
            "1 = less than 2 hours, 2 = 2–5 hours, "
            "3 = 5–10 hours, 4 = more than 10 hours."
        )

    if "absences" in df.columns:
        fig = px.scatter(
            df, x="absences", y="G3",
            title="Absences vs Final Grade",
            trendline="ols",
            color_discrete_sequence=["#2563eb"]
        )
        st.plotly_chart(fig, use_container_width=True)

    numeric = df.select_dtypes(include=np.number)
    fig = px.imshow(
        numeric.corr(),
        text_auto=".2f",
        color_continuous_scale="RdBu_r",
        title="Correlation Heatmap"
    )
    st.plotly_chart(fig, use_container_width=True)

# ---------- PREPROCESSING ----------
elif page == "⚙️ Data Preprocessing":

    st.title("Data Preprocessing")
    st.write("Overview of the preprocessing steps used in the notebook.")

    st.markdown("### 1. Missing Value Analysis")
    st.dataframe(
        df.isnull().sum().rename("Missing Values"),
        use_container_width=True
    )

    st.markdown("### 2. Duplicate Analysis")
    st.write("Duplicate records:", int(df.duplicated().sum()))

    st.markdown("### 3. Target and Feature Selection")
    st.code('''
X = df.drop(columns=["G1", "G2", "G3"])
y = df["G3"]
''', language="python")

    st.info(
        "G1 and G2 were excluded to avoid using earlier examination "
        "grades as inputs. G3 is the final grade prediction target."
    )

    st.markdown("### 4. Categorical Encoding")
    st.write(
        "Categorical features were converted into numerical "
        "indicator columns using one-hot encoding."
    )

    st.markdown("### 5. Train-Test Split")
    st.write("Training: 80% | Testing: 20% | random_state = 42")
    st.write("Training samples: 316 | Testing samples: 79")

# ---------- MODEL COMPARISON ----------
elif page == "🤖 Model Comparison":

    st.title("Machine Learning Model Comparison")
    st.write("Comparison of the three trained regression models.")

    st.dataframe(comparison, use_container_width=True)

    required_metrics = ["MAE", "RMSE", "R2"]
    if all(c in comparison.columns for c in required_metrics):
        for metric in required_metrics:
            fig = px.bar(
                comparison,
                x=comparison.columns[0],
                y=metric,
                color=comparison.columns[0],
                title=f"Model Comparison - {metric}"
            )
            fig.update_layout(showlegend=False, template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)

    st.info(
        "Lower MAE and RMSE indicate smaller prediction errors. "
        "Higher R² indicates more variance explained by the model."
    )

# ---------- EVALUATION ----------
elif page == "📈 Model Evaluation":

    st.title("Model Evaluation")

    st.markdown("### Best Model: Random Forest Regressor")

    c1, c2, c3 = st.columns(3)
    c1.metric("MAE", "3.0078")
    c2.metric("RMSE", "3.7729")
    c3.metric("R² Score", "0.3058")

    st.markdown("### Interpretation")
    st.write(
        "The Random Forest model achieved the best results among "
        "the three tested models on the held-out test set."
    )
    st.write(
        "R² = 0.3058 means the model explains approximately "
        "30.6% of the variation in final grades in this test set."
    )

    st.warning(
        "The model has moderate predictive performance. "
        "Predictions are estimates, not guaranteed student grades."
    )

    graph_path = os.path.join(GRAPH_DIR, "actual_vs_predicted.png")
    if os.path.exists(graph_path):
        st.image(graph_path, caption="Actual vs Predicted Grades")

# ---------- FEATURE IMPORTANCE ----------
elif page == "🔍 Feature Importance":

    st.title("Feature Importance")
    st.write("Features ranked by their importance in the Random Forest model.")

    st.dataframe(importance, use_container_width=True)

    if len(importance.columns) >= 2:
        feature_col = importance.columns[0]
        value_col = importance.columns[1]

        fig = px.bar(
            importance.head(10).sort_values(value_col),
            x=value_col,
            y=feature_col,
            orientation="h",
            title="Top 10 Important Features",
            color=value_col,
            color_continuous_scale="Blues"
        )
        st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "Feature importance indicates the model's relative reliance "
        "on features. It does not prove causation."
    )

# ---------- LIVE PREDICTION ----------
elif page == "🎯 Live Prediction":

    st.title("Live Student Grade Prediction")
    st.write(
        "Enter student information to estimate the final mathematics grade."
    )

    input_columns = [
        c for c in df.columns if c not in ["G1", "G2", "G3"]
    ]

    with st.form("prediction_form"):
        st.markdown("### Student Information")
        user_data = {}

        cols = st.columns(2)

        for i, col in enumerate(input_columns):
            with cols[i % 2]:
                series = df[col]

                if pd.api.types.is_numeric_dtype(series):
                    minimum = float(series.min())
                    maximum = float(series.max())
                    default = float(series.median())

                    if col == "studytime":
                        user_data[col] = st.selectbox(
                            "Weekly Study Time",
                            [1, 2, 3, 4],
                            format_func=lambda x: {
                                1: "Less than 2 hours",
                                2: "2–5 hours",
                                3: "5–10 hours",
                                4: "More than 10 hours"
                            }[x]
                        )
                    elif minimum == maximum:
                        user_data[col] = st.number_input(
                            col, value=minimum
                        )
                    elif float(minimum).is_integer() and float(maximum).is_integer():
                        user_data[col] = st.number_input(
                            col,
                            min_value=int(minimum),
                            max_value=int(maximum),
                            value=int(default)
                        )
                    else:
                        user_data[col] = st.number_input(
                            col,
                            min_value=minimum,
                            max_value=maximum,
                            value=default
                        )
                else:
                    options = sorted(series.dropna().astype(str).unique())
                    user_data[col] = st.selectbox(col, options)

        submitted = st.form_submit_button(
            "Predict Final Grade",
            use_container_width=True
        )

    if submitted:
        try:
            input_df = pd.DataFrame([user_data])
            encoded = pd.get_dummies(input_df)
            encoded = encoded.reindex(
                columns=model_columns,
                fill_value=0
            )
            prediction = float(model.predict(encoded)[0])
            prediction = float(np.clip(prediction, 0, 20))

            st.success("Prediction completed successfully!")

            c1, c2 = st.columns(2)
            c1.metric("Predicted Final Grade", f"{prediction:.2f} / 20")
            c2.metric(
                "Approximate Grade Percentage",
                f"{prediction / 20 * 100:.1f}%"
            )

            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prediction,
                title={"text": "Predicted Grade"},
                gauge={"axis": {"range": [0, 20]}}
            ))
            st.plotly_chart(fig, use_container_width=True)

            st.info(
                "This is an estimated grade generated by a machine "
                "learning model, not an official academic result."
            )

        except Exception as e:
            st.error(f"Prediction failed: {e}")

# ---------- ABOUT ----------
elif page == "📚 About Project":

    st.title("About EduPredict AI")

    st.markdown("""
    ### Project Title
    Student Academic Performance Prediction Using Machine Learning

    ### Objective
    To explore student-related factors and estimate final mathematics
    grades using machine learning regression models.

    ### Dataset
    UCI Student Performance Dataset (Mathematics).

    ### Machine Learning Algorithms
    - Linear Regression
    - Decision Tree Regressor
    - Random Forest Regressor

    ### Technologies
    - Python
    - Pandas and NumPy
    - Scikit-learn
    - Streamlit
    - Plotly
    - Joblib

    ### Target Variable
    G3 — Final Mathematics Grade (0–20).

    ### Important Limitation
    The dataset is relatively small and the model's test R² is about
    0.306. Predictions should be treated as estimates, not definitive
    judgments about students.
    """)

    st.markdown("### Dataset Reference")
    st.link_button(
        "Open UCI Student Performance Dataset",
        "https://archive.ics.uci.edu/dataset/320/student+performance"
    )

st.sidebar.divider()
st.sidebar.caption("EduPredict AI | Academic ML Project")
