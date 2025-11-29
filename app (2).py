# app.py 
# Based on original DSS Script from Colab, Optimized for HF

import pandas as pd
import numpy as np
from datetime import datetime
import random
import matplotlib.pyplot as plt
import gradio as gr

# ================================
# 1 — LOAD DATASET (GitHub) 
# ================================
url = "https://raw.githubusercontent.com/Pccgeo-hub/DSS_Employee_Performance/main/Extended_Employee_Performance_and_Productivity_Data.csv"
df = pd.read_csv(url)
print("Dataset loaded:", df.shape)

# ================================
# 2 — CLEANING
# ================================
if 'Hire_Date' in df.columns:
    df['Hire_Date'] = pd.to_datetime(df['Hire_Date'], errors='coerce')

if 'Resigned' in df.columns:
    if df['Resigned'].dtype == bool or df['Resigned'].dtype == np.bool_:
        df['Resigned'] = df['Resigned'].map({True: 1, False: 0})

df.fillna(0, inplace=True)

df.to_csv("cleaned_employee_dataset.csv", index=False)

# ================================
# 3 — FEATURE ENGINEERING
# ================================
df_features = df.copy()

for col in ['Projects_Handled', 'Work_Hours_Per_Week', 'Overtime_Hours']:
    if col in df_features.columns:
        df_features[col] = pd.to_numeric(df_features[col], errors='coerce').fillna(0)

if 'Hire_Date' in df_features.columns:
    now = datetime.now().year
    df_features['Tenure'] = now - df_features['Hire_Date'].dt.year.fillna(now)

if {'Projects_Handled', 'Work_Hours_Per_Week'}.issubset(df_features.columns):
    df_features['Workload_Ratio'] = (
        df_features['Projects_Handled'] /
        df_features['Work_Hours_Per_Week'].replace(0, 1)
    ).round(3)
else:
    df_features['Workload_Ratio'] = 0

if {'Overtime_Hours', 'Work_Hours_Per_Week'}.issubset(df_features.columns):
    df_features['Overtime_Intensity'] = (
        df_features['Overtime_Hours'] /
        df_features['Work_Hours_Per_Week'].replace(0, 1)
    ).round(3)
else:
    df_features['Overtime_Intensity'] = 0

if 'Performance_Score' in df_features.columns:
    df_features['Performance_Level'] = pd.cut(
        df_features['Performance_Score'],
        bins=[0, 2, 4, 5],
        labels=['Low', 'Medium', 'High']
    )

df_features.to_csv("employee_features_dataset.csv", index=False)

# ================================
# 4 — FEEDBACK COMMENT GENERATION
# ================================
positive_comments = [
    "I feel valued and supported.", "Happy with my role.",
    "The company provides great opportunities.",
    "I enjoy my work.", "Supportive team environment."
]
neutral_comments = [
    "My job is okay.", "Some days are challenging.",
    "Workload is manageable.", "Average experience overall."
]
negative_comments = [
    "I feel stressed.", "I feel unsupported.",
    "Workload is overwhelming.", "Poor communication impacts performance."
]

def satisfaction_to_feedback(score):
    try:
        score = float(score)
    except:
        return random.choice(neutral_comments)
    if score >= 4:
        return random.choice(positive_comments)
    elif score >= 3:
        return random.choice(neutral_comments)
    else:
        return random.choice(negative_comments)

df_features['Feedback_Comment'] = df_features['Employee_Satisfaction_Score'].apply(
    satisfaction_to_feedback
)

df_features.to_csv("employee_features_dataset.csv", index=False)

# ================================
# 5 — FINAL LOAD
# ================================
df = pd.read_csv("employee_features_dataset.csv")
df.columns = [c.strip() for c in df.columns]

# ============================================================
# ==========  STEP 5 — AI MODEL COMPONENTS  ==================
# ============================================================

from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import numpy as np

# ------------------------------------------------------------
# Employee ID Column Detection
# ------------------------------------------------------------
def find_employee_id_col(df):
    candidates = ["Employee_ID", "Emp_ID", "ID", "employee_id", "emp_id"]
    for c in candidates:
        if c in df.columns:
            return c
    return None

EMP_ID_COL = find_employee_id_col(df)

# ------------------------------------------------------------
# 1) PREDICTION MODEL — Salary → Performance Score 
# ------------------------------------------------------------
ml_pred_model = None
if 'Monthly_Salary' in df.columns and 'Performance_Score' in df.columns:
    try:
        clean = df[['Monthly_Salary','Performance_Score']].dropna()
        if len(clean) >= 10:
            X = clean[['Monthly_Salary']].values.reshape(-1,1)
            y = clean['Performance_Score'].astype(int).values
            ml_pred_model = RandomForestClassifier(
                n_estimators=300, random_state=42
            )
            ml_pred_model.fit(X,y)
    except Exception:
        ml_pred_model = None

def interpret_score(score):
    mapping = {
        1: "Very Low Performance",
        2: "Low Performance",
        3: "Average Performance",
        4: "High Performance",
        5: "Very High Performance"
    }
    try:
        return mapping.get(int(score), "Unknown")
    except:
        return "Unknown"

def ml_predict_only(salary):
    if ml_pred_model is None:
        return "ML unavailable", "Model not trained"
    try:
        s = float(salary)
    except:
        return "Invalid input", "Invalid salary value"

    try:
        pred = int(ml_pred_model.predict([[s]])[0])
    except:
        return "Error", "Prediction failed"

    label = interpret_score(pred)
    return str(pred), label

# ------------------------------------------------------------
# 2) CLASSIFICATION MODEL — Score → Performance Level
# ------------------------------------------------------------
classification_model = None
if "Performance_Score" in df.columns and "Performance_Level" in df.columns:
    cl = df[['Performance_Score','Performance_Level']].dropna()
    if len(cl) >= 5:
        X = cl[['Performance_Score']].astype(float)
        y = cl['Performance_Level']
        classification_model = RandomForestClassifier(
            n_estimators=200, random_state=42
        )
        classification_model.fit(X,y)

def classification_predict(score):
    try:
        ps = float(score)
    except:
        return "Invalid input"

    if ps < 0 or ps > 5:
        return "Performance Score must be 0–5"

    if classification_model:
        pred = classification_model.predict([[ps]])[0]
        mapping = {
            "High": "High Performer",
            "Medium": "Moderate Performer",
            "Low": "Low Performer"
        }
        return mapping.get(pred, pred)
    else:
        # fallback rule
        if ps <= 2: return "Low Performer"
        if ps == 3: return "Moderate Performer"
        return "High Performer"

# ------------------------------------------------------------
# 3) CLUSTERING MODEL — (Performance, Satisfaction)
# ------------------------------------------------------------
kmeans_model = None
scaler_cl = None
cluster_interpretations = {}

if set(['Performance_Score','Employee_Satisfaction_Score']).issubset(df.columns):
    cl = df[['Performance_Score','Employee_Satisfaction_Score']].dropna()

    if len(cl) >= 4:
        scaler_cl = StandardScaler()
        Xc = scaler_cl.fit_transform(cl.values)

        # For huggingface, keep numeric n_init
        kmeans_model = KMeans(
            n_clusters=4, random_state=42, n_init=10
        )
        kmeans_model.fit(Xc)

        centers = scaler_cl.inverse_transform(kmeans_model.cluster_centers_)
        med_perf = cl['Performance_Score'].median()
        med_sat = cl['Employee_Satisfaction_Score'].median()

        desired_map = {
            ("High","High"): (0, "High Performance", "High Satisfaction", "Top performer, highly motivated."),
            ("Low","Low"):   (1, "Low Performance", "Low Satisfaction", "Underperforming and disengaged."),
            ("Low","High"):  (2, "Low Performance", "High Satisfaction", "Happy but low-performing."),
            ("High","Low"):  (3, "High Performance", "Low Satisfaction", "High performer but unhappy.")
        }

        for cid, ctr in enumerate(centers):
            p,s = float(ctr[0]), float(ctr[1])
            key = (
                "High" if p>=med_perf else "Low",
                "High" if s>=med_sat else "Low"
            )
            if key in desired_map:
                m = desired_map[key]
                cluster_interpretations[cid] = (
                    f"Cluster {m[0]}", m[1], m[2], m[3]
                )
            else:
                cluster_interpretations[cid] = (
                    f"Cluster {cid}", "Mixed", "Mixed", "No interpretation"
                )

def clustering_predict(ps, ss):
    try:
        p=float(ps); s=float(ss)
    except:
        return "Invalid inputs",""

    if p<0 or p>5 or s<0 or s>5:
        return "Inputs must be 0–5", ""

    if kmeans_model is None:
        return "Model unavailable", ""

    arr = scaler_cl.transform([[p,s]])
    cid = int(kmeans_model.predict(arr)[0])

    label, perf, sat, exp = cluster_interpretations.get(
        cid, ("Cluster","Unknown","Unknown","No interpretation")
    )

    return label, f"{perf} – {sat}\n\n{exp}"

# ------------------------------------------------------------
# 4) FORECASTING MODEL — (Aggregate by Job_Title & Hire_Year -> Linear Regression)
# ------------------------------------------------------------
# Prepare Hire Year aggregation similar to Colab script
if 'Hire_Date' in df.columns:
    df['Hire_Date'] = pd.to_datetime(df['Hire_Date'], errors='coerce')
    df = df.dropna(subset=['Hire_Date'])
    df['Hire_Year'] = df['Hire_Date'].dt.year
else:
    df['Hire_Year'] = np.nan

agg_perf = (
    df.groupby(["Job_Title", "Hire_Year"])["Performance_Score"]
    .mean()
    .reset_index()
    .sort_values(["Job_Title", "Hire_Year"])
)

# Available titles 
AVAILABLE_TITLES = ["Manager", "Analyst"]

def jobtitle_performance_forecast(selected_titles, horizon_years):
    """
    selected_titles: list of job title strings
    horizon_years: integer (number of years to forecast)
    Returns: matplotlib Figure, forecast DataFrame, metrics string
    """
    if not selected_titles:
        return None, pd.DataFrame(), "No job title selected."

    fig, ax = plt.subplots(figsize=(10,5))
    metrics_lines = []
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    rows = {}

    for i, title in enumerate(selected_titles):
        sub = agg_perf[agg_perf["Job_Title"] == title].copy()

        if sub.empty:
            metrics_lines.append(f"{title}: no data")
            continue

        years = sub["Hire_Year"].values
        perf = sub["Performance_Score"].values

        # Fit linear model
        X = years.reshape(-1,1)
        y = perf
        lr = LinearRegression().fit(X, y)

        # Historical predictions and metrics
        hist_pred = lr.predict(X)
        mae = mean_absolute_error(y, hist_pred)
        rmse = np.sqrt(mean_squared_error(y, hist_pred))
        r2 = r2_score(y, hist_pred) if len(y) > 1 else np.nan

        metrics_lines.append(f"{title} — MAE: {mae:.4f}  RMSE: {rmse:.4f}  R²: {r2:.4f}")

        # Forecast future years
        last_year = int(sub["Hire_Year"].max())
        future_years = np.arange(last_year+1, last_year+1+int(horizon_years))
        future_pred = lr.predict(future_years.reshape(-1,1))

        # Plot historical and forecast
        c = colors[i % len(colors)]
        ax.plot(years, perf, marker="o", label=f"{title} (historical)", linewidth=2)
        ax.plot(future_years, future_pred, marker="o", linestyle="--", label=f"{title} (forecast)", linewidth=2)

        # Populate table rows
        for yv, pv in zip(future_years, future_pred):
            rows.setdefault(yv, {})[title] = round(float(pv), 4)

    ax.set_title("Job-title Performance — Historical & Forecast")
    ax.set_xlabel("Year")
    ax.set_ylabel("Average Performance Score")
    ax.grid(True)
    ax.legend()

    df_table = pd.DataFrame.from_dict(rows, orient="index")
    df_table.index.name = "Future_Year"
    if not df_table.empty:
        df_table = df_table.reset_index().sort_values("Future_Year")

    metrics_text = "\n".join(metrics_lines) if metrics_lines else "No metrics (no data)."

    return fig, df_table, metrics_text

# ------------------------------------------------------------
# 5) NLP Sentiment 
# ------------------------------------------------------------
analyzer = SentimentIntensityAnalyzer()

def nlp_sentiment(text):
    txt = str(text).strip()
    if not txt:
        return ""
    scores = analyzer.polarity_scores(txt)
    compound = scores.get("compound", 0.0)
    if compound >= 0.6:
        return "Strong Positive Feedback"
    elif compound > 0.2:
        return "Positive Feedback"
    elif compound > -0.2:
        return "Neutral Feedback"
    elif compound > -0.6:
        return "Negative Feedback"
    else:
        return "Strong Negative Feedback"

# ------------------------------------------------------------
# 6) Employee Lookup + HR Insight 
# ------------------------------------------------------------
def employee_lookup(emp_id):
    if EMP_ID_COL is None:
        return "Employee ID column not found in dataset.","",""

    if emp_id is None or str(emp_id).strip()=="":
        return "Please enter Employee ID","",""

    row = df[df[EMP_ID_COL].astype(str)==str(emp_id)]
    if row.empty:
        return "Employee not found","",""

    r = row.iloc[0]

    salary = r.get("Monthly_Salary","N/A")
    perf = r.get("Performance_Score","N/A")
    sat = r.get("Employee_Satisfaction_Score","N/A")
    tenure = r.get("Years_At_Company","N/A")
    comment = r.get("Feedback_Comment","")

    try:
        perf_val=float(perf)
    except:
        perf_val=None
    if perf_val is None: perf_cat="Unknown"
    elif perf_val<=2:   perf_cat="Low"
    elif perf_val==3:   perf_cat="Medium"
    else:               perf_cat="High"

    try:
        sat_val=float(sat)
    except:
        sat_val=None
    if sat_val is None: sat_cat="Unknown"
    elif sat_val<=2:    sat_cat="Low"
    elif sat_val==3:    sat_cat="Medium"
    else:               sat_cat="High"

    risks=[]
    recs=[]

    # ---- HIGH PERFORMANCE ----
    if perf_cat=="High":
        if sat_cat=="High":
            recs+=[
                "Reward and recognize high contribution",
                "Provide career growth opportunities",
                "Maintain engagement and morale"
            ]
        elif sat_cat=="Medium":
            risks+=["Engagement drift"]
            recs+=[
                "Maintain recognition",
                "Check workload and role alignment",
                "Provide regular check-ins"
            ]
        else:  # low sat
            risks+=["Retention risk"]
            recs+=[
                "Immediate check-in required",
                "Review role, workload, or manager alignment",
                "Monitor for burnout"
            ]

    # ---- MEDIUM PERFORMANCE ----
    elif perf_cat=="Medium":
        if sat_cat=="High":
            recs+=[
                "Provide targeted training",
                "Support skill development",
                "Encourage continued engagement"
            ]
        elif sat_cat=="Medium":
            risks+=["Performance stagnation"]
            recs+=[
                "Set measurable performance goals",
                "Provide coaching and periodic feedback",
                "Monitor progress consistently"
            ]
        else: # low sat
            risks+=["Motivation decline"]
            recs+=[
                "Identify causes of dissatisfaction",
                "Provide support with workload/role clarity",
                "Increase engagement through recognition",
                "Have follow-up check-in sessions"
            ]

    # ---- LOW PERFORMANCE ----
    elif perf_cat=="Low":
        if sat_cat=="High":
            risks+=["Performance gap"]
            recs+=[
                "Provide structured coaching",
                "Implement a training plan",
                "Monitor goal progress"
            ]
        elif sat_cat=="Medium":
            risks+=["Underperformance"]
            recs+=[
                "Implement a performance improvement plan",
                "Schedule weekly progress reviews",
                "Provide management support"
            ]
        else:
            risks+=["High risk of resignation"]
            recs+=[
                "Immediate HR intervention required",
                "Resolve dissatisfaction causes",
                "Evaluate role fit or consider reassignment"
            ]

    risks_section="None" if not risks else "\n".join(f"- {r}" for r in risks)
    recs_section="No immediate recommendations" if not recs else "\n".join(f"- {r}" for r in recs)

    hr_text=f"""Risks:
{risks_section}

Recommendations:
{recs_section}"""

    summary=f"""Employee ID: {emp_id}
Monthly Salary: {salary}
Performance Score: {perf} ({perf_cat} Performance)
Satisfaction Score: {sat} ({sat_cat} Satisfaction)
Years at Company: {tenure}
Feedback comment: {comment if comment else 'N/A'}"""

    return "Found", summary, hr_text

# ============================================================
# ==========  STEP 6 — UI STYLING + SALARY BOUNDS  ===========
# ============================================================

custom_css = """
.sidebar-panel {background:#F3F4F6;padding:22px;border-radius:12px;border:1px solid #E5E7EB;}
.section-card {background:#FFF;padding:20px;border-radius:12px;border:1px solid rgba(0,0,0,0.05);box-shadow:0 6px 18px rgba(0,0,0,0.06);margin-bottom:18px;}
.section-title {font-size:20px;font-weight:600;margin-bottom:10px;color:#111827;}
.small-note {font-size:13px;color:#374151;margin-top:8px;margin-bottom:8px;}
"""

if "Monthly_Salary" in df.columns and df['Monthly_Salary'].dropna().shape[0] > 0:
    salary_min = float(df['Monthly_Salary'].min())
    salary_max = float(df['Monthly_Salary'].max())
    salary_median = float(df['Monthly_Salary'].median())
else:
    salary_min, salary_max, salary_median = 3800.0, 9000.0, 6000.0


# ============================================================
# ==========  STEP 7 — BUILD GRADIO INTERFACE  ===============
# ============================================================

import gradio as gr

with gr.Blocks(title="DSS for Performance Evaluation") as demo:

    custom_css = """
:root {
    --bg: #f2f3f5;
    --card-bg: #ffffff;
    --card-border: #dcdfe4;
    --text-dark: #111827;
    --text-soft: #374151;
    --accent: #ff8c42;
}

body {
    background: var(--bg);
}

.sidebar-panel {
    background: var(--card-bg);
    padding: 20px;
    border-radius: 12px;
    border: 1px solid var(--card-border);
}

.section-card {
    background: var(--card-bg);
    padding: 20px;
    border-radius: 12px;
    border: 1px solid var(--card-border);
    box-shadow: 0px 4px 10px rgba(0,0,0,0.05);
    margin-bottom: 18px;
}

.section-title {
    font-size: 20px;
    font-weight: 600;
    color: var(--text-dark);
    margin-bottom: 10px;
}

.small-note {
    font-size: 13px;
    color: var(--text-soft);
}

/* Style radio + buttons */
input[type="radio"] + label {
    font-weight: 500;
}

/* Accent color on sliders */
input[type="range"] {
    accent-color: var(--accent);
}
"""


    gr.HTML(f"<style>{custom_css}</style>")

    # Header
    with gr.Row():
        gr.Markdown("<h2 style='margin:0px'>Decision Support System for Performance Evaluation</h2>")
        gr.Markdown("<div style='text-align:right;color:#6c757d'>Version 1.0</div>")

    # Main area with tabs instead of complicated show/hide
    with gr.Tabs():

        # -------------------------------------------------------
        # PREDICTION
        # -------------------------------------------------------
        with gr.Tab("Prediction"):
            with gr.Column(elem_classes="section-card"):
                gr.Markdown("<div class='section-title'>Prediction — Predict Employee Performance Based on Monthly Salary</div>")
                salary_slider = gr.Slider(minimum=salary_min, maximum=salary_max, step=1, value=salary_median, label="Monthly Salary")
                pred_ml_score = gr.Textbox(label="ML Predicted Score", interactive=False)
                pred_ml_label = gr.Textbox(label="ML Interpretation", interactive=False)
                gr.Button("Predict (ML)").click(ml_predict_only, salary_slider, [pred_ml_score, pred_ml_label])

        # -------------------------------------------------------
        # CLASSIFICATION
        # -------------------------------------------------------
        with gr.Tab("Classification"):
            with gr.Column(elem_classes="section-card"):
                gr.Markdown("<div class='section-title'>Classification — Classify Employees Based on performance Score</div>")
                default_perf = int(df["Performance_Score"].median()) if "Performance_Score" in df.columns else 3
                perf_slider = gr.Slider(minimum=0, maximum=5, step=1, value=default_perf, label="Performance Score")
                clf_out = gr.Textbox(label="Predicted Level", interactive=False)
                gr.Button("Classify").click(classification_predict, perf_slider, clf_out)

        # -------------------------------------------------------
        # CLUSTERING
        # -------------------------------------------------------
        with gr.Tab("Clustering"):
            with gr.Column(elem_classes="section-card"):
                gr.Markdown("<div class='section-title'>Clustering — Group Employees with Similar Performance & Satisfaction Attributes</div>")
                perf_cl = gr.Slider(minimum=0, maximum=5, step=1, label="Performance Score")
                sat_cl = gr.Slider(minimum=0, maximum=5, step=1, label="Satisfaction Score")
                cid = gr.Textbox(label="Cluster", interactive=False)
                cint = gr.Textbox(label="Cluster Interpretation", interactive=False)
                gr.Button("Cluster").click(clustering_predict, [perf_cl, sat_cl], [cid, cint])

        # -------------------------------------------------------
        # FORECASTING
        # -------------------------------------------------------
        with gr.Tab("Forecasting"):
            with gr.Column(elem_classes="section-card"):
                gr.Markdown("<div class='section-title'>Forecasting — Historical performance trends for two job titles: Manager & Analyst</div>")
                title_select = gr.CheckboxGroup(choices=AVAILABLE_TITLES, value=AVAILABLE_TITLES, label="Job Titles")
                horizon = gr.Slider(1, 5, step=1, value=3, label="Forecast Horizon (years)")
                fc_plot = gr.Plot()
                fc_table = gr.Dataframe()
                fc_metrics = gr.Textbox(interactive=False)
                gr.Button("Run Forecast").click(jobtitle_performance_forecast, [title_select, horizon], [fc_plot, fc_table, fc_metrics])

        # -------------------------------------------------------
        # NLP
        # -------------------------------------------------------
        with gr.Tab("NLP"):
            with gr.Column(elem_classes="section-card"):
                gr.Markdown("<div class='section-title'>NLP — Feedback Sentiment</div>")
                fb = gr.Textbox(label="Feedback", lines=5)
                nlp_out = gr.Textbox(label="Sentiment Interpretation", interactive=False)
                gr.Button("Analyze").click(nlp_sentiment, fb, nlp_out)

        # -------------------------------------------------------
        # EMPLOYEE LOOKUP
        # -------------------------------------------------------
        with gr.Tab("Employee Lookup"):
            with gr.Column(elem_classes="section-card"):
                gr.Markdown("<div class='section-title'>Employee Lookup & HR Insight</div>")
                emp_id_in = gr.Textbox(label="Employee ID")
                lookup_status = gr.Textbox(interactive=False, label="Status")
                emp_summary = gr.TextArea(lines=8, label="Employee Summary")
                emp_hr = gr.TextArea(lines=8, label="HR Insight & Recommendations")
                gr.Button("Lookup").click(employee_lookup, emp_id_in, [lookup_status, emp_summary, emp_hr])




# ============================================================
# ==========  STEP 8 — LAUNCH THE GRADIO APP  =================
# ============================================================
if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )
