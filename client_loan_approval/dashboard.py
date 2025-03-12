import pandas as pd
import streamlit as st
import streamviz
import numpy as np
import matplotlib.pyplot as plt
import requests
import json

# FONCTIONS
def request_prediction(model_uri, data):
    """ "Do the prediction
    data = list of features
    return a float as prediction and list of shap values for the client"""
    headers = {"Content-Type": "application/json"}
    data_json = {"features": data}
    response = requests.post(
        model_uri, headers=headers, data=json.dumps(data_json)
    )

    if response.status_code == 200:
        return response.json()
    else:
        st.error(
            f"Request failed with status: {response.status_code} - ||{response.text}"
        )
        return None

@st.cache_data
def load_data():
    """Load the model global shap values"""
    global_shap_values = pd.read_csv("Data/global_shap_values.csv")
    return global_shap_values

@st.cache_data
def load_description():
    """Load description of the features"""
    description = pd.read_csv("Data/feature_description.csv")
    description = description.set_index("Feature").to_dict()["Description"]
    return description

@st.cache_data
def read_clients_data():
    clients_data = pd.read_csv("Data/X_test_final.csv")
    clients_y = pd.read_csv("Data/y_test_final.csv")
    clients_y.rename(columns={"0": "Target"}, inplace=True)
    clients_data = clients_data.merge(
        clients_y, left_index=True, right_index=True
    )
    return clients_data

# Initialization
if "features" not in st.session_state:
    st.session_state["features"] = []

if "pred" not in st.session_state:
    st.session_state["pred"] = {
        "probability": 1,
        "shap_values": [0.] * 15,
    }

if "shap_table" not in st.session_state:
    st.session_state["shap_table"] = pd.DataFrame()

# Variables
feature_names = [
    "DAYS_BIRTH",
    "DAYS_ID_PUBLISH",
    "EXT_SOURCE_2",
    "AMT_REQ_CREDIT_BUREAU_QRT",
    "AMT_REQ_CREDIT_BUREAU_YEAR",
    "NAME_FAMILY_STATUS_Single / not married",
    "WEEKDAY_APPR_PROCESS_START_MONDAY",
    "BURO_DAYS_CREDIT_MIN",
    "BURO_DAYS_CREDIT_MAX",
    "BURO_CREDIT_DAY_OVERDUE_MEAN",
    "BURO_CNT_CREDIT_PROLONG_SUM",
    "BURO_CREDIT_TYPE_Microloan_MEAN",
    "BURO_STATUS_0_MEAN_MEAN",
    "PREV_DAYS_DECISION_MAX",
    "PREV_CNT_PAYMENT_MEAN",
]

predicted_class = 0

# API adress
FASTAPI_URI = "https://p7-implementer-un-modele-de-scoring.onrender.com/predict/"

# Contrast ratio for the text
# 

# App
def main():
    st.image("Data/banner.png")
    st.divider()
    st.title("Welcome to the Credit Approval Prediction Dashboard")
    st.write("This dashboard is designed to assist both clients and client advisors in evaluating the likelihood of loan approval based on personal and financial information. Using advanced machine learning techniques, we provide clear and understandable insights to support your decisions.")
    st.divider()
    st.header("1. Enter Information:")
    st.write("Fill in the fields with the relevant information. Each field represents an important characteristic for credit evaluation.")
    st.header("2. Prediction:")
    st.write("Click the 'Predict' button to obtain an estimate of the loan approval probability.")
    st.header("3. Analyse Results:")
    st.write("Explore the importance of each characteristic in the loan decision process. You can visualise your position in comparaison with other clients.")
    st.divider()
    description = load_description()
    st.sidebar.title("Feature description:")
    describ = st.sidebar.selectbox("", feature_names)
    st.sidebar.write(description[describ])
    st.sidebar.title("Feature of interest selection:")

    st.subheader(description["DAYS_BIRTH"]+ ":")
    day_birth = st.number_input(
        "DAYS_BIRTH", max_value=0, value=-16000, step=1
    )

    st.subheader(description["DAYS_ID_PUBLISH"]+ ":")
    days_id_publish = st.number_input(
        "DAYS_ID_PUBLISH", max_value=0, value=-4500, step=1
    )

    st.subheader(description["EXT_SOURCE_2"]+ ":")
    ext_source_2 = st.number_input(
        "EXT_SOURCE_2", min_value=0.0, max_value=1.0, value=0.5, step=0.01
    )

    st.subheader(description["AMT_REQ_CREDIT_BUREAU_QRT"]+ ":")
    amt_req_credit_bureau_qtr = st.number_input(
        "AMT_REQ_CREDIT_BUREAU_QRT", min_value=0, value=100, step=1
    )

    st.subheader(description["AMT_REQ_CREDIT_BUREAU_YEAR"]+ ":")
    amt_req_credit_bureau_year = st.number_input(
        "AMT_REQ_CREDIT_BUREAU_YEAR", min_value=0, value=0, step=1
    )

    st.subheader(description["NAME_FAMILY_STATUS_Single / not married"]+ ":")
    name_family_status = st.selectbox(
        "NAME_FAMILY_STATUS_Single / not married", ["Yes", "No"]
    )
    name_family_status = 1 if name_family_status == "Yes" else 0

    st.subheader(description["WEEKDAY_APPR_PROCESS_START_MONDAY"]+ ":")
    weekday_appr_process_start_monday = st.selectbox(
        "WEEKDAY_APPR_PROCESS_START_MONDAY", ["No", "Yes"]
    )
    weekday_appr_process_start_monday = (
            1 if weekday_appr_process_start_monday == "Yes" else 0
        )

    st.subheader(description["BURO_DAYS_CREDIT_MIN"]+ ":")
    buro_days_credit_min = st.number_input(
        "BURO_DAYS_CREDIT_MIN", max_value=0, value=0, step=1
    )

    st.subheader(description["BURO_DAYS_CREDIT_MAX"]+ ":")
    buro_days_credit_max = st.number_input(
        "BURO_DAYS_CREDIT_MAX", max_value=0, value=0, step=1
    )

    st.subheader(description["BURO_CREDIT_DAY_OVERDUE_MEAN"]+ ":")
    buro_credit_day_overdue_mean = st.number_input(
        "BURO_CREDIT_DAY_OVERDUE_MEAN", min_value=0, value=0, step=1
    )

    st.subheader(description["BURO_CNT_CREDIT_PROLONG_SUM"]+ ":")
    buro_cnt_credit_prolong_sum = st.number_input(
        "BURO_CNT_CREDIT_PROLONG_SUM", min_value=0, value=0, step=1
    )

    st.subheader(description["BURO_CREDIT_TYPE_Microloan_MEAN"]+ ":")
    buro_credit_type_microloan_mean = st.number_input(
        "BURO_CREDIT_TYPE_Microloan_MEAN", min_value=0.0, value=0.0, step=0.25
    )

    st.subheader(description["BURO_STATUS_0_MEAN_MEAN"]+ ":")
    buro_status_0_mean_mean = st.number_input(
        "BURO_STATUS_0_MEAN_MEAN", min_value=0.0, value=1.0, step=1.0
    )

    st.subheader(description["PREV_DAYS_DECISION_MAX"]+ ":")
    prev_days_decision_max = st.number_input(
        "PREV_DAYS_DECISION_MAX", max_value=0, value=0, step=1
    )

    st.subheader(description["PREV_CNT_PAYMENT_MEAN"]+ ":")
    prev_cnt_payment_mean = st.number_input(
        "PREV_CNT_PAYMENT_MEAN", min_value=0.0, value=10.0, step=1.0
    )

    st.divider()
    st.title("Client prediction")
    if st.button("Predict"):
        st.session_state["features"] = [
            day_birth,
            days_id_publish,
            ext_source_2,
            amt_req_credit_bureau_qtr,
            amt_req_credit_bureau_year,
            name_family_status,
            weekday_appr_process_start_monday,
            buro_days_credit_min,
            buro_days_credit_max,
            buro_credit_day_overdue_mean,
            buro_cnt_credit_prolong_sum,
            buro_credit_type_microloan_mean,
            buro_status_0_mean_mean,
            prev_days_decision_max,
            prev_cnt_payment_mean,
        ]

        st.session_state["pred"] = request_prediction(
            FASTAPI_URI, st.session_state["features"]
        )
        
        
    
    if st.session_state["pred"]:
        threshold = 0.4
        predicted_class = (
        1 if st.session_state["pred"]["probability"] > threshold else 0
         )

    # Display a gauge indicating the score of the client and if the loan is accepted or not
    streamviz.gauge(
        1 - round(st.session_state["pred"]["probability"], 3),
        gTitle="Probability that the client will pay is loan (Threshold = 60 %)",
        grLow=0.55,
        grMid=0.6,
        sFix="%",
        gSize="MED",
    )
    # Text version of the gauge in order for blind people to receive the information about the probability
    st.write("The probability that the client is going to reimburse the loan is: " + str(1 - round(st.session_state["pred"]["probability"], 3) ))
    if predicted_class == 0:
        st.success("Based on the data, the loan is ACCEPTED")
    else:
        st.warning("Based on the data, the loan is REFUSED")

    # Displaying advantage and issues with client application
    st.divider()
    st.title("Pros and cons about the client informations")
    shap_values = np.array(st.session_state["pred"]["shap_values"])
    shap_values = zip(feature_names, shap_values)

    st.session_state["shap_table"] = pd.DataFrame(
            shap_values, columns=["Feature", "Client value"]
        )
    st.session_state["shap_table"] = st.session_state[
            "shap_table"
        ].sort_values(by="Client value")
    st.info("The pieces of information that are good for the client are:")
    st.dataframe(st.session_state["shap_table"].head(3))
    st.warning(
            "The pieces of information that are not good for the client are:"
        )
    st.dataframe(
            st.session_state["shap_table"]
            .sort_values(by="Client value", ascending=False)
            .head(3)
        )

    # Connect feature name and variable name with a dictionnaire
    feature_dict = dict(zip(feature_names, st.session_state["features"]))

    st.divider()
    st.title("Strengths and Weaknesses in client informations regarding the prediction model")
    
    # Histogram plot of shap values
    global_shap_values = load_data()
    shap_data = global_shap_values.merge(
        st.session_state["shap_table"], on="Feature"
    )
    shap_data = shap_data.sort_values("Model value", ascending=False)
    shap_data["Client value"] = shap_data["Client value"].abs()
    
    # data_shap = global_shap_values.
    fig, ax = plt.subplots()
    ax.hlines(
        y=shap_data["Feature"],
        xmin=shap_data["Model value"],
        xmax=shap_data["Client value"],
        color="grey",
        zorder=0,
    )
    ax.scatter(
        shap_data["Model value"],
        shap_data["Feature"],
        color="#06006c",
        alpha=1,
        s=30,
        label="Model",
    )
    ax.scatter(
        shap_data["Client value"],
        shap_data["Feature"],
        color="#ff5876",
        alpha=1,
        s=50,
        label="Client",
        marker="+",
    )
    ax.set_title("Features importance for global Prediction and Client's prediction", fontsize=20, pad=10)
    ax.set_xlabel("Feature importance", fontsize=16)
    ax.legend(fontsize=16)
    st.pyplot(fig)

    # Data exploration by the client
    st.divider()
    st.title("Position of the client regarding other clients")
    clients_data = read_clients_data()

    # Connect feature name and variable name with a dictionnaire
    feature_dict = dict(zip(feature_names, st.session_state["features"]))

    # Feature selection
    feature1 = st.sidebar.selectbox("Feature 1", feature_names)
    feature2 = st.sidebar.selectbox("Feature 2", [f for f in feature_names if f != feature1])

    # colors of clients depending of target
    colors = {0: "#06006c", 1: "#ffb24a"}
    labels = {0: "Responsible borrowers", 1: "challenged customers"}
    pattern = {0: "x", 1: "/"}
    
    # Display selected feature and how the client is compare to others
    fig2, ax2 = plt.subplots()
    for customer_type in clients_data["Target"].unique():
        subset = clients_data[clients_data["Target"] == customer_type]
        ax2.hist(subset[feature1], bins=20, color=colors[customer_type], label=labels[customer_type], hatch=pattern[customer_type])
    ax2.axvline(
        x=feature_dict[feature1], color="#ff5876", linewidth=3, linestyle="--"
    )
    kwargs={"color": "#ff5876",
            "size": "large",
            "weight": "bold",
            }
    ax2.annotate("Client", xy=(feature_dict[feature1], ax2.axis()[3]), **kwargs)
    ax2.set_title("Distribution of " + feature1, fontsize=20, pad=20)
    ax2.set_xlabel(feature1, fontsize=16)
    ax2.set_ylabel("Number of clients", fontsize=16)
    ax2.legend(fontsize=14)
    st.pyplot(fig2)

    # Display selected feature and how the client is compare to others in two dimensions
    st.divider()
    st.title("Position of the client regarding other clients along two features")
    fig3, ax3 = plt.subplots()
    markers = {0: "o", 1:"+"}
    for customer_type in clients_data["Target"].unique():
        subset = clients_data[clients_data["Target"] == customer_type]
        ax3.scatter(subset[feature1], subset[feature2], alpha= 0.5, s=20, color=colors[customer_type], label=labels[customer_type], marker=markers[customer_type])
    ax3.plot(feature_dict[feature1], feature_dict[feature2], marker="o", markersize=10, color="#ff5876", label="Client")
    ax3.set_title("Scatter plot of " + feature1 + " versus " + feature2, fontsize=20, pad=10)
    ax3.set_xlabel(feature1, fontsize=16)
    ax3.set_ylabel(feature2, fontsize=16)
    ax3.legend(fontsize=14)
    st.pyplot(fig3)

    st.divider()
    st.header("Disclaimer:")
    st.write(
        "The predictions provided by this dashboard are based on a machine learning model trained on historical data. Although these predictions are accurate, they do not replace a comprehensive evaluation by a professional."
    )
    st.write(
        "This dashboard does not commit you to granting a loan. The insights provided are intended to support your decision-making process but should not be the sole basis for credit approval."
    )

if __name__ == "__main__":
    main()
