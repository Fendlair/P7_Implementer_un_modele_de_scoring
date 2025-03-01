import pandas as pd
import streamlit as st
import requests
import json

feature_names = ['DAYS_BIRTH', 'DAYS_ID_PUBLISH', 'EXT_SOURCE_2', 'AMT_REQ_CREDIT_BUREAU_QRT', 'AMT_REQ_CREDIT_BUREAU_YEAR', 'NAME_FAMILY_STATUS_Single / not married', 'WEEKDAY_APPR_PROCESS_START_MONDAY', 'BURO_DAYS_CREDIT_MIN', 'BURO_DAYS_CREDIT_MAX', 'BURO_CREDIT_DAY_OVERDUE_MEAN', 'BURO_CNT_CREDIT_PROLONG_SUM', 'BURO_CREDIT_TYPE_Microloan_MEAN', 'BURO_STATUS_0_MEAN_MEAN', 'PREV_DAYS_DECISION_MAX', 'PREV_CNT_PAYMENT_MEAN']

def request_prediction(model_uri, data):
    headers = {"Content-Type": "application/json"}
    data_json = {"features": data}
    response = requests.post(model_uri, headers=headers, data=json.dumps(data_json))

    if response.status_code == 200:
        return response.json()
    else:
        st.error(f"Request failed with status: {response.status_code} - ||{response.text}")
        return None

def main():
    FASTAPI_URI = "http://p7-implementer-un-modele-de-scoring.onrender.com/predict/"

    st.image("banner.png")
    st.title("Client Loan Approval")

    day_birth = st.number_input("Client's age in days at the time of application: DAYS_BIRTH", max_value=0, value=-15000, step=1)

    days_id_publish = st.number_input("How many days before the application did client change the identity document with which he applied for the loan: DAYS_ID_PUBLISH", max_value=0, value=-2500, step=1)

    ext_source_2 = st.number_input("Normalized score from external data source 2: EXT_SOURCE_2", min_value=0., max_value=1., value=0.5, step=0.01)

    amt_req_credit_bureau_qtr = st.number_input("Number of enquiries to Credit Bureau about the client 3 month before application (excluding one month before application): AMT_REQ_CREDIT_BUREAU_QRT", min_value=0, value=200, step=1)

    amt_req_credit_bureau_year = st.number_input("Number of enquiries to Credit Bureau about the client one day year (excluding last 3 months before application): AMT_REQ_CREDIT_BUREAU_YEAR", min_value=0, value=10, step=1)

    name_family_status = st.selectbox(
        "Family status: NAME_FAMILY_STATUS_Single / not married", ["Single", "Married"])

    weekday_appr_process_start_monday = st.selectbox("Did the client previous application was on a Monday: WEEKDAY_APPR_PROCESS_START_MONDAY", ["No", "Yes"])

    buro_days_credit_min = st.number_input("How many days before current application did client apply for Credit Bureau credit for the last time: BURO_DAYS_CREDIT_MIN",
                                max_value=0, value=0, step=1)

    buro_days_credit_max = st.number_input("How many days before current application did client apply for Credit Bureau credit for the first time: BURO_DAYS_CREDIT_MAX",
                                max_value=0, value=0, step=1)
    
    buro_credit_day_overdue_mean = st.number_input("Number of mean days past due on CB credit at the time of application for related loan in our sample: BURO_CREDIT_DAY_OVERDUE_MEAN",
                                min_value=0, value=0, step=1)
    
    buro_cnt_credit_prolong_sum = st.number_input("How many times was the Credit Bureau credits prolonged: BURO_CNT_CREDIT_PROLONG_SUM",
                                min_value=0, value=0, step=1)

    buro_credit_type_microloan_mean = st.number_input("Mean of previous micro loan: BURO_CREDIT_TYPE_Microloan_MEAN",
                                min_value=0., value=0., step=.25)

    buro_status_0_mean_mean = st.number_input("Mean of status of Credit Bureau loan during the month: BURO_STATUS_0_MEAN_MEAN",
                                min_value=0., value=0., step=1.)

    prev_days_decision_max = st.number_input("Relative to current application when was the decision about previous application made: PREV_DAYS_DECISION_MAX",
                                max_value=0, value=0, step=1)

    prev_cnt_payment_mean  = st.number_input("Mean of term of previous credit at application of the previous application: PREV_CNT_PAYMENT_MEAN",
                                min_value=0., value=0., step=1.)

    if st.button("Predict"):
        name_family_status = 0 if name_family_status == "Married" else 1
        weekday_appr_process_start_monday = 1 if weekday_appr_process_start_monday == "Yes" else 0
    
        features = [day_birth, days_id_publish, ext_source_2, amt_req_credit_bureau_qtr, amt_req_credit_bureau_year, name_family_status, weekday_appr_process_start_monday, buro_days_credit_min, buro_days_credit_max, buro_credit_day_overdue_mean, buro_cnt_credit_prolong_sum, buro_credit_type_microloan_mean, buro_status_0_mean_mean, prev_days_decision_max, prev_cnt_payment_mean]
    
        pred = None
        pred = request_prediction(FASTAPI_URI, features)
        if pred:
            st.info(f"The probability that the client will not repay their loan is: {pred['probability']:.2f}")
            threshold = 0.4
            predicted_class = 1 if pred["probability"] > threshold else 0
            if predicted_class == 0:
                    st.success("Based on the data; the loan is ACCEPTED")
            else:
                    st.warning("Based on the data, the loan is REFUSED")
            # Displaying advantage and issuues with client application
            shap_values = zip(feature_names, pred["shap_values"])
            shap_table = pd.DataFrame(shap_values, columns=["Information", "Weight"])
            shap_table = shap_table.sort_values(by="Weight")
            st.info("The pieces of information that are good for the client are:")
            st.dataframe(shap_table.head(3))
            st.info("The pieces of information that are not good for the client are:")
            st.dataframe(shap_table.sort_values(by="Weight", ascending=False).head(3))

if __name__ == "__main__":
    main()
