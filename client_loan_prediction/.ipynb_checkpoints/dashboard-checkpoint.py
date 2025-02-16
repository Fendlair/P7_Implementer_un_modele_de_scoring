import pandas as pd
import streamlit as st
import requests
import json


def request_prediction(model_uri, data):
    headers = {"Content-Type": "application/json"}
    data_json = {"features": data}
    response = requests.post(model_uri, headers=headers, data=json.dumps(data_json))

    if response.status_code == 200:
        return response.json()
    else:
        st.error(f"Request failed with status: {response.status_code} - ||{response.text}")
        return None
st.image("banner.png")
def main():
    FASTAPI_URI = 'http://127.0.0.1:8000/predict/'

    st.title('Client Loan Approval')

    day_birth = st.number_input("Client's age in days at the time of application",
                                 max_value=0, value=-15000, step=1)

    days_id_publish = st.number_input('How many days before the application did client change the identity document with which he applied for the loan',
                              max_value=0, value=-2500, step=1)

    ext_source_2 = st.number_input('Normalized score from external data source 2',
                                   min_value=0., max_value=1., value=0.5, step=0.01)

    amt_req_credit_bureau_qtr = st.number_input('Number of enquiries to Credit Bureau about the client 3 month before application (excluding one month before application)',
                                     min_value=0, value=200, step=1)

    amt_req_credit_bureau_year = st.number_input('Number of enquiries to Credit Bureau about the client one day year (excluding last 3 months before application)',
                                min_value=0, value=10, step=1)

    name_family_status = st.selectbox(
        'Family status', ['Single', 'Married'])

    weekday_appr_process_start_monday = st.number_input('On which day of the week did the client apply for previous application',
                                min_value=0, value=0, step=1)

    buro_days_credit_min = st.number_input('How many days before current application did client apply for Credit Bureau credit, Minimum',
                                max_value=0, value=0, step=1)


    buro_days_credit_max = st.number_input('How many days before current application did client apply for Credit Bureau credit, Maximum',
                                max_value=0, value=0, step=1)
    
    buro_credit_day_overdue_mean = st.number_input('Number of days past due on CB credit at the time of application for related loan in our sample, moyenne',
                                min_value=0, value=0, step=1)
    
    buro_cnt_credit_prolong_sum = st.number_input('How many times was the Credit Bureau credit prolonged',
                                min_value=0, value=0, step=1)

    buro_credit_type_microloan_mean = st.number_input('Mean of previous micro loan',
                                min_value=0., value=0., step=.25)

    buro_status_0_mean_mean = st.number_input('Status of Credit Bureau loan during the month (active, closed, DPD0-30, [C means closed, X means status unknown, 0 means no DPD, 1 means maximal did during month between 1-30, 2 means DPD 31-60, 5 means DPD 120+ or sold or written off ] )How many times was the Credit Bureau credit prolonged',
                                min_value=0., value=0., step=1.)

    prev_days_decision_max = st.number_input('Relative to current application when was the decision about previous application made',
                                max_value=0, value=0, step=1)

    prev_cnt_payment_mean  = st.number_input('Term of previous credit at application of the previous application',
                                min_value=0, value=0, step=1)

    if st.button("Predict"):
        if name_family_status == "Married":
            name_family_status = 0
        else:
            name_family_status = 1
    
        features = [day_birth, days_id_publish, ext_source_2, amt_req_credit_bureau_qtr, amt_req_credit_bureau_year, name_family_status, weekday_appr_process_start_monday, buro_days_credit_min, buro_days_credit_max, buro_credit_day_overdue_mean, buro_cnt_credit_prolong_sum, buro_credit_type_microloan_mean, buro_status_0_mean_mean, prev_days_decision_max, prev_cnt_payment_mean]
    
        pred = None
        pred = request_prediction(FASTAPI_URI, features)
        if pred:
            st.info(f"La probabilité que le client rembourse sont prêt est de : {1 - pred['probability']:.2f}")
            threshold = 0.4
            predicted_class = 1 if pred["probability"] >= threshold else 0
            if predicted_class == 0:
                    st.success(f"Le pret pour le client est ACCEPTE")
            else:
                    st.warning(f"Le prêt pour le client est REFUSE")
                    
if __name__ == '__main__':
    main()
