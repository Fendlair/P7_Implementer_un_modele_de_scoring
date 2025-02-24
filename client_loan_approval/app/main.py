#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 10:37:21 2025

@author: etienne
"""
import os
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field, ValidationError
from .model import load_model
import shap
import logging

# Configurer le logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# PATH to the model.pkl
pipeline_path = "../model/model.pkl"

# Loading the model
pipeline = load_model(pipeline_path)

# Extract classification model
model = pipeline.named_steps["model"]

# Shap explainer
feature_names = ['DAYS_BIRTH', 'DAYS_ID_PUBLISH', 'EXT_SOURCE_2', 'AMT_REQ_CREDIT_BUREAU_QRT', 'AMT_REQ_CREDIT_BUREAU_YEAR', 'NAME_FAMILY_STATUS_Single / not married', 'WEEKDAY_APPR_PROCESS_START_MONDAY', 'BURO_DAYS_CREDIT_MIN', 'BURO_DAYS_CREDIT_MAX', 'BURO_CREDIT_DAY_OVERDUE_MEAN', 'BURO_CNT_CREDIT_PROLONG_SUM', 'BURO_CREDIT_TYPE_Microloan_MEAN', 'BURO_STATUS_0_MEAN_MEAN', 'PREV_DAYS_DECISION_MAX', 'PREV_CNT_PAYMENT_MEAN']
explainer = shap.Explainer(model, feature_names=feature_names)

# Application FastAPI
app = FastAPI()

# Data shape for prediction
class PredictionRequest(BaseModel):
	features: list = Field(..., min_items=15, max_items=15,
	                       description="List of 15 informations about the client.")

# Data for answer
class PredictionResponse(BaseModel):
    probability: float
    shap_values: list

# End point for prediction
@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    try:
        # Convertion data into float
        features = [float(feature) for feature in request.features]
        # Make prediction
        prediction_proba = pipeline.predict_proba([features])[0]
        # Shap values calculation
        shap_values = explainer(pipeline.named_steps["scaler"].transform([features]))
        shap_values_list = shap_values.values[0].tolist()

        return PredictionResponse(probability=prediction_proba[1], shap_values=shap_values_list)

    except ValueError as e:
        logger.error(f"Conversion error: {e}")
        raise HTTPException(status_code=422, detail='Error, imput data must been numerical')
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail="Error during prediction")

# Function to return a description of the app
def get_app_description():
	return (
    	"Welcome to the Client loan approval API!"
    	"This API allows you to determine if we should allow a credit to a client based on some data."
    	"Use the '/predict/' endpoint with a POST request to make predictions."
    	"Example usage: POST to '/predict/' with JSON data containing"
	)

# Define the root endpoint to return the app description
@app.get("/")
async def root():
	return {"message": get_app_description()}

# Middleware pour gérer les erreurs de validation
@app.exception_handler(ValidationError)
async def validation_exception_handler(request: Request, exc: ValidationError):
    logger.error(f"Erreur de validation: {exc}")
    return JSONResponse(status_code=422, content={"detail": exc.errors()})

if __name__== "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
