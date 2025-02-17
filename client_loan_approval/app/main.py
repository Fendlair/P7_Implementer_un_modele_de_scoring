#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 10:37:21 2025

@author: etienne
"""

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field, ValidationError
from .model import load_model

# PATH to the model.pkl
model_path = "../model/model.pkl"

# Loading the model
model = load_model(model_path)

# Application FastAPI
app = FastAPI()

# Data shape for prediction
class PredictionRequest(BaseModel):
	features: list = Field(..., min_items=15, max_items=15,
	                       description="List of 15 informations about the client.")

# Data for answer
class PredictionResponse(BaseModel):
    probability: float

# End point for prediction
@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    try:
        # Faire la prédiction
        prediction_proba = model.predict_proba([request.features])[0]

        return PredictionResponse(probability=prediction_proba[1])

    except Exception as e:
        raise HTTPException(status_code=500, detail="Erreur lors de la prédiction.")

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
    return JSONResponse(status_code=422, content={"detail": exc.errors()})
