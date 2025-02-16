#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 10:37:21 2025

@author: etienne
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from .model import load_model
import unittest
import requests

# PATH to the model.pkl
model_path = "model/model.pkl"

# Loading the model
model = load_model(model_path)

# Application FastAPI
app = FastAPI()

class Test_api(unittest.TestCase):
    data_less = [-11577, -3287, 0.76, 0.0, 6.0, 0.0, 0.0, -2118.0, -1480.0, 0.0, 0.0, 0.0, 1.0, -109.0]
    data_more = [-11577, -3287, 0.76, 0.0, 6.0, 0.0, 0.0, -2118.0, -1480.0, 0.0, 0.0, 0.0, 1.0, -109.0, 11.43, 3]
    data_str = ["etienne", -3287, 0.76, 0.0, 6.0, 0.0, 0.0, -2118.0, -1480.0, 0.0, 0.0, 0.0, 1.0, -109.0, 11.43]
    
    def test_less_features(self):
        response = requests.post(self.data_less)
        self.assertEqual(response.status_code, 200)
        self.assertIsInstance(response.json(), float)

    def test_more_features(self):
        response = requests.post(self.data_more)
        self.assertEqual(response.status_code, 200)
        self.assertIsInstance(response.json(), float)

    def test_str_feature(self):
        response = requests.post(self.data_str)
        self.assertEqual(response.status_code, 200)
        self.assertIsInstance(response.json(), float)

if __name__ == '__main__':
    unittest.main()

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

# Define a function to return a description of the app
def get_app_description():
	return (
    	"Welcome to the Client loan aprouval API!"
    	"This API allows you to determine if we should allow a credit to a client based on some data."
    	"Use the '/predict/' endpoint with a POST request to make predictions."
    	"Example usage: POST to '/predict/' with JSON data containing"
	)

# Define the root endpoint to return the app description
@app.get("/")
async def root():
	return {"message": get_app_description()}
