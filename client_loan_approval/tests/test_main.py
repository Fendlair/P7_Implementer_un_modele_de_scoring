#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 08:44:37 2025

@author: etienne
"""

import unittest
import requests
import json


# FastAPI API URL
API_URL = "http://127.0.0.1:8000/predict"

class TestPredictionAPI(unittest.TestCase):

    # Test if inputs are numericals values
    def test_input_data_type(self):
        # Input data (Only numericals)
        valid_data = {
            "features": [-11577, -3287, 0.76, 0.0, 6.0, 0.0, 0.0, -2118.0, -1480.0, 0.0, 0.0, 0.0, 1.0, -109.0, 11.43]
        }

        # Input data (Containing a string)
        str_data = {
            "features": ["etienne", -3287, 0.76, 0.0, 6.0, 0.0, 0.0, -2118.0, -1480.0, 0.0, 0.0, 0.0, 1.0, -109.0, 11.43]
        }

        # Test API with numerical data
        response = requests.post(API_URL, json=valid_data)
        self.assertEqual(response.status_code, 200)
        self.assertIsInstance(response.json()["probability"], float)

        # Test data with numerical and str data
        response = requests.post(API_URL, json=str_data)
        self.assertEqual(response.status_code, 422)  # Error code for invalide data
        
    # Testing length of input data
    def test_data_length(self):
        # Data with right length
        correct_length_data = {
            "features": [-11577, -3287, 0.76, 0.0, 6.0, 0.0, 0.0, -2118.0, -1480.0, 0.0, 0.0, 0.0, 1.0, -109.0, 11.43]
        }

        # Data with shorter length
        incorrect_length_data = {
            "features": [-11577, -3287, 0.76, 0.0, 6.0, 0.0, 0.0, -2118.0, -1480.0, 0.0, 0.0, 0.0, 1.0, -109.0]
        }

        # Test if data of the right length
        response = requests.post(API_URL, json=correct_length_data)
        self.assertEqual(response.status_code, 200)

        # Test if data length is not right
        response = requests.post(API_URL, json=incorrect_length_data)
        self.assertEqual(response.status_code, 422)  # Error code for invalide data

    # Test answer of the API
    def test_prediction_response(self):
        # Input data
        valid_data = {
            "features": [-11577, -3287, 0.76, 0.0, 6.0, 0.0, 0.0, -2118.0, -1480.0, 0.0, 0.0, 0.0, 1.0, -109.0, 11.43]
        }

        # Prediction request
        response = requests.post(API_URL, json=valid_data)
        self.assertEqual(response.status_code, 200)
        self.assertIn("probability", response.json())
        self.assertIsInstance(response.json()["probability"], float)

# Run test
if __name__ == "__main__":
    unittest.main()
