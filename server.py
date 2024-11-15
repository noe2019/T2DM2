from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, conint, confloat
import joblib
import numpy as np
import pandas as pd

# Load the model
model = joblib.load("app/best_model.pkl")

# Class names for the prediction
class_names = np.array(["No risk", "Early diabetes risk"])

# Define the FastAPI app
app = FastAPI()

# Define expected input data schema with constrained integers
class PredictionRequest(BaseModel):
    RIAGENDR: conint(ge=1, le=2)  # Example: 1 or 2 for gender
    RIDAGEYR: confloat(ge=21, le=120)  # Example: Age from 0 to 120
    RACE: conint(ge=1, le=4)  # Example: Race category, 1 to 5
    COUPLE: conint(ge=1, le=3)  # Example: 0 or 1 for couple status
    SMOKER: conint(ge=0, le=1)  # Example: 0 or 1 for smoker status
    EDUC: conint(ge=1, le=3)  # Example: Education level, 1 to 5
    COVERED_INSURANCE: conint(ge=0, le=1)  # Example: 0 or 1 for insurance coverage
    FAT: conint(ge=1, le=3)  # Example: FAT score from 0 to 100
    Abdobesity: conint(ge=0, le=1)  # Example: 0 or 1 for abdominal obesity
    TOTAL_ACCULTURATION_SCORE: conint(ge=1, le=3)  # Example: Score from 0 to 100
    POVERTIES: conint(ge=0, le=1)  # Example: Poverty level, 0 to 5
    HTN: conint(ge=0, le=1)  # Example: 0 or 1 for hypertension

@app.get('/')
def read_root():
    return {'message': 'Early diabetes model API'}


@app.post('/predict')
def predict(data: PredictionRequest):
    """
    Predict the class of a given set of features.

    Args:
        data (PredictionRequest): A dictionary containing the features to predict.

    Returns:
        dict: A dictionary containing the predicted class.
    """

    # Convert the input data to a DataFrame
    input_data = pd.DataFrame([data.dict()])

    # Perform prediction
    prediction = model.predict(input_data)

    # Ensure prediction is an integer index
    prediction_index = int(prediction[0])  # Convert first prediction to an integer

    # Use the integer index to get the class name
    class_name = class_names[prediction_index]

    return {'predicted_class': class_name}