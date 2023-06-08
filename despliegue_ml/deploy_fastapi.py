import pickle

import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()


class InputData(BaseModel):
    age: int
    capital_gain: int
    capital_loss: int
    hours_per_week: int


with open("../models/adult_census_lr.pkl", "rb") as f:
    trained_model = pickle.load(f)


@app.post("/predict")
def predict(input_data: InputData):
    """
    Runs a prediction on the adult census data using a serialized logistic regression model.

    :param input_data: An instance of the InputData class containing values for age, capital-gain,
        capital-loss, and hours-per-week.
    :type input_data: InputData
    :return: A dictionary containing the predicted value.
    :rtype: dict
    """

    # Prepare input data as a pandas DataFrame
    input_df = pd.DataFrame([input_data.dict()])
    input_df.columns = ["age", "capital-gain", "capital-loss", "hours-per-week"]

    # Make predictions
    predictions = trained_model.predict(input_df)

    # Return the predicted value as a dictionary
    return {"prediction": predictions[0]}
