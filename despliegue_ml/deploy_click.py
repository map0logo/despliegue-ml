import pickle

import click
import numpy as np
import pandas as pd


np.seterr(all="ignore")


@click.command()
@click.option(
    "--input_data",
    "-i",
    help="Values of age, capital-gain, capital-loss and hours-per-week separated by comma",
)
def predict_adult_census(input_data: str) -> np.ndarray:
    """
    Runs a prediction on the adult census data using a serialized logistic regression model.

    :param input_data: A string containing values for age, capital-gain, capital-loss, and hours-per-week
        separated by commas.
    :type input_data: str
    :return: An array of predicted values.
    :rtype: numpy.ndarray
    """
    # Load the serialized model
    with open("../models/adult_census_lr.pkl", "rb") as f:
        trained_model = pickle.load(f)

    # Prepare input data as a pandas DataFrame
    input_df = pd.DataFrame(
        [input_data.split(",")],
        columns=["age", "capital-gain", "capital-loss", "hours-per-week"],
    )
    input_df = input_df.astype(int)

    # Make predictions
    predictions = trained_model.predict(input_df)
    # predictions = trained_model.predict(
    #     np.fromstring(input_data, dtype=int, sep=",").reshape(1, -1)
    # )
    for prediction in predictions:
        click.echo(prediction)
    # Print the predictions
    return predictions


if __name__ == "__main__":
    predict_adult_census()
