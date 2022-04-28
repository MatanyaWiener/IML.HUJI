import IMLearn.learners.regressors.linear_regression
from IMLearn.learners.regressors import PolynomialFitting
from IMLearn.utils import split_train_test

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
pio.templates.default = "simple_white"


def load_data(filename: str) -> pd.DataFrame:
    """
    Load city daily temperature dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (Temp)
    """
    df = pd.read_csv(filename, parse_dates=["Date"])
    df = df[(df.Temp > -70)]
    df["DayOfYear"] = df["Date"].dt.dayofyear
    df = df.drop(['Day', 'Date'], 1)  # Unecesary columns
    return df


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of city temperature dataset
    df = load_data(r"../datasets/City_Temperature.csv")

    # Question 2 - Exploring data for specific country
    df_israel = df[df.Country == "Israel"]

    graph = px.scatter(x=df_israel["DayOfYear"], y=df_israel["Temp"],
                       color=df_israel["Year"].astype(str),
                       title="Temperature in Israel around the year",

                       labels=dict(x="Day of the year", y="Temperature",
                                   color="Year"))
    graph.show()

    df_israel_g = df_israel.groupby("Month")["Temp"].agg(["std"]).reset_index()

    graph = px.bar(x=df_israel_g["Month"], y=df_israel_g["std"],
                   title="STD of temperature in Israel around the year",
                   labels=dict(x="Month", y="STD")).show()



    # Question 3 - Exploring differences between countries
    df_g = df.groupby(["Country", "Month"])["Temp"].agg(
        ["mean", "std"]).reset_index()
    px.line(df_g, x="Month", y='mean', error_y='std',
            color="Country",
            title="Average temperatures and standard deviation along the year in different countries").show()


    # Question 4 - Fitting model for different values of `k`
    X_train, y_train, X_test, y_test = split_train_test(df_israel["DayOfYear"],
                                                        df_israel["Temp"], .75)
    loss = np.zeros(10)
    rang = np.linspace(1, 10, 10).astype(int)
    for i in rang:
        model = PolynomialFitting(i)
        model.fit(X_train.to_numpy(), y_train.to_numpy())
        loss[i - 1] = model.loss(X_test.to_numpy(), y_test.to_numpy()).round(2)
        print(f"MSE for a polinom fit with degree {i}: ", loss[i - 1])

    px.bar(x=rang, y=loss, title="MSE for differnt polinomial degrees",
           labels=dict(x="Degree", y="Loss")).show()


    # Question 5 - Evaluating fitted model on different countries
    MSE_degree = np.argmin(loss) + 1  # it is 5
    model = PolynomialFitting(MSE_degree)
    countries = df["Country"].unique()
    losses = list()

    model.fit(df_israel["DayOfYear"].to_numpy(), df_israel["Temp"].to_numpy())

    for country in countries:
        X_test = df.loc[df["Country"] == country, ["DayOfYear"]]
        y_test = df.loc[df["Country"] == country, ["Temp"]]
        losses.append(model.loss(X_test.to_numpy().flatten(),
                                 y_test.to_numpy().flatten()))

    px.bar(x=countries, y=losses,
           title="Average loss of predicting Temperature in different Countries by stuying Israel",
           labels=dict(x="Country", y="Average Loss")).show()
