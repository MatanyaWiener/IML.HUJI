from IMLearn.utils import split_train_test
from IMLearn.learners.regressors import LinearRegression
import plotly.io as pio

from typing import NoReturn
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
pio.templates.default = "simple_white"


def load_data(filename: str):
    """
    Load house prices dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (prices) - either as a single
    DataFrame or a Tuple[DataFrame, Series]
    """
    
    df = pd.read_csv(filename)
    
    # Ceaning the data:
    df = df[(df.price > 0) & (df.bathrooms > 0) & (df.bedrooms > 0) & (df.sqft_lot15 > 0)]  # removing weird values
    df = pd.concat([df, pd.get_dummies(df.zipcode, drop_first = True)], axis = 1)
    
    df = df.drop(['date', 'id', 'zipcode'], 1) # Unecesary columns
    df = df.dropna().reset_index()
    df = df.drop(['index'], 1)
    
    y = df.price
    X = df.drop(["price"], 1)
    return X, y

def feature_evaluation(X: pd.DataFrame, y: pd.Series, output_path: str = ".") -> NoReturn:
    """
    Create scatter plot between each feature and the response.
        - Plot title specifies feature name
        - Plot title specifies Pearson Correlation between feature and response
        - Plot saved under given folder with file name including feature name
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector to evaluate against

    output_path: str (default ".")
        Path to folder in which plots are saved
    """
    plot_feature_corrolation("sqft_living", X, y, output_path)
    plot_feature_corrolation("bedrooms", X, y, output_path)
    
    
def plot_feature_corrolation(feature: str, X: pd.DataFrame, y: pd.Series, output_path: str = ".") -> NoReturn:
    x = X[feature]
    cov_matrix = np.cov(x, y)
    cov = cov_matrix[0][1]
    sig_x = np.sqrt(cov_matrix[0][0])
    sig_y = np.sqrt(cov_matrix[1][1])
    corr = cov / (sig_x * sig_y)
    
    graph = px.scatter(x=X[feature], y=y, 
               title = f"{feature} corrolation to price is: {corr.round(4)}", 
               labels = dict(x = feature, y = "price"))
    pio.write_image(graph, f"{output_path}/{feature}.png", format = "png")

if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of housing prices dataset
    X, y = load_data(r"..\datasets\house_prices.csv")

    # Question 2 - Feature evaluation with respect to response
    feature_evaluation(X,y)

    # Question 3 - Split samples into training- and testing sets.
    X_train, y_train, X_test, y_test = split_train_test(X, y, .75)

    # Question 4 - Fit model over increasing percentages of the overall training data
    # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
    #   1) Sample p% of the overall training data
    #   2) Fit linear model (including intercept) over sampled set
    #   3) Test fitted model over test set
    #   4) Store average and variance of loss over test set
    # Then plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)

    model = LinearRegression()

    loss_means = np.zeros(91)
    loss_std = np.zeros(91)

    for i in range(10, 101):
        p = 0.01 * i
        loss = np.zeros(10)
        for j in range(10):
            sample_X = X_train.sample(frac=p)
            sample_y = y_train[sample_X.index]
            model._fit(sample_X.to_numpy(), sample_y.to_numpy())
            loss[j] = model._loss(X_test.to_numpy(), y_test.to_numpy())

        loss_means[i-10] = np.mean(loss)
        loss_std[i-10] = np.std(loss)

    graph = go.Figure(
        go.Scatter(x=list(range(10, 101)), y=loss_means, name="means"))
    graph.add_traces([go.Scatter(x=list(range(10, 101)),
                                 y=loss_means + 2 * loss_std, fill="tonexty",
                                 mode="lines", line=dict(color="lightblue"),
                                 name="top of CI"),
                      go.Scatter(x=list(range(10, 101)),
                                 y=loss_means - 2 * loss_std, fill="tonexty",
                                 mode="lines", line=dict(color="lightblue"),
                                 name="bottom of CI")])
    graph.update_layout(
        title="Loss as Result of the Sample Size",
        xaxis_title="Size of the Sample",
        yaxis_title="Loss",
        legend_title=None)
    graph.show()
            
            
            