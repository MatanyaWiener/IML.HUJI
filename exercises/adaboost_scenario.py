import numpy as np
from typing import Tuple
from IMLearn.metalearners.adaboost import AdaBoost
from IMLearn.learners.classifiers import DecisionStump
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from IMLearn.metrics import accuracy


def generate_data(n: int, noise_ratio: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a dataset in R^2 of specified size

    Parameters
    ----------
    n: int
        Number of samples to generate

    noise_ratio: float
        Ratio of labels to invert

    Returns
    -------
    X: np.ndarray of shape (n_samples,2)
        Design matrix of samples

    y: np.ndarray of shape (n_samples,)
        Labels of samples
    """
    '''
    generate samples X with shape: (num_samples, 2) and labels y with shape (num_samples).
    num_samples: the number of samples to generate
    noise_ratio: invert the label for this ratio of the samples
    '''
    X, y = np.random.rand(n, 2) * 2 - 1, np.ones(n)
    y[np.sum(X ** 2, axis=1) < 0.5 ** 2] = -1
    y[np.random.choice(n, int(noise_ratio * n))] *= -1
    return X, y


def decision_surface_partial(T, predict_partial, xrange, yrange, density=120, colorscale=custom, showscale=True):
    """
    function that plots a decision surface - added the partial option
    """
    xrange, yrange = np.linspace(*xrange, density), np.linspace(*yrange, density)
    xx, yy = np.meshgrid(xrange, yrange)
    pred = predict_partial(np.c_[xx.ravel(), yy.ravel()], T)

    return go.Contour(x=xrange, y=yrange, z=pred.reshape(xx.shape),
                      colorscale=colorscale,
                      reversescale=False, opacity=.7, connectgaps=True,
                      hoverinfo="skip",
                      showlegend=False, showscale=showscale)


def fit_and_evaluate_adaboost(noise, n_learners=250, train_size=5000, test_size=500):
    (train_X, train_y), (test_X, test_y) = generate_data(train_size, noise), generate_data(test_size, noise)
    learner = AdaBoost(DecisionStump, n_learners)
    learner._fit(train_X, train_y)

    # Question 1: Train- and test errors of AdaBoost in noiseless case

    test_losses = [learner.partial_loss(test_X, test_y, T) for T in range(1, n_learners)]
    train_losses = [learner.partial_loss(train_X, train_y, T) for T in range(1, n_learners)]

    # plot errors
    fig = go.Figure()
    fig.add_traces([go.Scatter(x=np.arange(1, n_learners), y=train_losses,
                               name='train error', opacity=.75),
                    go.Scatter(x=np.arange(1, n_learners), y=test_losses,
                               name='test error', opacity=.75)])
    fig.update_layout(
        title=dict(text=f'Loss as function of number of Decision Stumps.  Noise: {noise}'))
    fig.layout.xaxis.title = 'Number of decision stumps'
    fig.layout.yaxis.title = 'Loss'
    fig.show()


    # Question 2: Plotting decision surfaces
    T = [5, 50, 100, 250]
    lims = np.array([np.r_[train_X, test_X].min(axis=0), np.r_[train_X, test_X].max(axis=0)]).T + np.array([-.1, .1])


    fig = make_subplots(rows=2, cols=2,
                        subplot_titles=[f"{i} Decision Stumps" for i in T],
                        horizontal_spacing=0.01, vertical_spacing=.03)
    for i, T in enumerate(T):
        fig.add_traces([decision_surface_partial(T, learner.partial_predict, lims[0],
                                          lims[1], showscale=False),
                        go.Scatter(x=test_X[:, 0], y=test_X[:, 1],
                                   mode="markers", showlegend=False,
                                   marker=dict(color=test_y.astype(int),
                                               symbol=test_y.astype(int) + 2,
                                               colorscale=[custom[0],
                                                           custom[-1]],
                                               line=dict(color="black",
                                                         width=1)))],
                       rows=1 + (i // 2), cols=1 + (i % 2))

    fig.update_layout(
        title=rf"$\textbf{{Decision Boundaries By Different Numbers of Learners. Noise: {noise}}}$",
        margin=dict(t=100)) \
        .update_xaxes(visible=False).update_yaxes(visible=False).show()

    # Question 3: Decision surface of best performing ensemble
    best_n_learners = np.argmin(test_losses) + 1
    y_pred = learner.partial_predict(test_X, best_n_learners)
    acc = accuracy(y_pred, test_y)

    # plot the results
    fig = go.Figure()
    fig.add_trace(
        decision_surface_partial(best_n_learners, learner.partial_predict, lims[0], lims[1],
                          showscale=False))
    fig.add_trace(go.Scatter(x=test_X[:, 0], y=test_X[:, 1], mode="markers",
                             showlegend=False,
                             marker=dict(color=test_y.astype(int),
                                         symbol=test_y.astype(int) + 2,
                                         colorscale=[custom[0], custom[-1]],
                                         line=dict(color="black", width=1))))

    fig.update_layout(title=dict(
        text=f'Best Classifier. Number of Classifiers: {best_n_learners}. accuracy:{acc}, Noise: {noise}'))
    fig.show()


    # Question 4: Decision surface with weighted samples
    D = learner.D_
    D = (D / np.max(D)) * 15
    fig = go.Figure()
    fig.add_trace(
        decision_surface_partial(learner.iterations_, learner.partial_predict,
                          lims[0], lims[1], showscale=False))
    fig.add_trace(go.Scatter(x=train_X[:, 0], y=train_X[:, 1], mode="markers",
                             showlegend=False,
                             marker=dict(color=train_y.astype(int),
                                         symbol=train_y.astype(int) + 2,
                                         colorscale=[custom[0], custom[-1]],
                                         line=dict(color="black", width=1),
                                         size=D)))

    fig.update_layout(
        title=dict(text=f'Samples Proportionally to Their Wight.  Noise: {noise}'))
    fig.show()


if __name__ == '__main__':
    np.random.seed(0)
    fit_and_evaluate_adaboost(noise=0)
    fit_and_evaluate_adaboost(noise=0.4)