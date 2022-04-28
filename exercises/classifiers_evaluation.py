from IMLearn.learners.classifiers import Perceptron, LDA, GaussianNaiveBayes
from typing import Tuple
from utils import *
import plotly.graph_objects as go
import plotly.io as pio
import plotly.express as px
from plotly.subplots import make_subplots
from math import atan2, pi


def load_dataset(filename: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load dataset for comparing the Gaussian Naive Bayes and LDA classifiers. File is assumed to be an
    ndarray of shape (n_samples, 3) where the first 2 columns represent features and the third column the class

    Parameters
    ----------
    filename: str
        Path to .npy data file

    Returns
    -------
    X: ndarray of shape (n_samples, 2)
        Design matrix to be used

    y: ndarray of shape (n_samples,)
        Class vector specifying for each sample its class

    """
    X = np.load("datasets/" + filename)
    y = X[:, 2]     # Separating
    X = X[:, :2]
    return X, y

def run_perceptron():
    """
    Fit and plot fit progression of the Perceptron algorithm over both the linearly separable and inseparable datasets

    Create a line plot that shows the perceptron algorithm's training loss values (y-axis)
    as a function of the training iterations (x-axis).
    """
    for n, f in [("Linearly Separable", "linearly_separable.npy"),
                 ("Linearly Inseparable", "linearly_inseparable.npy")]:
        # Load dataset
        X, y = load_dataset(f)

        # Fit Perceptron and record loss in each fit iteration
        losses = []

        def my_callback(fit: Perceptron, x_: np.ndarray, y_: int):
            # Loss tracking callback
            losses.append(fit.loss(X, y))

        pers = Perceptron(callback=my_callback)
        pers.fit(X, y)
        # Plot figure
        px.line(x=np.arange(0,len(losses)), y=losses,
                title=n,
                labels=dict(x="Iterations", y="Loss")).show()



def get_ellipse(mu: np.ndarray, cov: np.ndarray):
    """
    Draw an ellipse centered at given location and according to specified covariance matrix

    Parameters
    ----------
    mu : ndarray of shape (2,)
        Center of ellipse

    cov: ndarray of shape (2,2)
        Covariance of Gaussian

    Returns
    -------
        scatter: A plotly trace object of the ellipse
    """
    l1, l2 = tuple(np.linalg.eigvalsh(cov)[::-1])
    theta = atan2(l1 - cov[0, 0], cov[0, 1]) if cov[0, 1] != 0 else (np.pi / 2 if cov[0, 0] < cov[1, 1] else 0)
    t = np.linspace(0, 2 * pi, 100)
    xs = (l1 * np.cos(theta) * np.cos(t)) - (l2 * np.sin(theta) * np.sin(t))
    ys = (l1 * np.sin(theta) * np.cos(t)) + (l2 * np.cos(theta) * np.sin(t))

    return go.Scatter(x=mu[0] + xs, y=mu[1] + ys, mode="lines", marker_color="black")


def compare_gaussian_classifiers():
    """
    Fit both Gaussian Naive Bayes and LDA classifiers on both gaussians1 and gaussians2 datasets
    """
    for f in ["gaussian1.npy", "gaussian2.npy"]:
        # Load dataset
        X, y = load_dataset(f)

        # Fit models and predict over training set
        lda = LDA()
        lda.fit(X, y)
        gnb = GaussianNaiveBayes()
        gnb.fit(X, y)

        # Plot a figure with two suplots, showing the Gaussian Naive Bayes predictions on the left and LDA predictions
        # on the right. Plot title should specify dataset used and subplot titles should specify algorithm and accuracy
        # Create subplots
        from IMLearn.metrics import accuracy

        y_lda = lda.predict(X)
        y_gnb = gnb.predict(X)

        graph = make_subplots(rows=1, cols=2,
                            subplot_titles=(
                                f"Data from: {f}, Classifier: Naive Gaussian Bayes, Accuracy: {accuracy(y, y_gnb):.4f}",
                                f"Data from: {f}, Classifier: LDA, Accuracy: {accuracy(y, y_lda):.4f}"))

        # Add traces for data-points setting symbols and colors
        graph.add_trace(row=1, col=1,
                      trace=go.Scatter(x=X.T[0], y=X.T[1], mode='markers',
                                       marker=go.scatter.Marker(color=y_gnb,
                                                                symbol=y)))
        graph.add_trace(row=1, col=2,
                      trace=go.Scatter(x=X.T[0], y=X.T[1], mode='markers',
                                       marker=go.scatter.Marker(color=y_lda,
                                                                symbol=y)))

        # Add `X` dots specifying fitted Gaussians' means
        # On the GNB:
        graph.add_trace(row=1, col=1,
                      trace=go.Scatter(x=gnb.mu_[:, 0], y=gnb.mu_[:, 1],
                                       mode='markers',
                                       marker={'symbol': 'x', 'color': 'black',
                                               'size': 10}))
        # On the LDA:
        graph.add_trace(row=1, col=2,
                      trace=go.Scatter(x=lda.mu_[:, 0], y=lda.mu_[:, 1],
                                       mode='markers',
                                       marker={'symbol': 'x', 'color': 'black',
                                               'size': 10}))

        # Add ellipses depicting the covariances of the fitted Gaussians
        for i in range(len(lda.classes_)):
            d = X.shape[1]
            cov = np.zeros(shape=(d, d))
            np.fill_diagonal(cov, gnb.vars_[i])
            graph.add_trace(row=1, col=1, trace=get_ellipse(gnb.mu_[i], cov))
            graph.add_trace(row=1, col=2,
                          trace=get_ellipse(lda.mu_[i], lda.cov_))

        graph.update_layout(showlegend=False)
        graph.show()


if __name__ == '__main__':
    np.random.seed(0)
    run_perceptron()
    compare_gaussian_classifiers()


