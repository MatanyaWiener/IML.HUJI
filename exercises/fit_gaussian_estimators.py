from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
pio.templates.default = "simple_white"


def test_univariate_gaussian():
    # Question 1 - Draw samples and print fitted model
    mu, sigma, m = 10, 1, 1000
    X = np.random.normal(mu, sigma, size=m)
    estimator = UnivariateGaussian(False)
    estimator.fit(X)
    print("\n----------- Univariate Gaussian -----------\n")
    print("Estimated values:")
    print("expectation:", estimator.mu_, " variance:", estimator.var_)

    # Question 2 - Empirically showing sample mean is consistent
    distance_estimation = []
    for i in range(10, X.size, 10):
        estimator.fit(X[:i])
        distance_estimation.append(abs(estimator.mu_ - mu))

    go.Figure([go.Scatter(x=np.linspace(0, X.size, len(distance_estimation)),
                          y=distance_estimation,
                          mode='markers+lines',
                          name='distance_estimation')],
              layout=go.Layout(
                  title="Distance Between Estimated and True Value of Expectancy",
                  xaxis_title="Number of Samples",
                  yaxis_title="Distance",
                  height=300)).show()


    # Question 3 - Plotting Empirical PDF of fitted model
    pdf_arr = estimator.pdf(X)
    go.Figure([go.Scatter(x=X,
                          y=pdf_arr,
                          mode='markers',
                          name='distance_estimation')],
              layout=go.Layout(
                  title=" 1000 Samples ~N(10,1) and their PDFs",
                  xaxis_title="Sample Value",
                  yaxis_title="PDF of Sample",
                  height=300)).show()


def test_multivariate_gaussian():
    # Question 4 - Draw samples and print fitted model
    print("\n----------- Multivariate Gaussian -----------\n")
    m, d = 1000, 4
    mu = np.array([0, 0, 4, 0])
    sigma = np.array([[1, 0.2, 0, 0.5],
                        [0.2, 2, 0, 0],
                        [0, 0, 1, 0],
                        [0.5, 0, 0, 1]])
    X = np.random.multivariate_normal(mean=mu, cov=sigma, size=m)

    estimator = MultivariateGaussian()
    estimator.fit(X)
    print("Estimated values:")
    print("expectation:", estimator.mu_, "\ncovariance matrix:\n", estimator.cov_)

    # Question 5 - Likelihood evaluation
    f1 = f3 = np.linspace(-10, 10, 200)
    log_likelihood_arr = np.zeros((200, 200))
    # Filling the array with the likelihoods of running values
    for i in range(200):
        for j in range(200):
            mu = np.array([f1[i], 0, f3[j], 0])
            log_likelihood_arr[i][j] = \
                MultivariateGaussian.log_likelihood(mu, sigma, X)

    # Drawing the Heatmap
    fig = go.Figure(go.Heatmap(x=f3, y=f1, z=log_likelihood_arr,
                         colorbar=dict(title="Log-Liklihood View")),
              layout=go.Layout(
                  title=r"Heatmap of log-Likelihood "
                        r"For Models With expectations of [f1, 0, f3, 0]",
                               xaxis=dict(title="$\\text{ Row Sample (f1)}$"),
                               yaxis=dict(title="$\\text{ Column Sample (f3)}$")
                                          ))
    fig.show()

    # Question 6 - Maximum likelihood
    rounding_value = 4
    index = np.argmax(log_likelihood_arr)
    i, j = index // 200, index % 200
    print("\nEstimated Values of mu:")
    print(round(f3[i], rounding_value), ", ", round(f1[j], rounding_value))


if __name__ == '__main__':
    np.random.seed(0)
    test_univariate_gaussian()
    test_multivariate_gaussian()


    print("\n\nLog likelihoods:")
    X = np.array([1, 5, 2, 3, 8, -4, -2, 5, 1, 10, -10, 4, 5, 2, 7, 1, 1, 3, 2, -1, -3, 1, -4, 1, 2, 1,
          -4, -4, 1, 3, 2, 6, -6, 8, 3, -6, 4, 1, -2, 3, 1, 4, 1, 4, -2, 3, -1, 0, 3, 5, 0, -2])
    print(UnivariateGaussian.log_likelihood(1,1,X))
    print(UnivariateGaussian.log_likelihood(10,1,X))
