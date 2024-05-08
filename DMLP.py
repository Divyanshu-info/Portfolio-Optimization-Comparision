from cvxpy import Maximize
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from scipy.optimize import minimize

def build_dmlp_model(input_shape):
    """
    Builds the DMLP model architecture.
    """
    model = Sequential()
    model.add(Dense(15, activation='relu'))
    for _ in range(5):
        model.add(Dense(15, activation='relu'))
    model.add(Dense(1))
    return model

def train_dmlp(model, X_train, y_train, epochs, batch_size):
    """
    Trains the DMLP model on the training data.
    """
    model.compile(optimizer=Adam(learning_rate=0.01), loss='mean_absolute_error')
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)
    return model

def portfolio_return(weights, returns):
    """
    Computes the portfolio's return for a given set of weights and asset returns.
    """
    return np.sum(weights * returns)

def portfolio_volatility(weights, covariances):
    """
    Computes the portfolio's volatility for a given set of weights and asset covariances.
    """
    return np.sqrt(np.dot(weights.T, np.dot(covariances, weights)))

def negative_sharpe_ratio(weights, returns, covariances, risk_free_rate):
    """
    Computes the negative of the Sharpe ratio for a given set of weights, asset returns, asset covariances, and risk-free rate.
    """
    portfolio_return_rate = portfolio_return(weights, returns)
    portfolio_volatility_rate = portfolio_volatility(weights, covariances)
    sharpe_ratio = (portfolio_return_rate - risk_free_rate) / portfolio_volatility_rate
    return -sharpe_ratio

def optimize_portfolio(predicted_returns, covariances, risk_free_rate):
    """
    Optimizes the portfolio weights to maximize the Sharpe ratio.
    """
    num_assets = predicted_returns.shape[0]
    initial_weights = np.ones(num_assets) / num_assets
    bounds = tuple((0.0, 1.0) for asset in range(num_assets))
    constraint = {'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1.0}

    result = minimize(negative_sharpe_ratio, initial_weights, args=(predicted_returns, covariances, risk_free_rate), method='SLSQP', bounds=bounds, constraints=constraint)

    return result.x

# Example usage
if __name__ == "__main__":
    # Load stock prices and compute returns
    stock_prices = pd.read_csv("stock_prices.csv", index_col="Date", parse_dates=True)
    returns = stock_prices.pct_change().dropna()

    # Preprocess data for DMLP
    scaler = MinMaxScaler()
    X = scaler.fit_transform(stock_prices.values)
    y = returns.values

    # Split data into train and test sets
    split_ratio = 0.8
    split_index = int(len(X) * split_ratio)
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]

    # Build and train DMLP model
    input_shape = (X_train.shape[1], )
    dmlp_model = build_dmlp_model(input_shape)
    trained_model = train_dmlp(dmlp_model, X_train, y_train, epochs=100, batch_size=100)

    # Predict stock returns using the trained DMLP model
    predicted_returns = trained_model.predict(X_test)

    # Compute covariances from the test set
    covariances = np.cov(predicted_returns.T)

    # Optimize the portfolio weights
    risk_free_rate = 0.02  # Assuming a 2% risk-free rate
    optimal_weights = optimize_portfolio(predicted_returns.max(axis=0), covariances, risk_free_rate)

    # Print the optimal weights
    print("Optimal Portfolio Weights:")
    for stock, weight in zip(stock_prices.columns, optimal_weights):
        print(f"{stock}: {weight:.4f}")