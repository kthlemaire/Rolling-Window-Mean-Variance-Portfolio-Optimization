import yfinance as yf
import numpy as np
from scipy.optimize import minimize


###################

# Global Variables

##################

# Stocks
STOCKS = ["MSFT", "NVDA", "CRWD", "GOOGL", "LITE"]
START = "2020-01-01"
END = "2026-01-01"

# Number of daily observations used in each rolling window
N = 60

# Risk-aversion parameter in the objective function
GAMMA = 2

# Assumed risk-free rate
RISK_FREE_ANNUAL = 0.04
RISK_FREE_DAILY = RISK_FREE_ANNUAL / 252


#######################

# Functions

########################


def get_price_data():
    '''
    Downloads the historical daily adjusted closing prices for stocks listed in STOCKS
    Converts the closing prices into daily returns

    Returns: The daily returns for all stocks
    '''
        
    # Getting adj close prices
    prices = yf.download(STOCKS, 
                        start = START,
                        end = END, 
                        auto_adjust= False
                        )["Adj Close"]

    # Converting daily prices to into percentage returns
    returns = prices.pct_change().dropna()

    return(returns)

def get_window_statistics(returns, start_index):
    '''
    Calculates the expected returns a covariance matrix using N observations in the current rolling window. 

    Inputs:
        - returns: dataframe of daily returns
        - start_index: starting index of the rolling window

    Returns:
        - mu: expected daily returns for each stock
        - sigma: covariance matrix of daily returns
    '''

    # Observations in rolling window
    window = returns.iloc[start_index:start_index + N]

    mu = window.mean()
    sigma = window.cov()

    return(mu, sigma)

def objective(w, mu, sigma):
    '''
    Mean-variance objective function

    Inputs:
        - w: portfolio weights
        - mu: expected returns
        - sigma: covariance matrix

    Returns:
        Negative portfolio utility
    '''
    portfolio_return = w @ mu
    portfolio_var = w @ sigma @ w

    utility = portfolio_return - (GAMMA/2) * portfolio_var
    return(-utility)

def optimize(mu, sigma):
    '''
    Finds the portfolio weights that maximize the mean-variance objective function.

    Constraints:
        - Portfolio weights sum to 1
        - Short selling is not allowed
        - Each weight must be between 0 and 1

    Inputs:
        - mu: expected returns
        - sigma: covariance matrix

    Returns:
        - Optimal portfolio weights
    
    '''

    n = len(mu)

    # Starting with equal weights
    initial_weights = np.ones(n) / n

    # Portfolio weights must sum to 1
    constraints = {
        'type': 'eq', 
        'fun': lambda w: np.sum(w) - 1
    }

    # Weights between 0 and 1
    bounds = [(0, 1) for _ in range(n)]

    result = minimize(
        lambda w: objective(w, mu, sigma),
        initial_weights, 
        method = 'SLSQP', 
        bounds = bounds, 
        constraints = constraints
    )

    return result.x

def calculate_statistics(optimized_returns, equal_returns):
    '''
    Calculates mean and standard deviation of the realized portfolio returns

    Inputs:
        - optimized_returns: returns from the optimized portfolio
        - equal_returns: returns from the equal weight portfolio

    Returns:
        - mean and standard deviation for both portfolios
    '''

    # Mean daily returns
    optimized_mean = np.mean(optimized_returns)
    equal_mean = np.mean(equal_returns)

    # Daily std
    optimized_std = np.std(optimized_returns, ddof=1)
    equal_std = np.std(equal_returns, ddof=1)

    return optimized_mean, equal_mean, optimized_std, equal_std

def calculate_sharpe(mean, std):
    '''
    Calculates the daily Sharpe ratio

    Inputs:
        - mean: mean daily portfolio returns
        - std: daily portfolio std

    Return:
        - daily sharpe ratio
    '''

    sharpe = (mean - RISK_FREE_DAILY) / std

    return sharpe


#######################

# Main

#######################

def main():

    # Getting data
    returns = get_price_data()

    # Lists to store portfolio returns
    optimized_returns = []
    equal_returns = []

    # 1/N benchmark portfolio
    equal_weights = np.ones(len(STOCKS)) / len(STOCKS)

    # Rolling window
    for i in range(len(returns) - N):

        mu, sigma = get_window_statistics(returns, i)

        weights = optimize(mu, sigma)

        actual_return = returns.iloc[i + N]

        optimized_return = weights @ actual_return
        equal_return = equal_weights @ actual_return

        optimized_returns.append(optimized_return)
        equal_returns.append(equal_return)

    # Performance statistics
    optimized_mean, equal_mean, optimized_std, equal_std = calculate_statistics(optimized_returns, equal_returns)

    # Sharpe ratios
    optimized_sharpe = calculate_sharpe(optimized_mean, optimized_std)
    equal_sharpe = calculate_sharpe(equal_mean, equal_std)

    # Utility of the portfolios
    optimized_utility = optimized_mean - (GAMMA / 2) * optimized_std**2
    equal_utility = equal_mean - (GAMMA / 2) * equal_std**2


    print("Optimized Portfolio")
    print("Mean Return:", optimized_mean)
    print("Standard Deviation:", optimized_std)
    print("Utility:", optimized_utility)
    print("Sharpe Ratio:", optimized_sharpe)

    print("\nEqual-Weight Portfolio")
    print("Mean Return:", equal_mean)
    print("Standard Deviation:", equal_std)
    print("Utility:", equal_utility)
    print("Sharpe Ratio:", equal_sharpe)


if __name__ == '__main__':
    main()
