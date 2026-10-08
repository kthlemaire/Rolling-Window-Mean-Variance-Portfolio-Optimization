# Rolling Window Portfolio Optimization

A Python-based portfolio optimization project that uses historical stock returns and **mean-variance optimization** to construct a portfolio. The optimized portfolio is evaluated against an equal-weight benchmark using realized returns, volatility, utility, and Sharpe ratio.

## Features

- Downloads historical stock data using `yfinance`
- Calculates daily stock returns
- Uses a 60-day rolling window to estimate expected returns and covariance
- Optimizes portfolio weights using mean-variance utility
- Restricts weights to 0–100% with no short selling
- Compares the optimized portfolio with an equal-weight portfolio
- Calculates:
  - Mean return
  - Standard deviation
  - Mean-variance utility
  - Sharpe ratio

## Stocks

The current portfolio consists of:

`MSFT`, `NVDA`, `CRWD`, `GOOGL`, `LITE`

Historical data is downloaded for the period **2020–2025**.

## Requirements

Install the required Python packages:

```bash
pip install yfinance numpy scipy
```

## Usage

Run the script with:

```bash
python portfolio_optimization.py
```

The program will download the required market data, perform rolling-window optimization, and print the performance statistics for both portfolios.

## Methodology

The portfolio maximizes the following mean-variance utility:

$$
U = \mu_p - \frac{\gamma}{2}\sigma_p^2\
$$

where `γ = 2` represents the investor's risk-aversion parameter.

The optimized portfolio is rebalanced at each step using the previous 60 trading days of data.

## Disclaimer

This project is for educational and research purposes only. It is not financial advice, and historical performance does not guarantee future results.
