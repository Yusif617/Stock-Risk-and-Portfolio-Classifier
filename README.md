# **Stock Risk and Portfolio Classifier**

This repository contains a Jupyter Notebook (`stock_risk_and_portfolio_classifier_v2.ipynb`) that provides a comprehensive analysis of stock risk and portfolio classification. The project aims to calculate various risk metrics for individual stocks, derive a composite risk score, and then classify user portfolios based on the aggregate risk of their holdings using clustering algorithms.

## **Table of Contents**

* [Business Problem](#business-problem)
* [Solution](#solution)
* [Dataset](#dataset)
* [Methodology](#methodology)
* [Prerequisites](#prerequisites)
* [How to Run](#how-to-run)
* [Key Steps in the Code](#key-steps-in-the-code)
* [Results](#results)
* [Visualizations](#visualizations)
* [Future Improvements](#future-improvements)

## **Business Problem**

In financial markets, investors face the challenge of managing risk while optimizing returns. Understanding the risk profile of individual stocks and, more importantly, the overall risk of a diversified portfolio, is crucial for informed investment decisions. Traditional risk metrics often provide a partial view. There is a need for a systematic approach to:

1.  Quantify the risk of individual stocks using multiple indicators.
2.  Aggregate these indicators into a single, comprehensive risk score.
3.  Classify user portfolios based on their aggregated risk profiles, enabling personalized advice or risk management strategies.

This project addresses these challenges by developing a robust framework for stock risk assessment and portfolio classification.

## **Solution**

This project provides a data-driven solution for assessing stock risk and classifying investment portfolios. The Jupyter Notebook demonstrates a process that involves:

1.  **Data Acquisition:** Downloading historical stock prices for a defined set of tickers and market index data using `yfinance`.
2.  **Data Preprocessing:** Handling missing values in stock price data using K-Nearest Neighbors (KNN) imputation.
3.  **Risk Metric Calculation:**
    * Calculating Beta for each stock to measure its volatility relative to the overall market.
    * Calculating annualized mean returns and volatility (standard deviation of returns).
    * Calculating Value at Risk (VaR) at a 95% confidence level to estimate potential losses.
    * **Calculating Sharpe Ratio to measure risk-adjusted return.**
    * **Calculating Conditional Value at Risk (CVaR) at a 95% confidence level to estimate expected shortfall beyond VaR.**
    * **Calculating Maximum Drawdown to measure the largest peak-to-trough decline over a specific period.**
    * Fetching additional financial metrics like P/E ratio, EPS, and Average Volume using `yfinance`.
4.  **Composite Risk Score Derivation:** Normalizing the individual risk metrics and combining them using predefined weights to create a single 'Risk Score' for each stock.
5.  **Portfolio Risk Aggregation:** Loading user portfolio data (shares held per stock), merging it with the calculated stock risk scores, and calculating a 'Weighted Risk Score' for each user's portfolio.
6.  **Portfolio Clustering:** Applying K-Means clustering and DBSCAN to group portfolios into distinct risk profiles based on their features (e.g., Volatility and Risk Score).

By implementing these steps, the project aims to provide a comprehensive tool for risk management and portfolio analysis, aiding investors in making more informed decisions.

## **Dataset**

The project utilizes two main sources of data:

1.  **Historical Stock Data:** Downloaded dynamically using `yfinance` for a list of tickers provided in `tickers.xlsx` and for the market index 'SPY'. This includes daily closing prices.
2.  **User Portfolios:** Loaded from a CSV file named `user_portfolios.csv`. This file is expected to contain a `user_id` column and columns for stock tickers, with values representing the number of shares held.

**Key Dataframes:**

* `stock_prices`: Raw historical stock prices from `yfinance`.
* `stock_close_prices`: Cleaned closing prices for all stocks.
* `market_returns`: Daily percentage changes of the market index (SPY).
* `returns`: Daily percentage changes of individual stock prices.
* `betas`: DataFrame containing calculated Beta values for each stock.
* `stats`: A comprehensive DataFrame compiling 'Mean Return', 'Volatility', 'VaR_95', 'Beta', 'P/E', 'EPS', 'Volume', 'Sharpe', 'CVaR_95', 'Max_Drawdown', and the derived 'Risk Score' for each stock.
* `portfolio_df`: Raw user portfolio data.
* `portfolio_melted`: Transformed user portfolio data with one row per stock holding.
* `portfolio_with_risk`: Merged DataFrame containing user portfolios and stock-specific risk scores.
* `portfolio_risk_scores`: Aggregated weighted risk scores for each user's portfolio.
* `stats_normalized`: Normalized version of the `stats` DataFrame, used for clustering.

## **Methodology**

The analysis follows these main steps:

1.  **Load Libraries:** Import necessary Python libraries (`yfinance`, `pandas`, `numpy`, `scipy.stats.norm`, `matplotlib.pyplot`, `seaborn`, `sklearn.cluster.KMeans`, `sklearn.cluster.DBSCAN`, `sklearn.preprocessing.RobustScaler`, `plotly.express`, `sklearn.impute.KNNImputer`, `google.colab.drive`).
2.  **Mount Google Drive:** If running in Google Colab, mount Google Drive to access `tickers.xlsx` and `user_portfolios.csv`.
3.  **Data Acquisition:**
    * Load stock tickers from `tickers.xlsx`.
    * Download historical 'SPY' (market) data and individual stock data using `yf.download()`.
4.  **Data Cleaning & Imputation:**
    * Extract 'Close' prices from downloaded stock data.
    * Apply `KNNImputer` to handle missing values (e.g., for `WRD` ticker in the provided output) in `stock_close_prices`.
5.  **Calculate Returns:** Compute daily percentage changes for both market and individual stock close prices.
6.  **Calculate Beta:** For each stock, calculate Beta using the covariance between stock returns and market returns, divided by the variance of market returns.
7.  **Calculate Risk/Return Statistics:**
    * Calculate annualized `Mean Return` and `Volatility` (standard deviation) for each stock.
    * Calculate 95% Value at Risk (`VaR_95`) for each stock.
    * **Calculate Sharpe Ratio:** Compute the Sharpe Ratio for each stock, measuring risk-adjusted return (Excess Return / Volatility).
    * **Calculate Conditional Value at Risk (CVaR_95):** Calculate the Conditional Value at Risk at the 95% confidence level, representing the expected loss given that the loss exceeds the VaR.
    * **Calculate Maximum Drawdown:** Determine the maximum drawdown for each stock, which is the largest percentage drop from a peak to a trough in the stock's price over the period.
8.  **Fetch Fundamental Data:** Retrieve 'P/E', 'EPS', and 'Volume' for each stock using `yfinance.Ticker().info`.
9.  **Derive Composite Risk Score:**
    * Normalize all calculated metrics (`Beta`, `Volatility`, `VaR_95`, `P/E`, `EPS`, `Volume`, `Sharpe`, `CVaR_95`, `Max_Drawdown`) using min-max scaling.
    * Define weights for each normalized metric.
    * Calculate a 'Risk Score' for each stock as a weighted sum of its normalized metrics.
10. **Analyze Portfolio Risk:**
    * Load user portfolios from `user_portfolios.csv`.
    * Melt the portfolio DataFrame to a long format, keeping only positive share holdings.
    * Merge portfolio data with individual stock risk scores.
    * Calculate `Total Shares` per user.
    * Compute `Weight` of each stock in a portfolio and `Weighted Risk Score`.
    * Aggregate `Weighted Risk Score` per user to get `Portfolio Risk Scores`.
11. **Cluster Risk Profiles:**
    * Normalize the `stats` DataFrame (excluding 'Mean Return' for clustering purposes here, focusing on risk metrics).
    * Apply `KMeans` clustering to group stocks based on their risk characteristics, assigning a 'Cluster' label.
    * Reorder clusters based on their mean 'Risk Score'.
    * Apply `DBSCAN` clustering as an alternative method for identifying density-based clusters and potential outliers.
12. **Visualization:** Generate interactive scatter plots using `plotly.express` to visualize:
    * 'Volatility' vs. 'Mean Return' for all stocks.
    * 'Volatility' vs. 'Risk Score', colored by K-Means cluster.
    * 'Volatility' vs. 'Risk Score', colored by DBSCAN cluster.

## **Prerequisites**

Make sure you have Python installed. The following Python libraries are required:

```python
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import RobustScaler
import plotly.express as px
from sklearn.impute import KNNImputer
from google.colab import drive # Needed if running in Google Colab
```

You can install these packages using pip:

```bash
pip install yfinance pandas numpy scipy scikit-learn plotly openpyxl
```

Note: `openpyxl` is needed by pandas to read `.xlsx` files.

If you are using Google Colab, `google.colab` is available by default.

## **How to Run**

1.  Download the Jupyter Notebook (`stock_risk_and_portfolio_classifier_v2.ipynb`).
2.  Create two files:
    * `tickers.xlsx`: An Excel file with a single column listing the stock tickers you want to analyze.
    * `user_portfolios.csv`: A CSV file containing user portfolios. It should have a `user_id` column and columns named after the stock tickers, with values indicating the number of shares.
3.  **For Google Colab:**
    * Upload `tickers.xlsx` and `user_portfolios.csv` to your Google Drive in a folder named `stock` (e.g., `/MyDrive/stock/`).
    * Ensure the `drive.mount('/content/drive')` cell is executed to access these files.
4.  **For Local Jupyter Environment:**
    * Place `tickers.xlsx` and `user_portfolios.csv` in the same directory as the notebook, or update the file paths in the data loading cells (`pd.read_excel(...)` and `pd.read_csv(...)`).
5.  Open the notebook in a Jupyter environment (like Jupyter Notebook or JupyterLab).
6.  Run all cells sequentially. The notebook will download data, perform calculations, and generate visualizations.

## **Key Steps in the Code**

This section outlines the main logical steps performed in the Python script without including the code itself.

* Importing necessary libraries for financial data, data manipulation, statistics, machine learning, and visualization.
* Mounting Google Drive (if in Colab) to access input files.
* Loading a list of stock tickers from an Excel file.
* Downloading market index (`SPY`) and individual stock historical closing prices using `yfinance`.
* Imputing any missing stock close price data using KNN Imputer.
* Calculating daily percentage returns for both the market and individual stocks.
* Calculating the Beta coefficient for each stock.
* Creating a `stats` DataFrame to store various metrics: annualized mean return, annualized volatility, and 95% Value at Risk (VaR).
* **Calculating Sharpe Ratio, Conditional Value at Risk (CVaR), and Maximum Drawdown.**
* Fetching additional fundamental data (P/E ratio, EPS, average volume) for each stock.
* Normalizing all risk-related metrics.
* Defining weights for each risk metric.
* Calculating a composite 'Risk Score' for each stock based on the weighted sum of normalized metrics.
* Loading user portfolio data.
* Transforming portfolio data to a melted format for easier processing.
* Merging stock risk scores with portfolio holdings.
* Calculating a 'Weighted Risk Score' for each user's portfolio.
* Normalizing features for clustering.
* Applying K-Means clustering to identify groups of stocks based on risk profiles.
* Reordering K-Means clusters based on their average risk score.
* Applying DBSCAN clustering to identify density-based clusters and potential outliers among stocks.
* Generating interactive scatter plots to visualize:
    * Stock volatility vs. mean return.
    * Stock volatility vs. composite risk score, colored by K-Means cluster.
    * Stock volatility vs. composite risk score, colored by DBSCAN cluster.

## **Results**

The notebook will output various results throughout the execution, including:

1.  Confirmation of Google Drive mounting.
2.  Output from `yf.download` showing data fetching progress.
3.  Preview of `stock_close_prices` DataFrame (raw and after imputation).
4.  Preview of `returns` DataFrame.
5.  A dictionary of calculated Beta values for each stock.
6.  A DataFrame showing Beta values.
7.  A DataFrame displaying `Portfolio Risk Scores` per user, sorted by risk score.
8.  Interactive scatter plots visualizing risk vs. return and composite risk scores with clustering results.
9.  A DataFrame displaying `Sharpe`, `CVaR_95`, and `Max_Drawdown` for each stock, sorted by Sharpe Ratio.

## **Visualizations**

The script generates the following interactive visualizations using `plotly.express`:

* **Interactive Risk vs. Return (Annualized) Scatter Plot:** Displays each stock's annualized volatility on the x-axis and annualized mean return on the y-axis. Stock tickers are shown as text labels, providing a quick overview of the risk-return trade-off for individual assets.
* **Composite Risk Score (K-Means) Scatter Plot:** Plots stock volatility against the calculated 'Risk Score', with points colored according to the K-Means cluster they belong to. This helps in visually identifying different risk profiles among stocks.
* **Composite Risk Score (DBSCAN) Scatter Plot:** Similar to the K-Means plot, but with points colored according to DBSCAN clusters. This can reveal different clustering patterns, especially for identifying outliers (noise points, often colored differently or not clustered).

## **Future Improvements**

* **Portfolio Optimization:** Implement portfolio optimization techniques (e.g., Markowitz Portfolio Theory) to suggest optimal asset allocations based on risk tolerance.
* **Time-Series Analysis:** Implement time-series models (e.g., ARIMA, GARCH) to forecast volatility and returns.
* **Machine Learning Models:** Explore more advanced machine learning models for predicting future risk or return categories.
* **User Interface:** Develop a simple web application or dashboard for users to input their portfolios and visualize risk profiles interactively.
* **Dynamic Data Updates:** Implement a scheduled process to automatically update stock data.
* **Sensitivity Analysis:** Perform sensitivity analysis on the risk score weights to understand their impact on the overall score.
* **Robustness:** Add more robust error handling for API calls (e.g., `yfinance` failures for specific tickers) and data processing.
* **More Clustering Algorithms:** Experiment with hierarchical clustering or Gaussian Mixture Models for portfolio segmentation.
* **Backtesting:** Implement backtesting capabilities to evaluate the performance of portfolio strategies based on the risk classifications.
