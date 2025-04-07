import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from sklearn.linear_model import LinearRegression

# Set page config
st.set_page_config(page_title="Advanced Portfolio Predictor", layout="wide")

# App title
st.title("📊 Advanced Portfolio Predictor with Beta & Volatility Analysis")
st.markdown("""
Predict portfolio performance at different NSEI levels using calculated betas and volatilities.
""")

# Sidebar for inputs
with st.sidebar:
    st.header("Data Inputs")
    uploaded_file = st.file_uploader("Upload Portfolio (CSV/Excel)", type=["csv", "xlsx"])
    
    st.header("Market Parameters")
    current_nsei = st.number_input("Current NSEI Level", value=22161.0, min_value=1000.0, step=100.0)
    risk_free_rate = st.number_input("Risk-free Rate (%)", value=5.0, min_value=0.0, max_value=15.0, step=0.1)/100
    
    st.header("Analysis Period")
    lookback_days = st.selectbox("Lookback Period (Days)", [90, 180, 365], index=2)
    confidence_level = st.slider("Confidence Level (%)", 50, 99, 75)

# Download sample data
with st.sidebar:
    st.markdown("### Sample Data")
    sample_data = pd.DataFrame({
        'Symbol': ['RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS'],
        'Quantity': [10, 15, 20],
        'Avg Cost': [2450, 3350, 1450],
        'LTP': [2650, 3550, 1550]
    })
    csv = sample_data.to_csv(index=False).encode('utf-8')
    st.download_button("Download Sample", data=csv, file_name="sample_portfolio.csv")

# Constants
START_DATE = (datetime.now() - timedelta(days=lookback_days)).strftime('%Y-%m-%d')
END_DATE = datetime.now().strftime('%Y-%m-%d')
BENCHMARK = '^NSEI'

# Calculate daily returns
def get_returns(ticker):
    data = yf.download(ticker, start=START_DATE, end=END_DATE)['Adj Close']
    return data.pct_change().dropna()

# Calculate beta and volatility for a stock
def calculate_stock_metrics(stock_ticker):
    try:
        stock_returns = get_returns(stock_ticker)
        benchmark_returns = get_returns(BENCHMARK)
        
        # Align dates
        common_dates = stock_returns.index.intersection(benchmark_returns.index)
        stock_returns = stock_returns[common_dates]
        benchmark_returns = benchmark_returns[common_dates]
        
        # Calculate beta (regression slope)
        lr = LinearRegression()
        lr.fit(benchmark_returns.values.reshape(-1,1), stock_returns.values)
        beta = lr.coef_[0]
        
        # Calculate volatility (annualized std dev)
        volatility = stock_returns.std() * np.sqrt(252)
        
        return beta, volatility, len(common_dates)
    except:
        return np.nan, np.nan, 0

# Main analysis
if uploaded_file:
    try:
        # Load portfolio
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        
        # Clean data
        df = df.dropna(subset=['Symbol'])
        df['Symbol'] = df['Symbol'].str.upper()
        df['Quantity'] = pd.to_numeric(df['Quantity'], errors='coerce')
        df['Avg Cost'] = pd.to_numeric(df['Avg Cost'], errors='coerce')
        df['LTP'] = pd.to_numeric(df['LTP'], errors='coerce')
        df = df.dropna(subset=['Quantity', 'Avg Cost', 'LTP'])
        
        # Calculate position values
        df['Invested Value'] = df['Quantity'] * df['Avg Cost']
        df['Current Value'] = df['Quantity'] * df['LTP']
        df['P&L'] = df['Current Value'] - df['Invested Value']
        df['P&L %'] = (df['P&L'] / df['Invested Value']) * 100
        
        # Fetch market data and calculate metrics
        st.info("⏳ Downloading market data and calculating betas...")
        
        progress_bar = st.progress(0)
        results = []
        for i, symbol in enumerate(df['Symbol']):
            beta, volatility, data_points = calculate_stock_metrics(symbol)
            results.append({
                'Symbol': symbol,
                'Beta': beta,
                'Volatility': volatility,
                'Data Points': data_points
            })
            progress_bar.progress((i+1)/len(df['Symbol']))
        
        metrics_df = pd.DataFrame(results)
        df = pd.merge(df, metrics_df, on='Symbol')
        
        # Handle missing betas
        missing_beta = df['Beta'].isna()
        if missing_beta.any():
            st.warning(f"Couldn't calculate beta for {missing_beta.sum()} stocks. Using default beta=1.")
            df.loc[missing_beta, 'Beta'] = 1.0
            df.loc[missing_beta, 'Volatility'] = 0.25  # Default 25% volatility
        
        # Portfolio calculations
        total_invested = df['Invested Value'].sum()
        total_current = df['Current Value'].sum()
        total_pnl = total_current - total_invested
        total_pnl_pct = (total_pnl / total_invested) * 100
        
        # Calculate weighted portfolio beta and volatility
        df['Weight'] = df['Current Value'] / total_current
        portfolio_beta = (df['Beta'] * df['Weight']).sum()
        
        # Portfolio volatility considering correlations (simplified)
        portfolio_volatility = np.sqrt((df['Weight']**2 * df['Volatility']**2).sum())
        
        # Display portfolio summary
        st.subheader("Portfolio Summary")
        cols = st.columns(4)
        cols[0].metric("Invested Value", f"₹{total_invested:,.0f}")
        cols[1].metric("Current Value", f"₹{total_current:,.0f}")
        cols[2].metric("P&L", f"₹{total_pnl:,.0f}", f"{total_pnl_pct:.1f}%")
        cols[3].metric("Beta", f"{portfolio_beta:.2f}")
        
        # Show stock-level metrics
        st.subheader("Stock-Level Metrics")
        st.dataframe(df[['Symbol', 'Quantity', 'LTP', 'Current Value', 
                        'Beta', 'Volatility', 'Data Points']].sort_values('Current Value', ascending=False)
                     .style.format({
                         'LTP': '{:.1f}',
                         'Current Value': '₹{:,.0f}',
                         'Beta': '{:.2f}',
                         'Volatility': '{:.1%}',
                     }), height=400)
        
        # Prediction model
        st.subheader("Portfolio Predictions")
        
        def predict_portfolio(target_nsei):
            market_return = (target_nsei - current_nsei) / current_nsei
            expected_return = risk_free_rate + portfolio_beta * (market_return - risk_free_rate)
            predicted_value = total_current * (1 + expected_return)
            
            # Calculate confidence interval
            z_score = {50: 0.67, 75: 1.15, 90: 1.645, 95: 1.96}.get(confidence_level, 1.0)
            std_error = portfolio_volatility * np.sqrt(abs(market_return))
            lower = predicted_value * (1 - z_score * std_error)
            upper = predicted_value * (1 + z_score * std_error)
            
            return predicted_value, lower, upper
        
        # Breakeven calculation
        def breakeven_eq(x):
            pred, _, _ = predict_portfolio(x[0])
            return pred - total_invested
        
        breakeven_nsei = fsolve(breakeven_eq, current_nsei)[0]
        
        # User input for prediction
        target_nsei = st.number_input("Target NSEI Level", value=round(breakeven_nsei), min_value=1000.0)
        
        if st.button("Calculate Prediction"):
            pred, lower, upper = predict_portfolio(target_nsei)
            pnl = pred - total_invested
            pnl_pct = (pnl / total_invested) * 100
            
            st.success(f"Prediction at NSEI {target_nsei:,.0f} ({confidence_level}% Confidence)")
            
            cols = st.columns(3)
            cols[0].metric("Predicted Value", f"₹{pred:,.0f}", 
                          f"{(pred/total_current-1)*100:.1f}% from current")
            cols[1].metric("Expected P&L", f"₹{pnl:,.0f}", f"{pnl_pct:.1f}%")
            cols[2].metric("Prediction Range", f"₹{lower:,.0f} - ₹{upper:,.0f}")
            
            st.info(f"Breakeven at NSEI {breakeven_nsei:,.0f} ({(breakeven_nsei/current_nsei-1)*100:.1f}% change needed)")
        
        # Visualizations
        st.subheader("Projections")
        
        nsei_levels = np.linspace(current_nsei*0.7, current_nsei*1.5, 20)
        predictions = [predict_portfolio(x) for x in nsei_levels]
        pred_values = [x[0] for x in predictions]
        lower_bounds = [x[1] for x in predictions]
        upper_bounds = [x[2] for x in predictions]
        
        fig, ax = plt.subplots(figsize=(10,6))
        ax.plot(nsei_levels, pred_values, label='Predicted Value')
        ax.fill_between(nsei_levels, lower_bounds, upper_bounds, alpha=0.2, label='Confidence Range')
        ax.axvline(current_nsei, color='red', linestyle='--', label='Current NSEI')
        ax.axhline(total_invested, color='green', linestyle='--', label='Invested Value')
        ax.set_xlabel("NSEI Level")
        ax.set_ylabel("Portfolio Value (₹)")
        ax.set_title("Portfolio Value Projection")
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)
        
    except Exception as e:
        st.error(f"Error in analysis: {str(e)}")
else:
    st.info("Please upload a portfolio file to begin analysis")

st.markdown("---")
st.caption("""
**Methodology**: 
- Betas calculated using linear regression of stock returns vs NSEI returns
- Volatility measured as annualized standard deviation of returns
- Predictions use CAPM with confidence intervals based on portfolio volatility
- Data from Yahoo Finance (last {} trading days)
""".format(lookback_days))
