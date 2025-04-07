import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from io import BytesIO
from scipy.optimize import fsolve
import yfinance as yf
from datetime import datetime, timedelta

# Set page config
st.set_page_config(page_title="Portfolio NSEI Predictor", layout="wide")

# App title
st.title("📈 Portfolio Performance Predictor Based on NSEI Levels")
st.markdown("""
Predict how your portfolio will perform at different NSEI (Nifty 50) index levels.
Enter your stock holdings or upload a file to see predictions.
""")

# Function to calculate beta
def calculate_beta(stock_returns, market_returns):
    covariance = np.cov(stock_returns, market_returns)[0][1]
    market_variance = np.var(market_returns)
    beta = covariance / market_variance
    return beta

# Sidebar for input method selection
with st.sidebar:
    st.header("Input Method")
    input_method = st.radio(
        "Choose how to input your portfolio:",
        ("Manual Entry", "File Upload")
    )
    
    st.header("Settings")
    current_nsei = st.number_input(
        "Current NSEI Level",
        min_value=1000.0,
        max_value=50000.0,
        value=22161.0,
        step=100.0
    )
    
    # Sample data download
    st.markdown("### Need sample data?")
    sample_data = pd.DataFrame({
        'Symbol': ['RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS'],
        'Quantity': [10, 5, 8],
        'Avg. Cost Price': [2500, 3200, 1500],
        'Current Price': [2800, 3500, 1450]
    })
    
    csv = sample_data.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Sample CSV",
        data=csv,
        file_name="sample_portfolio.csv",
        mime="text/csv"
    )

# Initialize portfolio dataframe
if 'portfolio_df' not in st.session_state:
    st.session_state.portfolio_df = pd.DataFrame(columns=['Symbol', 'Quantity', 'Avg. Cost Price', 'Current Price'])

# Manual entry section
if input_method == "Manual Entry":
    st.subheader("Enter Your Stock Holdings")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        symbol = st.text_input("Stock Symbol (e.g., RELIANCE.NS)", key="sym_input")
    with col2:
        quantity = st.number_input("Quantity", min_value=1, step=1, key="qty_input")
    with col3:
        avg_cost = st.number_input("Avg. Cost Price (₹)", min_value=0.01, step=1.0, key="cost_input")
    with col4:
        current_price = st.number_input("Current Price (₹)", min_value=0.01, step=1.0, key="price_input")
    
    if st.button("Add to Portfolio"):
        if symbol:
            new_row = pd.DataFrame({
                'Symbol': [symbol],
                'Quantity': [quantity],
                'Avg. Cost Price': [avg_cost],
                'Current Price': [current_price]
            })
            st.session_state.portfolio_df = pd.concat([st.session_state.portfolio_df, new_row], ignore_index=True)
            st.success(f"Added {symbol} to portfolio")
        else:
            st.error("Please enter a stock symbol")
    
    if not st.session_state.portfolio_df.empty:
        st.subheader("Your Current Portfolio")
        st.dataframe(st.session_state.portfolio_df)
        
        if st.button("Clear Portfolio"):
            st.session_state.portfolio_df = pd.DataFrame(columns=['Symbol', 'Quantity', 'Avg. Cost Price', 'Current Price'])
            st.rerun()
    
    df = st.session_state.portfolio_df.copy()

# File upload section
else:
    st.subheader("Upload Your Portfolio")
    uploaded_file = st.file_uploader(
        "Choose Excel/CSV file with portfolio data",
        type=["csv", "xlsx"],
        help="File should contain columns: Symbol, Quantity, Avg. Cost Price, Current Price"
    )
    
    if uploaded_file is not None:
        try:
            # Read the uploaded file
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            # Check required columns
            required_cols = ['Symbol', 'Quantity', 'Avg. Cost Price', 'Current Price']
            
            if not all(col in df.columns for col in required_cols):
                missing = [col for col in required_cols if col not in df.columns]
                st.error(f"Missing required columns: {', '.join(missing)}")
                st.stop()
            
            # Clean data
            num_cols = ['Quantity', 'Avg. Cost Price', 'Current Price']
            for col in num_cols:
                if col in df.columns:
                    if df[col].dtype == 'object':
                        df[col] = df[col].astype(str).str.replace(',', '').astype(float)
                    df[col] = df[col].astype(float)
        
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
            st.stop()

# Calculate portfolio metrics if we have data
if ('df' in locals() and not df.empty) or ('portfolio_df' in st.session_state and not st.session_state.portfolio_df.empty):
    if 'df' not in locals():
        df = st.session_state.portfolio_df.copy()
    
    # Calculate derived metrics
    df['Invested Value'] = df['Quantity'] * df['Avg. Cost Price']
    df['Market Value'] = df['Quantity'] * df['Current Price']
    df['Unrealized P&L'] = df['Market Value'] - df['Invested Value']
    df['Unrealized P&L %'] = (df['Unrealized P&L'] / df['Invested Value']) * 100
    
    # Calculate portfolio metrics
    portfolio_value = float(df['Market Value'].sum())
    invested_value = float(df['Invested Value'].sum())
    unrealized_pnl = float(df['Unrealized P&L'].sum())
    unrealized_pnl_pct = (unrealized_pnl / invested_value) * 100
    
    # Display portfolio summary
    st.subheader("Portfolio Summary")
    col1, col2, col3 = st.columns(3)
    col1.metric("Current Portfolio Value", f"₹{portfolio_value:,.2f}")
    col2.metric("Invested Value", f"₹{invested_value:,.2f}")
    col3.metric("Unrealized P&L", 
               f"₹{unrealized_pnl:,.2f}", 
               f"{unrealized_pnl_pct:.2f}%",
               delta_color="inverse")
    
    # Fetch historical data for beta calculation
    st.subheader("Portfolio Sensitivity Analysis")
    
    try:
        # Get historical data for stocks and NSEI
        end_date = datetime.today()
        start_date = end_date - timedelta(days=365)  # 1 year data
        
        # Download NSEI data
        nsei_data = yf.download('^NSEI', start=start_date, end=end_date)['Close']
        nsei_returns = nsei_data.pct_change().dropna()
        
        # Download stock data and calculate beta for each stock
        betas = []
        weights = []
        
        for _, row in df.iterrows():
            stock_data = yf.download(row['Symbol'], start=start_date, end=end_date)['Close']
            stock_returns = stock_data.pct_change().dropna()
            
            # Align dates between stock and market returns
            common_dates = nsei_returns.index.intersection(stock_returns.index)
            aligned_market = nsei_returns[common_dates]
            aligned_stock = stock_returns[common_dates]
            
            if len(aligned_stock) > 10:  # Minimum data points required
                beta = calculate_beta(aligned_stock, aligned_market)
                betas.append(beta)
                weights.append(row['Market Value'] / portfolio_value)
        
        # Calculate weighted average portfolio beta
        if betas:
            portfolio_beta = np.average(betas, weights=weights)
            st.write(f"Calculated Portfolio Beta: {portfolio_beta:.2f}")
            
            # Calculate portfolio volatility (standard deviation)
            portfolio_volatility = np.std(betas) * np.sqrt(252)  # Annualized
            st.write(f"Portfolio Volatility (Annualized): {portfolio_volatility:.2%}")
            
            # Calculate correlation with NSEI
            portfolio_returns = []
            for _, row in df.iterrows():
                stock_data = yf.download(row['Symbol'], start=start_date, end=end_date)['Close']
                stock_returns = stock_data.pct_change().dropna()
                common_dates = nsei_returns.index.intersection(stock_returns.index)
                aligned_stock = stock_returns[common_dates]
                if len(portfolio_returns) == 0:
                    portfolio_returns = aligned_stock * (row['Market Value'] / portfolio_value)
                else:
                    portfolio_returns += aligned_stock * (row['Market Value'] / portfolio_value)
            
            correlation = np.corrcoef(portfolio_returns, nsei_returns[common_dates])[0, 1]
            st.write(f"Correlation with NSEI: {correlation:.2f}")
            
        else:
            st.warning("Couldn't calculate beta for all stocks. Using simplified estimation.")
            avg_downside = df[df['Unrealized P&L %'] < 0]['Unrealized P&L %'].mean()
            portfolio_beta = float(abs(avg_downside / 10)) if avg_downside < 0 else 1.0
            st.write(f"Estimated Portfolio Beta: {portfolio_beta:.2f}")
        
    except Exception as e:
        st.warning(f"Couldn't fetch historical data: {str(e)}. Using simplified beta estimation.")
        avg_downside = df[df['Unrealized P&L %'] < 0]['Unrealized P&L %'].mean()
        portfolio_beta = float(abs(avg_downside / 10)) if avg_downside < 0 else 1.0
        st.write(f"Estimated Portfolio Beta: {portfolio_beta:.2f}")
    
    st.caption("""
    - Beta measures your portfolio's sensitivity to NSEI movements. 
    - Volatility shows how much your portfolio fluctuates.
    - Correlation indicates how closely your portfolio follows NSEI.
    """)
    
    # Prediction function
    def predict_portfolio_value(target_nsei):
        target_nsei = float(target_nsei)
        nsei_return_pct = ((target_nsei - current_nsei) / current_nsei) * 100
        portfolio_return_pct = portfolio_beta * nsei_return_pct
        predicted_portfolio_value = portfolio_value * (1 + portfolio_return_pct/100)
        predicted_pnl_pct = ((predicted_portfolio_value - invested_value) / invested_value) * 100
        return float(predicted_portfolio_value), float(predicted_pnl_pct)
    
    # Find breakeven point
    def breakeven_equation(nsei_level):
        port_value, _ = predict_portfolio_value(float(nsei_level[0]))
        return port_value - invested_value
    
    # Numerical solution for breakeven
    breakeven_nsei = float(fsolve(breakeven_equation, current_nsei)[0])
    
    # User input for prediction
    st.subheader("Portfolio Prediction")
    
    col1, col2 = st.columns(2)
    with col1:
        target_nsei = st.number_input(
            "Enter target NSEI level for prediction:",
            min_value=1000.0,
            max_value=50000.0,
            value=float(round(breakeven_nsei)),
            step=100.0
        )
    
    if st.button("Predict Portfolio Value"):
        pred_value, pred_pnl = predict_portfolio_value(target_nsei)
        
        col1, col2 = st.columns(2)
        col1.metric(
            f"Predicted Portfolio Value at NSEI {target_nsei:,.0f}",
            f"₹{pred_value:,.2f}",
            f"{(pred_value - portfolio_value)/portfolio_value*100:.2f}% from current"
        )
        col2.metric(
            "Predicted Unrealized P&L",
            f"₹{pred_value - invested_value:,.2f}",
            f"{pred_pnl:.2f}%",
            delta_color="inverse" if pred_pnl < 0 else "normal"
        )
        
        st.success(f"Your portfolio will break even at NSEI {breakeven_nsei:,.2f} ({(breakeven_nsei - current_nsei)/current_nsei * 100:.2f}% from current)")
    
    # Generate predictions for visualization
    st.subheader("Portfolio Projections")
    nsei_range = np.linspace(
        current_nsei * 0.7, 
        current_nsei * 1.5, 
        50
    ).astype(float)
    predicted_values = [predict_portfolio_value(n)[0] for n in nsei_range]
    predicted_pnls = [predict_portfolio_value(n)[1] for n in nsei_range]
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Portfolio value plot
    ax1.plot(nsei_range, predicted_values, color='blue')
    ax1.axvline(x=current_nsei, color='red', linestyle='--', label='Current NSEI')
    ax1.axhline(y=invested_value, color='green', linestyle='--', label='Invested Value')
    ax1.set_xlabel('NSEI Level')
    ax1.set_ylabel('Portfolio Value (₹)')
    ax1.set_title('Portfolio Value vs NSEI Level')
    ax1.legend()
    ax1.grid(True)
    
    # P&L percentage plot
    ax2.plot(nsei_range, predicted_pnls, color='orange')
    ax2.axvline(x=current_nsei, color='red', linestyle='--', label='Current NSEI')
    ax2.axhline(y=0, color='green', linestyle='--', label='Breakeven')
    ax2.set_xlabel('NSEI Level')
    ax2.set_ylabel('Unrealized P&L (%)')
    ax2.set_title('Portfolio P&L % vs NSEI Level')
    ax2.legend()
    ax2.grid(True)
    
    st.pyplot(fig)
    
    # Show portfolio holdings
    st.subheader("Your Portfolio Holdings")
    st.dataframe(df.style.format({
        'Avg. Cost Price': '{:.2f}',
        'Current Price': '{:.2f}',
        'Invested Value': '₹{:,.2f}',
        'Market Value': '₹{:,.2f}',
        'Unrealized P&L': '₹{:,.2f}',
        'Unrealized P&L %': '{:.2f}%'
    }), use_container_width=True)

else:
    st.info("Please enter your stock holdings or upload a file to get started")
    st.image("https://via.placeholder.com/800x400?text=Enter+your+stock+holdings+or+upload+a+file", use_column_width=True)

# Add some footer information
st.markdown("---")
st.caption("""
Note: This tool provides estimates based on historical beta and correlation. 
Actual market performance may vary due to:
- Individual stock fundamentals
- Market sentiment
- Economic conditions
- Portfolio rebalancing
""")
