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
Upload your portfolio file and enter target NSEI values to see predictions.
""")

# Sidebar for file upload
with st.sidebar:
    st.header("Upload Portfolio Data")
    uploaded_file = st.file_uploader(
        "Choose Excel/CSV file with portfolio data",
        type=["csv", "xlsx"],
        help="File should contain columns: Symbol, Net Quantity, Avg. Cost Price, LTP, Invested value, Market Value, Unrealized P&L, Unrealized P&L (%)"
    )
    
    st.header("Settings")
    current_nsei = st.number_input(
        "Current NSEI Level",
        min_value=1000.0,
        max_value=50000.0,
        value=22161.0,
        step=100.0
    )
    
    lookback_days = st.slider(
        "Lookback period for beta calculation (days)",
        min_value=30,
        max_value=365,
        value=90,
        help="Longer periods provide more stable beta estimates but may not capture recent changes"
    )
    
    # Sample data download
    st.markdown("### Need sample data?")
    sample_data = pd.DataFrame({
        'Symbol': ['STAR.NS', 'ORCHPHARMA.NS', 'APARINDS.NS'],
        'Net Quantity': [30, 30, 3],
        'Avg. Cost Price': [1397.1, 1680.92, 11145.0],
        'LTP': [575.8, 720.35, 4974.35],
        'Invested value': [41913.0, 50427.60, 33435.0],
        'Market Value': [17274.0, 21610.50, 14923.05],
        'Unrealized P&L': [-24639.0, -28817.10, -18511.95],
        'Unrealized P&L (%)': [-58.79, -57.15, -55.37]
    })
    
    csv = sample_data.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Sample CSV",
        data=csv,
        file_name="sample_portfolio.csv",
        mime="text/csv"
    )

def download_stock_data(tickers, start_date, end_date):
    """Download historical stock data using yfinance"""
    try:
        data = yf.download(tickers, start=start_date, end=end_date)['Close']
        return data
    except Exception as e:
        st.error(f"Error downloading data: {str(e)}")
        return None

def calculate_beta_and_volatility(portfolio_data, nsei_data):
    """Calculate beta and volatility for each stock and portfolio"""
    # Calculate daily returns
    portfolio_returns = portfolio_data.pct_change().dropna()
    nsei_returns = nsei_data.pct_change().dropna()
    
    # Align the dates
    common_dates = portfolio_returns.index.intersection(nsei_returns.index)
    portfolio_returns = portfolio_returns.loc[common_dates]
    nsei_returns = nsei_returns.loc[common_dates]
    
    # Calculate beta for each stock
    betas = {}
    volatilities = {}
    for stock in portfolio_returns.columns:
        X = nsei_returns.values.reshape(-1, 1)
        y = portfolio_returns[stock].values
        
        model = LinearRegression()
        model.fit(X, y)
        betas[stock] = model.coef_[0]
        
        # Calculate annualized volatility
        volatilities[stock] = portfolio_returns[stock].std() * np.sqrt(252)
    
    return betas, volatilities

def calculate_portfolio_beta(betas, weights):
    """Calculate weighted portfolio beta"""
    portfolio_beta = 0
    for stock, beta in betas.items():
        portfolio_beta += beta * weights[stock]
    return portfolio_beta

def calculate_portfolio_volatility(volatilities, weights, betas, nsei_volatility):
    """Calculate portfolio volatility considering market correlation"""
    # This is a simplified approach - for more accuracy you'd need the full covariance matrix
    systematic_risk = sum(betas[stock] * weights[stock] for stock in weights) * nsei_volatility
    idiosyncratic_risk = sum((weights[stock] * volatilities[stock])**2 for stock in weights if stock in volatilities)
    return np.sqrt(systematic_risk**2 + idiosyncratic_risk)

def format_numeric_columns(df):
    """Format numeric columns for display"""
    format_dict = {
        'Avg. Cost Price': '{:,.2f}',
        'LTP': '{:,.2f}',
        'Invested value': '₹{:,.2f}',
        'Market Value': '₹{:,.2f}',
        'Unrealized P&L': '₹{:,.2f}',
        'Unrealized P&L (%)': '{:.2f}%',
        'Weight': '{:.2%}'
    }
    
    formatted_df = df.copy()
    for col, fmt in format_dict.items():
        if col in formatted_df.columns and pd.api.types.is_numeric_dtype(formatted_df[col]):
            formatted_df[col] = formatted_df[col].apply(lambda x: fmt.format(x) if not pd.isna(x) else '')
    
    return formatted_df

# Main content
if uploaded_file is not None:
    try:
        # Read the uploaded file
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        
        # Check required columns
        required_cols = ['Symbol', 'Net Quantity', 'Avg. Cost Price', 'LTP', 
                        'Invested value', 'Market Value', 'Unrealized P&L', 
                        'Unrealized P&L (%)']
        
        if not all(col in df.columns for col in required_cols):
            missing = [col for col in required_cols if col not in df.columns]
            st.error(f"Missing required columns: {', '.join(missing)}")
            st.stop()
        
        # Clean data - ensure numerical columns are float type
        num_cols = ['Avg. Cost Price', 'LTP', 'Invested value', 'Market Value', 'Unrealized P&L']
        for col in num_cols:
            if col in df.columns:
                if df[col].dtype == 'object':
                    df[col] = df[col].astype(str).str.replace(',', '').astype(float)
                df[col] = df[col].astype(float)
        
        # Calculate portfolio metrics
        portfolio_value = float(df['Market Value'].sum())
        invested_value = float(df['Invested value'].sum())
        unrealized_pnl = float(df['Unrealized P&L'].sum())
        unrealized_pnl_pct = (unrealized_pnl / invested_value) * 100
        
        # Display portfolio summary
        col1, col2, col3 = st.columns(3)
        col1.metric("Current Portfolio Value", f"₹{portfolio_value:,.2f}")
        col2.metric("Invested Value", f"₹{invested_value:,.2f}")
        col3.metric("Unrealized P&L", 
                   f"₹{unrealized_pnl:,.2f}", 
                   f"{unrealized_pnl_pct:.2f}%",
                   delta_color="inverse")
        
        # Calculate weights for each stock
        df['Weight'] = df['Market Value'] / portfolio_value
        weights = dict(zip(df['Symbol'], df['Weight']))
        
        # Download historical data
        st.subheader("Portfolio Sensitivity Analysis")
        with st.spinner("Downloading historical data and calculating beta..."):
            end_date = datetime.now()
            start_date = end_date - timedelta(days=lookback_days)
            
            # Get NSEI data
            nsei_data = download_stock_data('^NSEI', start_date, end_date)
            
            # Get portfolio stocks data
            tickers = list(df['Symbol'].unique())
            portfolio_data = download_stock_data(tickers, start_date, end_date)
            
            if portfolio_data is not None and nsei_data is not None:
                # Calculate beta and volatility for each stock
                betas, volatilities = calculate_beta_and_volatility(portfolio_data, nsei_data)
                
                # Calculate NSEI volatility (annualized)
                nsei_volatility = nsei_data.pct_change().std() * np.sqrt(252)
                
                # Calculate portfolio beta
                portfolio_beta = calculate_portfolio_beta(betas, weights)
                
                # Calculate portfolio volatility
                portfolio_volatility = calculate_portfolio_volatility(volatilities, weights, betas, nsei_volatility)
                
                # Display results
                col1, col2 = st.columns(2)
                col1.metric("Portfolio Beta", f"{portfolio_beta:.2f}")
                col1.caption("""
                Beta measures your portfolio's sensitivity to NSEI movements. 
                A beta of 1 means your portfolio moves with the market. 
                Higher beta means more volatile than market.
                """)
                
                col2.metric("Portfolio Volatility (Annualized)", f"{portfolio_volatility*100:.2f}%")
                col2.caption("""
                Volatility measures how much your portfolio's returns fluctuate over time.
                Higher volatility means higher risk.
                """)
                
                # Show beta for individual stocks
                st.write("### Individual Stock Betas")
                beta_df = pd.DataFrame.from_dict(betas, orient='index', columns=['Beta'])
                beta_df['Weight'] = beta_df.index.map(weights)
                st.dataframe(format_numeric_columns(beta_df), use_container_width=True)
                
                # Prediction function with improved model
                def predict_portfolio_value(target_nsei):
                    target_nsei = float(target_nsei)
                    nsei_return_pct = ((target_nsei - current_nsei) / current_nsei) * 100
                    
                    # Use CAPM model: Expected Return = Risk Free Rate + Beta * (Market Return - Risk Free Rate)
                    # For simplicity, we'll assume risk free rate is 0 here
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
                ax1.axvline(x=breakeven_nsei, color='purple', linestyle=':', label='Breakeven NSEI')
                ax1.set_xlabel('NSEI Level')
                ax1.set_ylabel('Portfolio Value (₹)')
                ax1.set_title('Portfolio Value vs NSEI Level')
                ax1.legend()
                ax1.grid(True)
                
                # P&L percentage plot
                ax2.plot(nsei_range, predicted_pnls, color='orange')
                ax2.axvline(x=current_nsei, color='red', linestyle='--', label='Current NSEI')
                ax2.axhline(y=0, color='green', linestyle='--', label='Breakeven')
                ax2.axvline(x=breakeven_nsei, color='purple', linestyle=':', label='Breakeven NSEI')
                ax2.set_xlabel('NSEI Level')
                ax2.set_ylabel('Unrealized P&L (%)')
                ax2.set_title('Portfolio P&L % vs NSEI Level')
                ax2.legend()
                ax2.grid(True)
                
                st.pyplot(fig)
                
                # Show portfolio holdings
                st.subheader("Your Portfolio Holdings")
                st.dataframe(format_numeric_columns(df), use_container_width=True)
            
            else:
                st.error("Failed to download historical data. Please try again later.")
    
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        st.error("Please check your file format and try again.")
else:
    st.info("Please upload your portfolio file to get started")
    st.image("https://via.placeholder.com/800x400?text=Upload+your+portfolio+CSV/Excel+file", use_column_width=True)

# Add some footer information
st.markdown("---")
st.caption("""
Note: This tool provides estimates based on historical data and statistical models. 
Actual market performance may vary due to many factors including:
- Individual stock fundamentals
- Market sentiment
- Economic conditions
- Portfolio rebalancing
- Changes in company-specific risk factors
""")
