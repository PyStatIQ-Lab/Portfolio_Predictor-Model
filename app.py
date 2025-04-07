import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from datetime import datetime, timedelta

# Set page config
st.set_page_config(
    page_title="Portfolio NSEI Predictor", 
    layout="wide",
    page_icon="📈"
)

# App title
st.title("📈 Portfolio Performance Predictor Based on NSEI Levels")
st.markdown("""
Predict how your portfolio will perform at different NSEI (Nifty 50) index levels.
Upload your portfolio file and enter target NSEI values to see predictions.
""")

# ----------------------------------------
# SIDEBAR CONTROLS
# ----------------------------------------
with st.sidebar:
    st.header("📤 Upload Portfolio Data")
    uploaded_file = st.file_uploader(
        "Choose Excel/CSV file with portfolio data",
        type=["csv", "xlsx"],
        help="File should contain: Symbol, Quantity, Avg. Cost, LTP, Invested Value, Market Value"
    )
    
    st.header("⚙️ Settings")
    current_nsei = st.number_input(
        "Current NSEI Level",
        min_value=1000.0,
        max_value=50000.0,
        value=22161.0,
        step=100.0
    )
    
    lookback_period = st.selectbox(
        "Historical Data Period",
        options=["3 months", "6 months", "1 year", "2 years"],
        index=2
    )
    
    # Sample data
    st.markdown("### 🧪 Need sample data?")
    sample_data = pd.DataFrame({
        'Symbol': ['RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS'],
        'Net Quantity': [10, 15, 5],
        'Avg. Cost Price': [2500, 3200, 1500],
        'LTP': [2800, 3400, 1600],
        'Invested value': [25000, 48000, 7500],
        'Market Value': [28000, 51000, 8000],
    })
    
    csv = sample_data.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Sample CSV",
        data=csv,
        file_name="sample_portfolio.csv",
        mime="text/csv"
    )

# ----------------------------------------
# DATA PROCESSING FUNCTIONS
# ----------------------------------------
def clean_symbol(symbol):
    """Ensure symbol has .NS suffix for NSE stocks"""
    symbol = str(symbol).upper().strip()
    if not symbol.endswith('.NS'):
        return f"{symbol.split('.')[0]}.NS"
    return symbol

def get_historical_data(symbols, period):
    """Download historical stock data from Yahoo Finance"""
    period_map = {
        "3 months": "3mo",
        "6 months": "6mo",
        "1 year": "1y",
        "2 years": "2y"
    }
    
    try:
        cleaned_symbols = [clean_symbol(sym) for sym in symbols]
        
        data = yf.download(
            cleaned_symbols,
            period=period_map[period],
            interval="1d",
            group_by='ticker',
            progress=False,
            threads=True
        )
        
        # Process downloaded data
        result = {}
        found_symbols = []
        
        if len(cleaned_symbols) == 1:
            if not data.empty:
                result[cleaned_symbols[0]] = data
                found_symbols.append(cleaned_symbols[0])
        else:
            available_symbols = list(set(col[0] for col in data.columns.levels[0])) if isinstance(data.columns, pd.MultiIndex) else []
            
            for sym in cleaned_symbols:
                if sym in available_symbols:
                    result[sym] = data[sym]
                    found_symbols.append(sym)
        
        # Identify missing symbols
        missing = set(cleaned_symbols) - set(found_symbols)
        if missing:
            st.warning(f"⚠️ Could not find data for: {', '.join(missing)}")
        
        return result if result else None
        
    except Exception as e:
        st.error(f"❌ Error downloading data: {str(e)}")
        return None

def calculate_stock_metrics(stock_data, index_data):
    """Calculate beta and volatility for a stock"""
    try:
        # Get closing prices (handles both 'Close' and 'Adj Close')
        stock_close = stock_data['Adj Close'] if 'Adj Close' in stock_data.columns else stock_data['Close']
        index_close = index_data['Adj Close'] if 'Adj Close' in index_data.columns else index_data['Close']
        
        merged = pd.merge(
            stock_close.rename('Stock'),
            index_close.rename('Index'),
            left_index=True,
            right_index=True,
            how='inner'
        ).dropna()
        
        # Calculate daily returns
        returns = merged.pct_change().dropna()
        
        # Calculate beta and volatility
        covariance = np.cov(returns['Stock'], returns['Index'])[0, 1]
        variance = np.var(returns['Index'])
        beta = covariance / variance
        volatility = np.std(returns['Stock']) * np.sqrt(252)  # Annualized
        
        return beta, volatility, returns
    
    except Exception as e:
        st.warning(f"⚠️ Error calculating metrics: {str(e)}")
        return 1.0, 0.3, None  # Default values

# Sector beta mapping for fallback
SECTOR_BETAS = {
    'IT': 0.9,          # TCS, INFY
    'BANK': 1.1,        # HDFCBANK, ICICIBANK
    'AUTO': 1.2,        # MARUTI, TATAMOTORS
    'PHARMA': 0.8,      # SUNPHARMA, DRREDDY
    'FMCG': 0.7,        # HUL, ITC
    'METAL': 1.5,       # TATASTEEL, JSWSTEEL
    'OIL': 1.3,         # RELIANCE, ONGC
    'TELECOM': 0.9,     # BHARTIARTL
    'DEFAULT': 1.0      # Fallback
}

def get_sector_beta(symbol):
    """Get sector-specific beta when data is unavailable"""
    symbol = str(symbol).upper()
    
    if any(x in symbol for x in ['BANK', 'FIN', 'HDFC', 'ICICI', 'AXIS']):
        return SECTOR_BETAS['BANK']
    elif any(x in symbol for x in ['TECH', 'INFO', 'TCS', 'WIPRO', 'INFY']):
        return SECTOR_BETAS['IT']
    elif any(x in symbol for x in ['PHARMA', 'LIFE', 'DRREDDY', 'SUN']):
        return SECTOR_BETAS['PHARMA']
    elif any(x in symbol for x in ['OIL', 'GAS', 'RELIANCE', 'ONGC']):
        return SECTOR_BETAS['OIL']
    elif any(x in symbol for x in ['STEEL', 'METAL', 'TATASTEEL']):
        return SECTOR_BETAS['METAL']
    elif any(x in symbol for x in ['MOTORS', 'AUTO', 'MARUTI', 'HERO']):
        return SECTOR_BETAS['AUTO']
    elif any(x in symbol for x in ['HUL', 'ITC', 'NESTLE', 'BRITANNIA']):
        return SECTOR_BETAS['FMCG']
    
    return SECTOR_BETAS['DEFAULT']

# ----------------------------------------
# MAIN APP LOGIC
# ----------------------------------------
if uploaded_file is not None:
    try:
        # Read uploaded file
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        
        # Validate required columns
        required_cols = ['Symbol', 'Net Quantity', 'Avg. Cost Price', 'LTP', 'Invested value', 'Market Value']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            st.error(f"❌ Missing required columns: {', '.join(missing_cols)}")
            st.stop()
        
        # Clean numerical data
        num_cols = ['Avg. Cost Price', 'LTP', 'Invested value', 'Market Value']
        for col in num_cols:
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', ''), errors='coerce')
        
        # Calculate portfolio metrics
        portfolio_value = df['Market Value'].sum()
        invested_value = df['Invested value'].sum()
        unrealized_pnl = portfolio_value - invested_value
        unrealized_pnl_pct = (unrealized_pnl / invested_value) * 100
        
        # Display portfolio summary
        st.subheader("📊 Portfolio Summary")
        col1, col2, col3 = st.columns(3)
        col1.metric("Current Value", f"₹{portfolio_value:,.2f}")
        col2.metric("Invested Value", f"₹{invested_value:,.2f}")
        col3.metric("Unrealized P&L", 
                   f"₹{unrealized_pnl:,.2f}", 
                   f"{unrealized_pnl_pct:.2f}%",
                   delta_color="inverse")
        
        # Calculate portfolio beta and volatility
        st.subheader("📈 Risk Analysis")
        with st.spinner("Downloading market data and calculating risk metrics..."):
            symbols = df['Symbol'].unique().tolist()
            
            # Download stock and index data
            stock_data = get_historical_data(symbols, lookback_period)
            nsei_data = get_historical_data(['^NSEI'], lookback_period)
            nsei_data = nsei_data['^NSEI'] if nsei_data else None
            
            # Calculate metrics for each stock
            beta_values = []
            volatility_values = []
            
            for symbol in symbols:
                if stock_data and symbol in stock_data and nsei_data is not None:
                    beta, volatility, _ = calculate_stock_metrics(stock_data[symbol], nsei_data)
                    beta_values.append(beta)
                    volatility_values.append(volatility)
                else:
                    # Use sector-based beta as fallback
                    beta = get_sector_beta(symbol)
                    beta_values.append(beta)
                    volatility_values.append(0.3)  # Default volatility
                    st.warning(f"Using sector beta ({beta:.2f}) for {symbol}")
            
            # Calculate portfolio-level metrics
            df['Weight'] = df['Market Value'] / portfolio_value
            df['Beta'] = beta_values
            df['Volatility'] = volatility_values
            portfolio_beta = (df['Weight'] * df['Beta']).sum()
            portfolio_volatility = (df['Weight'] * df['Volatility']).sum()
        
        # Display risk metrics
        col1, col2 = st.columns(2)
        col1.metric("Portfolio Beta", f"{portfolio_beta:.2f}")
        col2.metric("Annual Volatility", f"{portfolio_volatility:.2%}")
        
        st.caption("""
        - **Beta**: Measures sensitivity to NSEI movements (1 = market average)
        - **Volatility**: Annualized standard deviation of returns
        """)
        
        # Show stock-level metrics
        st.subheader("📋 Stock-Level Metrics")
        st.dataframe(df[['Symbol', 'Market Value', 'Weight', 'Beta', 'Volatility']].style.format({
            'Market Value': '₹{:,.2f}',
            'Weight': '{:.2%}',
            'Beta': '{:.2f}',
            'Volatility': '{:.2%}'
        }), height=400)
        
        # ----------------------------------------
        # PREDICTION ENGINE
        # ----------------------------------------
        st.subheader("🔮 Portfolio Predictions")
        
        def predict_portfolio_value(target_nsei):
            """Predict portfolio value at target NSEI level"""
            nsei_return_pct = ((target_nsei - current_nsei) / current_nsei) * 100
            portfolio_return_pct = portfolio_beta * nsei_return_pct
            predicted_value = portfolio_value * (1 + portfolio_return_pct/100)
            predicted_pnl_pct = ((predicted_value - invested_value) / invested_value) * 100
            return predicted_value, predicted_pnl_pct
        
        # Find breakeven NSEI level
        def breakeven_equation(nsei_level):
            port_value, _ = predict_portfolio_value(nsei_level[0])
            return port_value - invested_value
        
        breakeven_nsei = fsolve(breakeven_equation, current_nsei)[0]
        
        # Prediction interface
        col1, col2 = st.columns(2)
        with col1:
            target_nsei = st.number_input(
                "Enter target NSEI level:",
                min_value=1000.0,
                max_value=50000.0,
                value=float(round(breakeven_nsei)),
                step=100.0
            )
        
        if st.button("🚀 Predict Portfolio Value"):
            pred_value, pred_pnl = predict_portfolio_value(target_nsei)
            
            col1, col2 = st.columns(2)
            col1.metric(
                f"Predicted Value at NSEI {target_nsei:,.0f}",
                f"₹{pred_value:,.2f}",
                f"{(pred_value - portfolio_value)/portfolio_value*100:.2f}% from current"
            )
            col2.metric(
                "Predicted P&L",
                f"₹{pred_value - invested_value:,.2f}",
                f"{pred_pnl:.2f}%",
                delta_color="inverse" if pred_pnl < 0 else "normal"
            )
            
            st.success(f"✨ Break-even at NSEI {breakeven_nsei:,.2f} ({(breakeven_nsei - current_nsei)/current_nsei * 100:.2f}% change needed)")
        
        # Generate projection charts
        st.subheader("📊 Portfolio Projections")
        nsei_range = np.linspace(current_nsei * 0.7, current_nsei * 1.5, 50)
        pred_values = [predict_portfolio_value(n)[0] for n in nsei_range]
        pred_pnls = [predict_portfolio_value(n)[1] for n in nsei_range]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Value projection
        ax1.plot(nsei_range, pred_values, color='blue')
        ax1.axvline(current_nsei, color='red', linestyle='--', label='Current')
        ax1.axhline(invested_value, color='green', linestyle='--', label='Invested')
        ax1.set_xlabel('NSEI Level')
        ax1.set_ylabel('Portfolio Value (₹)')
        ax1.set_title('Value Projection')
        ax1.legend()
        ax1.grid(True)
        
        # P&L projection
        ax2.plot(nsei_range, pred_pnls, color='orange')
        ax2.axvline(current_nsei, color='red', linestyle='--', label='Current')
        ax2.axhline(0, color='green', linestyle='--', label='Break-even')
        ax2.set_xlabel('NSEI Level')
        ax2.set_ylabel('P&L (%)')
        ax2.set_title('P&L Projection')
        ax2.legend()
        ax2.grid(True)
        
        st.pyplot(fig)
        
        # Show full portfolio
        st.subheader("🧾 Portfolio Holdings")
        st.dataframe(df.style.format({
            'Avg. Cost Price': '₹{:.2f}',
            'LTP': '₹{:.2f}',
            'Invested value': '₹{:,.2f}',
            'Market Value': '₹{:,.2f}',
            'Weight': '{:.2%}',
            'Beta': '{:.2f}',
            'Volatility': '{:.2%}'
        }), height=600)
    
    except Exception as e:
        st.error(f"❌ Error processing file: {str(e)}")
        st.stop()
else:
    st.info("ℹ️ Please upload your portfolio file to get started")
    st.image("https://via.placeholder.com/800x400?text=Upload+your+portfolio+CSV/Excel+file", use_column_width=True)

# Footer
st.markdown("---")
st.caption("""
**Note**: Predictions are based on historical beta calculations. Actual performance may vary due to:
- Changes in stock fundamentals
- Market sentiment shifts
- Macroeconomic conditions
- Corporate actions
""")
