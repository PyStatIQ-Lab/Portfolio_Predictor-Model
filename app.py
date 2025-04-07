import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from scipy.optimize import fsolve
from io import BytesIO
from stqdm import stqdm
from datetime import datetime, timedelta

# Set page config
st.set_page_config(page_title="Advanced Portfolio Analyzer", layout="wide", initial_sidebar_state="expanded")

# App title and description
st.title("📊 Advanced Portfolio Analyzer with Predictive Modeling")
st.markdown("""
Analyze your portfolio performance with real-time market data, technical indicators, fundamental analysis, 
and predictive modeling based on NSEI movements.
""")

# Sidebar for file upload and settings
with st.sidebar:
    st.header("📂 Portfolio Data")
    uploaded_file = st.file_uploader(
        "Upload your portfolio file (CSV/Excel)",
        type=["csv", "xlsx"],
        help="Required columns: Symbol, Net Quantity, Avg. Cost Price, LTP, Invested value, Market Value"
    )
    
    st.header("⚙️ Settings")
    current_nsei = st.number_input(
        "Current NSEI Level",
        min_value=1000.0,
        max_value=50000.0,
        value=22161.0,
        step=100.0
    )
    
    analysis_period = st.selectbox(
        "Analysis Period",
        ["1mo", "3mo", "6mo", "1y", "2y"],
        index=3
    )
    
    st.header("🔍 Features")
    features = st.multiselect(
        "Select analysis features",
        [
            "Real-time Prices", 
            "Technical Indicators",
            "Fundamental Analysis",
            "Correlation Matrix",
            "Predictive Modeling"
        ],
        default=["Real-time Prices", "Technical Indicators"]
    )
    
    # Sample data download
    with st.expander("Sample Data"):
        sample_data = pd.DataFrame({
            'Symbol': ['RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS'],
            'Net Quantity': [10, 15, 20],
            'Avg. Cost Price': [2450.75, 3250.50, 1450.25],
            'LTP': [2600.25, 3350.75, 1500.50],
            'Invested value': [24507.50, 48757.50, 29005.00],
            'Market Value': [26002.50, 50261.25, 30010.00]
        })
        
        csv = sample_data.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Sample CSV",
            data=csv,
            file_name="sample_portfolio.csv",
            mime="text/csv"
        )

# Cache for API calls
@st.cache_data(ttl=3600, show_spinner=False)
def get_yfinance_data(ticker, period):
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period=period)
        info = stock.info
        return hist, info
    except:
        return None, None

# Technical indicators calculation
def calculate_technical_indicators(hist):
    if hist is None or hist.empty:
        return pd.Series({
            'RSI': np.nan,
            'SMA_50': np.nan,
            'SMA_200': np.nan,
            'MACD': np.nan,
            '52W High': np.nan,
            '52W Low': np.nan
        })
    
    # RSI
    delta = hist['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain/loss
    rsi = 100 - (100/(1+rs))
    
    # Moving Averages
    sma_50 = hist['Close'].rolling(50).mean()
    sma_200 = hist['Close'].rolling(200).mean()
    
    # MACD
    exp12 = hist['Close'].ewm(span=12, adjust=False).mean()
    exp26 = hist['Close'].ewm(span=26, adjust=False).mean()
    macd = exp12 - exp26
    
    # 52 Week High/Low
    high_52w = hist['High'].rolling(252).max()
    low_52w = hist['Low'].rolling(252).min()
    
    return pd.Series({
        'RSI': rsi.iloc[-1],
        'SMA_50': sma_50.iloc[-1],
        'SMA_200': sma_200.iloc[-1],
        'MACD': macd.iloc[-1],
        '52W High': high_52w.iloc[-1],
        '52W Low': low_52w.iloc[-1]
    })

# Fundamental analysis metrics
def get_fundamental_metrics(info):
    if info is None:
        return pd.Series({
            'PE': np.nan,
            'PB': np.nan,
            'ROE': np.nan,
            'Debt/Equity': np.nan,
            'Div Yield': np.nan,
            'Beta': np.nan,
            'Market Cap': np.nan
        })
    
    return pd.Series({
        'PE': info.get('trailingPE'),
        'PB': info.get('priceToBook'),
        'ROE': info.get('returnOnEquity', 0)*100,
        'Debt/Equity': info.get('debtToEquity'),
        'Div Yield': info.get('dividendYield', 0)*100,
        'Beta': info.get('beta'),
        'Market Cap': info.get('marketCap')
    })

# Predictive modeling functions
def calculate_portfolio_beta(df):
    try:
        avg_beta = df['Beta'].mean()
        return avg_beta if not np.isnan(avg_beta) else 1.0
    except:
        return 1.0

def predict_portfolio_value(target_nsei, current_nsei, portfolio_value, portfolio_beta):
    nsei_return_pct = ((target_nsei - current_nsei) / current_nsei) * 100
    portfolio_return_pct = portfolio_beta * nsei_return_pct
    return portfolio_value * (1 + portfolio_return_pct/100)

def train_predictive_model(df, current_nsei):
    try:
        # Prepare data
        X = df[['PE', 'PB', 'Beta', 'RSI']].fillna(0)
        y = df['Unrealized P&L (%)']
        
        # Train models
        lr = LinearRegression()
        lr.fit(X, y)
        
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X, y)
        
        return lr, rf
    except:
        return None, None

# Main app logic
if uploaded_file is not None:
    try:
        # Read and clean data
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        
        required_cols = ['Symbol', 'Net Quantity', 'Avg. Cost Price', 'LTP', 'Invested value', 'Market Value']
        if not all(col in df.columns for col in required_cols):
            missing = [col for col in required_cols if col not in df.columns]
            st.error(f"Missing required columns: {', '.join(missing)}")
            st.stop()
        
        # Convert numeric columns
        num_cols = ['Avg. Cost Price', 'LTP', 'Invested value', 'Market Value']
        for col in num_cols:
            if df[col].dtype == 'object':
                df[col] = df[col].astype(str).str.replace(',', '').astype(float)
        
        # Calculate basic metrics
        df['Unrealized P&L'] = df['Market Value'] - df['Invested value']
        df['Unrealized P&L (%)'] = (df['Unrealized P&L'] / df['Invested value']) * 100
        portfolio_value = df['Market Value'].sum()
        invested_value = df['Invested value'].sum()
        unrealized_pnl = df['Unrealized P&L'].sum()
        unrealized_pnl_pct = (unrealized_pnl / invested_value) * 100
        
        # Display portfolio summary
        st.subheader("📊 Portfolio Summary")
        col1, col2, col3 = st.columns(3)
        col1.metric("Current Value", f"₹{portfolio_value:,.2f}")
        col2.metric("Invested Value", f"₹{invested_value:,.2f}")
        col3.metric("Unrealized P&L", 
                   f"₹{unrealized_pnl:,.2f}", 
                   f"{unrealized_pnl_pct:.2f}%",
                   delta_color="inverse" if unrealized_pnl < 0 else "normal")
        
        # Fetch additional data from Yahoo Finance
        with st.spinner("Fetching market data from Yahoo Finance..."):
            results = []
            for symbol in stqdm(df['Symbol'], desc="Processing stocks"):
                hist, info = get_yfinance_data(symbol, analysis_period)
                tech = calculate_technical_indicators(hist)
                fund = get_fundamental_metrics(info)
                results.append(pd.concat([tech, fund]))
            
            additional_data = pd.concat(results, axis=1).T
            df = pd.concat([df, additional_data], axis=1)
        
        # Feature displays
        if "Real-time Prices" in features:
            st.subheader("💹 Real-time Market Data")
            price_cols = ['Symbol', 'Avg. Cost Price', 'LTP', 
                         'Current Price', '52W High', '52W Low']
            if 'Current Price' not in df.columns:
                df['Current Price'] = df['LTP']  # Fallback to LTP if no real-time data
            
            st.dataframe(
                df[price_cols].style.format({
                    'Avg. Cost Price': '{:,.2f}',
                    'LTP': '{:,.2f}',
                    'Current Price': '{:,.2f}',
                    '52W High': '{:,.2f}',
                    '52W Low': '{:,.2f}'
                }).applymap(
                    lambda x: 'color: green' if isinstance(x, (int, float)) and x > 0 else 'color: red',
                    subset=['Current Price']
                ),
                use_container_width=True
            )
        
        if "Technical Indicators" in features:
            st.subheader("📈 Technical Indicators")
            tech_cols = ['Symbol', 'RSI', 'SMA_50', 'SMA_200', 'MACD']
            
            st.dataframe(
                df[tech_cols].style.format({
                    'RSI': '{:.2f}',
                    'SMA_50': '{:,.2f}',
                    'SMA_200': '{:,.2f}',
                    'MACD': '{:.4f}'
                }).applymap(
                    lambda x: 'background-color: yellow' if isinstance(x, (int, float)) and (x > 70 or x < 30) else '',
                    subset=['RSI']
                ),
                use_container_width=True
            )
            
            # RSI distribution chart
            fig, ax = plt.subplots(figsize=(10, 4))
            sns.histplot(df['RSI'].dropna(), bins=20, kde=True, ax=ax)
            ax.axvline(30, color='red', linestyle='--')
            ax.axvline(70, color='red', linestyle='--')
            ax.set_title('RSI Distribution')
            st.pyplot(fig)
        
        if "Fundamental Analysis" in features:
            st.subheader("🏛️ Fundamental Analysis")
            fund_cols = ['Symbol', 'PE', 'PB', 'ROE', 'Debt/Equity', 'Div Yield', 'Beta', 'Market Cap']
            
            st.dataframe(
                df[fund_cols].style.format({
                    'PE': '{:.2f}',
                    'PB': '{:.2f}',
                    'ROE': '{:.2f}%',
                    'Debt/Equity': '{:.2f}',
                    'Div Yield': '{:.2f}%',
                    'Beta': '{:.2f}',
                    'Market Cap': '₹{:,.2f}'
                }).applymap(
                    lambda x: 'background-color: lightgreen' if isinstance(x, (int, float)) and x > 0 else '',
                    subset=['ROE']
                ),
                use_container_width=True
            )
            
            # Valuation scatter plot
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.scatterplot(data=df, x='PE', y='PB', hue='ROE', size='Market Cap', ax=ax)
            ax.set_title('Valuation Scatter (PE vs PB)')
            st.pyplot(fig)
        
        if "Correlation Matrix" in features:
            st.subheader("🔄 Stock Correlation Matrix")
            
            # Get correlation data
            symbols = df['Symbol'].tolist()
            try:
                prices = yf.download(' '.join(symbols), period=analysis_period)['Adj Close']
                corr = prices.corr()
                
                # Plot correlation heatmap
                fig, ax = plt.subplots(figsize=(12, 10))
                sns.heatmap(corr, annot=True, cmap='coolwarm', center=0, ax=ax)
                ax.set_title(f'Stock Correlation Matrix ({analysis_period})')
                st.pyplot(fig)
            except Exception as e:
                st.warning(f"Could not generate correlation matrix: {str(e)}")
        
        if "Predictive Modeling" in features:
            st.subheader("🔮 Predictive Modeling")
            
            # Calculate portfolio beta
            portfolio_beta = calculate_portfolio_beta(df)
            st.metric("Portfolio Beta", f"{portfolio_beta:.2f}",
                     help="Measures portfolio sensitivity to market movements")
            
            # Simple linear prediction
            st.markdown("#### 📉 Linear Prediction Model")
            target_nsei = st.number_input(
                "Enter target NSEI level for prediction:",
                min_value=1000.0,
                max_value=50000.0,
                value=current_nsei * 1.1,
                step=100.0
            )
            
            if st.button("Predict Portfolio Value"):
                pred_value = predict_portfolio_value(
                    target_nsei, current_nsei, portfolio_value, portfolio_beta
                )
                pred_pnl = ((pred_value - invested_value) / invested_value) * 100
                
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
                
                # Find breakeven point
                def breakeven_equation(nsei_level):
                    port_value = predict_portfolio_value(
                        nsei_level[0], current_nsei, portfolio_value, portfolio_beta
                    )
                    return port_value - invested_value
                
                breakeven_nsei = fsolve(breakeven_equation, current_nsei)[0]
                st.success(f"Breakeven at NSEI: {breakeven_nsei:,.2f} ({(breakeven_nsei - current_nsei)/current_nsei * 100:.2f}% from current)")
            
            # Machine learning prediction
            st.markdown("#### 🤖 Machine Learning Prediction")
            st.write("Using portfolio characteristics to predict returns")
            
            lr_model, rf_model = train_predictive_model(df, current_nsei)
            
            if lr_model and rf_model:
                # Prepare prediction input
                with st.form("ml_prediction"):
                    col1, col2 = st.columns(2)
                    with col1:
                        avg_pe = st.number_input("Average PE Ratio", value=df['PE'].mean())
                        avg_pb = st.number_input("Average PB Ratio", value=df['PB'].mean())
                    with col2:
                        avg_beta = st.number_input("Average Beta", value=df['Beta'].mean())
                        avg_rsi = st.number_input("Average RSI", value=df['RSI'].mean())
                    
                    if st.form_submit_button("Predict with ML"):
                        X_pred = [[avg_pe, avg_pb, avg_beta, avg_rsi]]
                        lr_pred = lr_model.predict(X_pred)[0]
                        rf_pred = rf_model.predict(X_pred)[0]
                        
                        col1, col2 = st.columns(2)
                        col1.metric(
                            "Linear Regression Prediction",
                            f"{lr_pred:.2f}%",
                            help="Predicted portfolio return based on fundamentals"
                        )
                        col2.metric(
                            "Random Forest Prediction",
                            f"{rf_pred:.2f}%",
                            help="Predicted portfolio return using ensemble method"
                        )
            
            # Prediction visualization
            st.markdown("#### 📊 Portfolio Projections")
            nsei_range = np.linspace(
                current_nsei * 0.7, 
                current_nsei * 1.5, 
                20
            )
            pred_values = [predict_portfolio_value(n, current_nsei, portfolio_value, portfolio_beta) for n in nsei_range]
            pred_pnls = [((v - invested_value) / invested_value) * 100 for v in pred_values]
            
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(nsei_range, pred_pnls, label='Projected P&L %')
            ax.axvline(x=current_nsei, color='r', linestyle='--', label='Current NSEI')
            ax.axhline(y=0, color='g', linestyle='--', label='Breakeven')
            ax.set_xlabel('NSEI Level')
            ax.set_ylabel('Portfolio P&L (%)')
            ax.set_title('Portfolio Performance Projection')
            ax.legend()
            ax.grid(True)
            st.pyplot(fig)
        
        # Show full portfolio data
        st.subheader("📋 Full Portfolio Details")
        st.dataframe(
            df.style.format({
                'Avg. Cost Price': '{:,.2f}',
                'LTP': '{:,.2f}',
                'Invested value': '₹{:,.2f}',
                'Market Value': '₹{:,.2f}',
                'Unrealized P&L': '₹{:,.2f}',
                'Unrealized P&L (%)': '{:.2f}%'
            }),
            use_container_width=True
        )
    
    except Exception as e:
        st.error(f"Error processing data: {str(e)}")
else:
    st.info("ℹ️ Please upload your portfolio file to begin analysis")
    st.image("https://via.placeholder.com/800x400?text=Upload+your+portfolio+CSV/Excel+file", use_container_width=True)

# Footer
st.markdown("---")
st.caption("""
Note: This tool provides estimates based on market data and statistical models. 
Actual performance may vary due to market conditions and other factors. 
Data provided by Yahoo Finance. Not investment advice.
""")
