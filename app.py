import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from io import BytesIO
from scipy.optimize import fsolve

# Set page config
st.set_page_config(page_title="Portfolio NSEI Predictor", layout="wide")

# App title
st.title("📈 Portfolio Performance Predictor Based on NSEI Levels")
st.markdown("""
Predict how your portfolio will perform at different NSEI (Nifty 50) index levels.
Upload your portfolio file and enter target NSEI values to see predictions.
""")

# Sidebar for file upload and settings
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
    
    risk_free_rate = st.number_input(
        "Risk-free rate (%)", 
        min_value=0.0, 
        max_value=10.0, 
        value=5.0,
        step=0.1
    ) / 100
    
    confidence_level = st.slider(
        "Prediction Confidence Level (%)",
        min_value=50,
        max_value=95,
        value=75,
        step=5
    )
    
    # Sample data download
    st.markdown("### Need sample data?")
    sample_data = pd.DataFrame({
        'Symbol': ['RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'HDFC.NS'],
        'Net Quantity': [10, 15, 20, 25, 30],
        'Avg. Cost Price': [2450.1, 3350.25, 1450.75, 1650.5, 2750.0],
        'LTP': [2650.8, 3550.35, 1550.25, 1750.75, 2850.5],
        'Invested value': [24501.0, 50253.75, 29015.0, 41262.5, 82500.0],
        'Market Value': [26508.0, 53255.25, 31005.0, 43768.75, 85515.0],
        'Unrealized P&L': [2007.0, 3001.5, 1990.0, 2506.25, 3015.0],
        'Unrealized P&L (%)': [8.19, 5.97, 6.86, 6.07, 3.65]
    })
    
    csv = sample_data.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Sample CSV",
        data=csv,
        file_name="sample_portfolio.csv",
        mime="text/csv"
    )

# Default beta values for common stocks
DEFAULT_BETAS = {
    'RELIANCE.NS': 1.03,
    'TCS.NS': 0.79,
    'HDFCBANK.NS': 1.11,
    'INFY.NS': 0.85,
    'HDFC.NS': 1.15,
    'ICICIBANK.NS': 1.25,
    'ITC.NS': 0.65,
    'BHARTIARTL.NS': 0.92,
    'KOTAKBANK.NS': 1.08,
    'LT.NS': 1.12
}

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
        num_cols = ['Net Quantity', 'Avg. Cost Price', 'LTP', 'Invested value', 'Market Value', 'Unrealized P&L']
        for col in num_cols:
            if col in df.columns:
                if df[col].dtype == 'object':
                    df[col] = df[col].astype(str).str.replace(',', '').astype(float)
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Drop rows with missing critical values
        df = df.dropna(subset=['Symbol', 'Market Value', 'Invested value'])
        
        # Calculate portfolio metrics
        portfolio_value = float(df['Market Value'].sum())
        invested_value = float(df['Invested value'].sum())
        unrealized_pnl = float(df['Unrealized P&L'].sum())
        unrealized_pnl_pct = (unrealized_pnl / invested_value) * 100 if invested_value != 0 else 0
        
        # Display portfolio summary
        st.subheader("Portfolio Summary")
        col1, col2, col3 = st.columns(3)
        col1.metric("Current Portfolio Value", f"₹{portfolio_value:,.2f}")
        col2.metric("Invested Value", f"₹{invested_value:,.2f}")
        col3.metric("Unrealized P&L", 
                   f"₹{unrealized_pnl:,.2f}", 
                   f"{unrealized_pnl_pct:.2f}%",
                   delta_color="inverse" if unrealized_pnl < 0 else "normal")
        
        # Calculate portfolio beta (weighted average of individual betas)
        st.subheader("Portfolio Sensitivity Analysis")
        
        df['Weight'] = df['Market Value'] / portfolio_value
        df['Beta'] = df['Symbol'].str.upper().map(DEFAULT_BETAS).fillna(1.0)  # Default to 1 if unknown
        portfolio_beta = (df['Beta'] * df['Weight']).sum()
        
        # Calculate portfolio volatility (simplified)
        portfolio_volatility = portfolio_beta * 0.15  # Assuming market volatility of 15%
        confidence_z = {50: 0.67, 75: 1.15, 90: 1.645, 95: 1.96}.get(confidence_level, 1.0)
        
        st.write(f"Estimated Portfolio Beta: {portfolio_beta:.2f}")
        st.write(f"Estimated Annual Volatility: {portfolio_volatility*100:.1f}%")
        st.caption("""
        - Beta measures your portfolio's sensitivity to NSEI movements
        - Volatility indicates expected fluctuation range
        - Using default betas for unknown stocks (assumed beta=1.0)
        """)
        
        # Prediction function with confidence intervals
        def predict_portfolio_value(target_nsei):
            market_return_pct = ((target_nsei - current_nsei) / current_nsei) * 100
            portfolio_return_pct = risk_free_rate + portfolio_beta * (market_return_pct - risk_free_rate)
            
            # Calculate confidence interval
            std_dev = portfolio_volatility * np.sqrt(abs(market_return_pct)/100)
            lower_bound = portfolio_return_pct - confidence_z * std_dev * 100
            upper_bound = portfolio_return_pct + confidence_z * std_dev * 100
            
            predicted_portfolio_value = portfolio_value * (1 + portfolio_return_pct/100)
            predicted_pnl_pct = ((predicted_portfolio_value - invested_value) / invested_value) * 100
            
            return (
                float(predicted_portfolio_value), 
                float(predicted_pnl_pct),
                float(portfolio_return_pct),
                (float(predicted_portfolio_value * (1 + lower_bound/100)), 
                 float(predicted_portfolio_value * (1 + upper_bound/100)))
            )
        
        # Find breakeven point
        def breakeven_equation(nsei_level):
            port_value, _, _, _ = predict_portfolio_value(float(nsei_level[0]))
            return port_value - invested_value
        
        try:
            breakeven_nsei = float(fsolve(breakeven_equation, current_nsei)[0])
        except:
            # Fallback calculation if fsolve fails
            breakeven_nsei = current_nsei * (invested_value/portfolio_value) ** (1/portfolio_beta)
        
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
            pred_value, pred_pnl, pred_return, (lower_bound, upper_bound) = predict_portfolio_value(target_nsei)
            
            st.success(f"**Prediction at NSEI {target_nsei:,.0f}** ({confidence_level}% confidence)")
            
            col1, col2, col3 = st.columns(3)
            col1.metric(
                "Predicted Value",
                f"₹{pred_value:,.2f}",
                f"{(pred_value - portfolio_value)/portfolio_value*100:.2f}% from current"
            )
            col2.metric(
                "Predicted P&L",
                f"₹{pred_value - invested_value:,.2f}",
                f"{pred_pnl:.2f}%",
                delta_color="inverse" if pred_pnl < 0 else "normal"
            )
            col3.metric(
                "Prediction Range",
                f"₹{lower_bound:,.2f} - ₹{upper_bound:,.2f}",
                f"±{(upper_bound-lower_bound)/pred_value*100:.1f}%"
            )
            
            st.info(f"**Breakeven Analysis**: Your portfolio will break even at NSEI {breakeven_nsei:,.2f} ({(breakeven_nsei - current_nsei)/current_nsei * 100:.2f}% from current)")
        
        # Generate predictions for visualization
        st.subheader("Portfolio Projections")
        nsei_range = np.linspace(
            max(current_nsei * 0.5, 1000),  # Don't go below 1000
            current_nsei * 1.5, 
            30
        ).astype(float)
        
        predictions = [predict_portfolio_value(n) for n in nsei_range]
        predicted_values = [p[0] for p in predictions]
        predicted_pnls = [p[1] for p in predictions]
        lower_bounds = [p[3][0] for p in predictions]
        upper_bounds = [p[3][1] for p in predictions]
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Portfolio value plot
        ax1.plot(nsei_range, predicted_values, color='blue', label='Predicted Value')
        ax1.fill_between(nsei_range, lower_bounds, upper_bounds, color='blue', alpha=0.1, label='Confidence Range')
        ax1.axvline(x=current_nsei, color='red', linestyle='--', label='Current NSEI')
        ax1.axhline(y=invested_value, color='green', linestyle='--', label='Invested Value')
        ax1.set_xlabel('NSEI Level')
        ax1.set_ylabel('Portfolio Value (₹)')
        ax1.set_title('Portfolio Value vs NSEI Level')
        ax1.legend()
        ax1.grid(True)
        
        # P&L percentage plot
        ax2.plot(nsei_range, predicted_pnls, color='orange', label='Predicted P&L%')
        ax2.axvline(x=current_nsei, color='red', linestyle='--', label='Current NSEI')
        ax2.axhline(y=0, color='green', linestyle='--', label='Breakeven')
        ax2.set_xlabel('NSEI Level')
        ax2.set_ylabel('Unrealized P&L (%)')
        ax2.set_title('Portfolio P&L % vs NSEI Level')
        ax2.legend()
        ax2.grid(True)
        
        st.pyplot(fig)
        
        # Show portfolio holdings with beta information
        st.subheader("Your Portfolio Holdings")
        st.dataframe(df[['Symbol', 'Net Quantity', 'Avg. Cost Price', 'LTP', 
                        'Invested value', 'Market Value', 'Unrealized P&L', 
                        'Unrealized P&L (%)', 'Beta']].style.format({
            'Avg. Cost Price': '{:.2f}',
            'LTP': '{:.2f}',
            'Invested value': '₹{:,.2f}',
            'Market Value': '₹{:,.2f}',
            'Unrealized P&L': '₹{:,.2f}',
            'Unrealized P&L (%)': '{:.2f}%',
            'Beta': '{:.2f}'
        }), use_container_width=True)
        
        # Export predictions
        st.subheader("Export Predictions")
        predictions_df = pd.DataFrame({
            'NSEI Level': nsei_range,
            'Predicted Value': predicted_values,
            'Lower Bound': lower_bounds,
            'Upper Bound': upper_bounds,
            'P&L %': predicted_pnls
        })
        
        csv = predictions_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Predictions as CSV",
            data=csv,
            file_name="portfolio_predictions.csv",
            mime="text/csv"
        )
    
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        st.exception(e)
else:
    st.info("Please upload your portfolio file to get started")
    st.image("https://images.unsplash.com/photo-1611974789855-9c2a0a7236a3?ixlib=rb-1.2.1&auto=format&fit=crop&w=800&h=400&q=80", 
             use_column_width=True, 
             caption="Upload your portfolio CSV/Excel file to begin analysis")

# Add some footer information
st.markdown("---")
st.caption("""
**Disclaimer**: This tool provides estimates based on simplified assumptions using the Capital Asset Pricing Model (CAPM). 
Actual market performance may vary due to many factors including:
- Individual stock fundamentals and news
- Market sentiment and macroeconomic conditions
- Portfolio concentration and diversification
- Changes in beta over time

For more accurate predictions, consider using historical beta values from financial data providers.
""")
