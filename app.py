import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from io import BytesIO
import yfinance as yf  # For fetching historical data
from datetime import datetime, timedelta

# Set page config
st.set_page_config(page_title="Enhanced Portfolio Predictor", layout="wide")

# App title
st.title("📊 Enhanced Portfolio Performance Predictor")
st.markdown("""
Predict how your portfolio will perform at different NSEI (Nifty 50) index levels with 
advanced sensitivity analysis and scenario modeling.
""")

# Sidebar for file upload and settings
with st.sidebar:
    st.header("📂 Upload Portfolio Data")
    uploaded_file = st.file_uploader(
        "Choose Excel/CSV file with portfolio data",
        type=["csv", "xlsx"],
        help="File should contain: Symbol, Quantity, Avg Cost, LTP, Invested Value"
    )
    
    st.header("⚙️ Settings")
    current_nsei = st.number_input(
        "Current NSEI Level",
        min_value=1000.0,
        max_value=50000.0,
        value=22161.0,
        step=100.0
    )
    
    prediction_method = st.selectbox(
        "Prediction Method",
        ["Linear Beta Model", "Weighted Sensitivity", "Conservative Estimate"],
        index=0,
        help="Different methods for estimating portfolio sensitivity"
    )
    
    historical_days = st.slider(
        "Historical Window (days) for Beta Calculation",
        30, 365, 90,
        help="Longer periods provide more stable beta estimates"
    )
    
    st.markdown("### Need sample data?")
    sample_data = pd.DataFrame({
        'Symbol': ['RELIANCE.NS', 'HDFCBANK.NS', 'INFY.NS', 'TCS.NS', 'HDFC.NS'],
        'Net Quantity': [10, 15, 20, 5, 8],
        'Avg. Cost Price': [2450.75, 1450.50, 1650.25, 3250.00, 2750.40],
        'LTP': [2600.25, 1500.75, 1700.50, 3400.25, 2800.60],
        'Invested value': [24507.50, 21757.50, 33005.00, 16250.00, 22003.20],
        'Market Value': [26002.50, 22511.25, 34010.00, 17001.25, 22404.80],
        'Unrealized P&L': [1495.00, 753.75, 1005.00, 751.25, 401.60],
        'Unrealized P&L (%)': [6.10, 3.46, 3.04, 4.62, 1.83]
    })
    
    csv = sample_data.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Sample CSV",
        data=csv,
        file_name="sample_portfolio.csv",
        mime="text/csv"
    )

# Main content
if uploaded_file is not None:
    try:
        # Read and validate the uploaded file
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        
        required_cols = ['Symbol', 'Net Quantity', 'Avg. Cost Price', 'LTP', 
                        'Invested value', 'Market Value', 'Unrealized P&L']
        
        if not all(col in df.columns for col in required_cols):
            missing = [col for col in required_cols if col not in df.columns]
            st.error(f"Missing required columns: {', '.join(missing)}")
            st.stop()
        
        # Clean and prepare data
        num_cols = ['Avg. Cost Price', 'LTP', 'Invested value', 'Market Value', 'Unrealized P&L']
        for col in num_cols:
            if df[col].dtype == 'object':
                df[col] = df[col].astype(str).str.replace(',', '').astype(float)
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df.dropna(subset=num_cols, inplace=True)
        
        # Calculate portfolio metrics
        portfolio_value = float(df['Market Value'].sum())
        invested_value = float(df['Invested value'].sum())
        unrealized_pnl = float(df['Unrealized P&L'].sum())
        unrealized_pnl_pct = (unrealized_pnl / invested_value) * 100
        
        # Display portfolio summary
        st.subheader("📊 Portfolio Summary")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Current Value", f"₹{portfolio_value:,.2f}")
        col2.metric("Invested Value", f"₹{invested_value:,.2f}")
        col3.metric("Unrealized P&L", 
                   f"₹{unrealized_pnl:,.2f}", 
                   f"{unrealized_pnl_pct:.2f}%",
                   delta_color="inverse" if unrealized_pnl < 0 else "normal")
        
        # Fetch historical data for beta calculation
        st.subheader("📈 Portfolio Sensitivity Analysis")
        
        with st.spinner("Fetching historical data for accurate beta calculation..."):
            end_date = datetime.today()
            start_date = end_date - timedelta(days=historical_days)
            
            # Get NSEI data
            nsei = yf.download("^NSEI", start=start_date, end=end_date)['Close']
            nsei_returns = nsei.pct_change().dropna()
            
            # Get portfolio constituents data
            portfolio_returns = pd.DataFrame()
            for symbol in df['Symbol']:
                try:
                    stock_data = yf.download(symbol, start=start_date, end=end_date)['Adj Close']
                    stock_returns = stock_data.pct_change().dropna()
                    portfolio_returns[symbol] = stock_returns * (df[df['Symbol'] == symbol]['Market Value'].values[0] / portfolio_value)
                except:
                    st.warning(f"Could not fetch data for {symbol}")
            
            if len(portfolio_returns) == 0:
                st.error("Failed to fetch historical data for any stocks. Using simplified beta calculation.")
                avg_downside = df[df['Unrealized P&L (%)'] < 0]['Unrealized P&L (%)'].mean()
                portfolio_beta = float(abs(avg_downside / 10)) if avg_downside < 0 else 1.0
            else:
                portfolio_returns['Total'] = portfolio_returns.sum(axis=1)
                portfolio_total_returns = portfolio_returns['Total']
                
                # Calculate beta using covariance
                covariance = np.cov(portfolio_total_returns, nsei_returns[-len(portfolio_total_returns):])[0, 1]
                market_variance = np.var(nsei_returns[-len(portfolio_total_returns):])
                portfolio_beta = covariance / market_variance
        
        st.write(f"Calculated Portfolio Beta: {portfolio_beta:.2f}")
        st.caption("""
        Beta measures your portfolio's sensitivity to NSEI movements. 
        A beta of 1 means your portfolio moves with the market. 
        Higher beta means more volatile than market, lower beta means less volatile.
        """)
        
        # Enhanced prediction functions
        def predict_portfolio_value(target_nsei, method=prediction_method):
            target_nsei = float(target_nsei)
            nsei_return_pct = ((target_nsei - current_nsei) / current_nsei) * 100
            
            if method == "Linear Beta Model":
                portfolio_return_pct = portfolio_beta * nsei_return_pct
            elif method == "Weighted Sensitivity":
                weights = df['Market Value'] / portfolio_value
                individual_returns = df['Unrealized P&L (%)'] / ((current_nsei / df['LTP'].mean()) - 1)
                portfolio_return_pct = np.sum(weights * individual_returns) * nsei_return_pct
            else:  # Conservative Estimate
                portfolio_return_pct = np.sqrt(abs(portfolio_beta)) * nsei_return_pct * (1 if portfolio_beta > 0 else -1)
            
            predicted_portfolio_value = portfolio_value * (1 + portfolio_return_pct/100)
            predicted_pnl = predicted_portfolio_value - invested_value
            predicted_pnl_pct = (predicted_pnl / invested_value) * 100
            return predicted_portfolio_value, predicted_pnl, predicted_pnl_pct
        
        def calculate_breakeven():
            if unrealized_pnl >= 0:
                return current_nsei  # Already at or above breakeven
            
            def breakeven_func(n):
                _, pnl, _ = predict_portfolio_value(n[0], prediction_method)
                return pnl
            
            breakeven_nsei = fsolve(breakeven_func, current_nsei)[0]
            return max(breakeven_nsei, 0)  # Ensure non-negative
            
        breakeven_nsei = calculate_breakeven()
        
        # Prediction interface
        st.subheader("🔮 Portfolio Prediction")
        
        col1, col2 = st.columns(2)
        with col1:
            target_nsei = st.number_input(
                "Enter target NSEI level for prediction:",
                min_value=1000.0,
                max_value=50000.0,
                value=float(round(breakeven_nsei)),
                step=100.0
            )
        
        if st.button("Predict Portfolio Performance"):
            pred_value, pred_pnl, pred_pnl_pct = predict_portfolio_value(target_nsei)
            current_pnl_pct = (portfolio_value - invested_value) / invested_value * 100
            
            col1, col2, col3 = st.columns(3)
            col1.metric(
                f"Predicted Value at NSEI {target_nsei:,.0f}",
                f"₹{pred_value:,.2f}",
                f"{(pred_value - portfolio_value)/portfolio_value*100:.2f}% change"
            )
            col2.metric(
                "Predicted P&L (₹)",
                f"₹{pred_pnl:,.2f}",
                f"{pred_pnl_pct:.2f}%",
                delta_color="inverse" if pred_pnl < 0 else "normal"
            )
            col3.metric(
                "Breakeven NSEI Level",
                f"{breakeven_nsei:,.2f}",
                f"{(breakeven_nsei - current_nsei)/current_nsei*100:.2f}% from current",
                delta_color="normal"
            )
            
            # Show prediction details
            with st.expander("Prediction Details"):
                st.write(f"**Method Used:** {prediction_method}")
                st.write(f"**Current NSEI:** {current_nsei:,.2f}")
                st.write(f"**Target NSEI:** {target_nsei:,.2f}")
                st.write(f"**Market Change:** {(target_nsei - current_nsei)/current_nsei*100:.2f}%")
                st.write(f"**Expected Portfolio Change:** {(pred_value - portfolio_value)/portfolio_value*100:.2f}%")
        
        # Generate predictions for visualization
        st.subheader("📉 Portfolio Projections")
        
        nsei_range = np.linspace(
            max(current_nsei * 0.5, 1000),  # Don't go below 1000
            current_nsei * 1.5, 
            50
        )
        
        predicted_values = []
        predicted_pnls = []
        for n in nsei_range:
            val, pnl, _ = predict_portfolio_value(n, prediction_method)
            predicted_values.append(val)
            predicted_pnls.append(pnl)
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Portfolio value plot
        ax1.plot(nsei_range, predicted_values, color='#1f77b4', linewidth=2.5)
        ax1.axvline(x=current_nsei, color='red', linestyle='--', label='Current NSEI')
        ax1.axhline(y=invested_value, color='green', linestyle='--', label='Invested Value')
        ax1.axvline(x=breakeven_nsei, color='purple', linestyle=':', label='Breakeven NSEI')
        ax1.fill_between(nsei_range, invested_value, predicted_values, 
                        where=(np.array(predicted_values) >= invested_value),
                        facecolor='green', alpha=0.1, interpolate=True)
        ax1.fill_between(nsei_range, invested_value, predicted_values, 
                        where=(np.array(predicted_values) < invested_value),
                        facecolor='red', alpha=0.1, interpolate=True)
        ax1.set_xlabel('NSEI Level', fontsize=12)
        ax1.set_ylabel('Portfolio Value (₹)', fontsize=12)
        ax1.set_title('Portfolio Value Projection', fontsize=14)
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # P&L plot
        ax2.plot(nsei_range, predicted_pnls, color='#ff7f0e', linewidth=2.5)
        ax2.axvline(x=current_nsei, color='red', linestyle='--', label='Current NSEI')
        ax2.axhline(y=0, color='black', linestyle='-', label='Breakeven')
        ax2.axvline(x=breakeven_nsei, color='purple', linestyle=':', label='Breakeven NSEI')
        ax2.fill_between(nsei_range, 0, predicted_pnls, 
                        where=(np.array(predicted_pnls) >= 0),
                        facecolor='green', alpha=0.1, interpolate=True)
        ax2.fill_between(nsei_range, 0, predicted_pnls, 
                        where=(np.array(predicted_pnls) < 0),
                        facecolor='red', alpha=0.1, interpolate=True)
        ax2.set_xlabel('NSEI Level', fontsize=12)
        ax2.set_ylabel('Portfolio P&L (₹)', fontsize=12)
        ax2.set_title('Portfolio Profit & Loss Projection', fontsize=14)
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        st.pyplot(fig)
        
        # Portfolio holdings analysis
        st.subheader("🧾 Portfolio Holdings Analysis")
        
        # Calculate individual sensitivities
        df['Weight'] = df['Market Value'] / portfolio_value
        df['Individual Beta'] = df.apply(lambda row: portfolio_beta * row['Weight'], axis=1)  # Simplified
        
        # Format and display
        st.dataframe(
            df.style.format({
                'Avg. Cost Price': '₹{:,.2f}',
                'LTP': '₹{:,.2f}',
                'Invested value': '₹{:,.2f}',
                'Market Value': '₹{:,.2f}',
                'Unrealized P&L': '₹{:,.2f}',
                'Unrealized P&L (%)': '{:.2f}%',
                'Weight': '{:.2%}',
                'Individual Beta': '{:.2f}'
            }).bar(subset=['Unrealized P&L (%)'], align='mid', color=['#d65f5f', '#5fba7d']),
            use_container_width=True
        )
        
        # Export results
        st.subheader("📤 Export Results")
        
        # Create Excel report
        output = BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df.to_excel(writer, sheet_name='Portfolio Holdings', index=False)
            
            # Add summary sheet
            summary_data = {
                'Metric': ['Current Portfolio Value', 'Invested Value', 'Unrealized P&L', 
                          'Portfolio Beta', 'Current NSEI', 'Breakeven NSEI'],
                'Value': [portfolio_value, invested_value, unrealized_pnl,
                         portfolio_beta, current_nsei, breakeven_nsei]
            }
            pd.DataFrame(summary_data).to_excel(writer, sheet_name='Summary', index=False)
            
            # Add projections sheet
            projections = pd.DataFrame({
                'NSEI Level': nsei_range,
                'Portfolio Value': predicted_values,
                'Portfolio P&L': predicted_pnls
            })
            projections.to_excel(writer, sheet_name='Projections', index=False)
        
        st.download_button(
            label="Download Full Report (Excel)",
            data=output.getvalue(),
            file_name="portfolio_analysis_report.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.error("Please check your input file format and try again.")
else:
    st.info("ℹ️ Please upload your portfolio file to get started")
    st.image("https://via.placeholder.com/800x400?text=Upload+your+portfolio+CSV/Excel+file", use_container_width=True)

# Footer
st.markdown("---")
st.caption("""
**Disclaimer:** This tool provides estimates based on historical data and statistical models. 
Actual market performance may vary due to factors including but not limited to:
- Individual stock fundamentals and news
- Market sentiment and macroeconomic conditions
- Portfolio composition changes
- Black swan events

Always consult with a financial advisor before making investment decisions.
""")
