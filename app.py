import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from io import BytesIO

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
    
    # Sample data download
    st.markdown("### Need sample data?")
    sample_data = pd.DataFrame({
        'Symbol': ['STAR.NS', 'ORCHPHARMA.NS', 'APARINDS.NS'],
        'Net Quantity': [30, 30, 3],
        'Avg. Cost Price': [1397.1, 1680.92, 11145],
        'LTP': [575.8, 720.35, 4974.35],
        'Invested value': [41913, 50427.60, 33435],
        'Market Value': [17274, 21610.50, 14923.05],
        'Unrealized P&L': [-24639, -28817.10, -18511.95],
        'Unrealized P&L (%)': [-58.79, -57.15, -55.37]
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
        
        # Clean data
        df = df.dropna(subset=required_cols)
        for col in ['Invested value', 'Market Value', 'Unrealized P&L']:
            if df[col].dtype == 'object':
                df[col] = df[col].str.replace(',', '').astype(float)
        
        # Calculate portfolio metrics
        portfolio_value = df['Market Value'].sum()
        invested_value = df['Invested value'].sum()
        unrealized_pnl = df['Unrealized P&L'].sum()
        unrealized_pnl_pct = (unrealized_pnl / invested_value) * 100
        
        # Display portfolio summary
        col1, col2, col3 = st.columns(3)
        col1.metric("Current Portfolio Value", f"₹{portfolio_value:,.2f}")
        col2.metric("Invested Value", f"₹{invested_value:,.2f}")
        col3.metric("Unrealized P&L", 
                   f"₹{unrealized_pnl:,.2f}", 
                   f"{unrealized_pnl_pct:.2f}%",
                   delta_color="inverse")
        
        # Calculate portfolio beta (simplified approach)
        st.subheader("Portfolio Sensitivity Analysis")
        
        # For beta calculation, we'll assume a relationship based on unrealized P&L%
        # In a real app, you would use historical data for accurate beta
        avg_downside = df[df['Unrealized P&L (%)'] < 0]['Unrealized P&L (%)'].mean()
        portfolio_beta = abs(avg_downside / 10)  # Assuming market dropped 10% to reach current P&L
        
        st.write(f"Estimated Portfolio Beta: {portfolio_beta:.2f}")
        st.caption("""
        Beta measures your portfolio's sensitivity to NSEI movements. 
        A beta of 1 means your portfolio moves with the market. 
        Higher beta means more volatile than market.
        """)
        
        # Prediction function
        def predict_portfolio_value(target_nsei):
            nsei_return_pct = ((target_nsei - current_nsei) / current_nsei) * 100
            portfolio_return_pct = portfolio_beta * nsei_return_pct
            predicted_portfolio_value = portfolio_value * (1 + portfolio_return_pct/100)
            predicted_pnl_pct = ((predicted_portfolio_value - invested_value) / invested_value) * 100
            return predicted_portfolio_value, predicted_pnl_pct
        
        # Find breakeven point
        def breakeven_equation(nsei_level):
            port_value, _ = predict_portfolio_value(nsei_level)
            return port_value - invested_value
        
        # Numerical solution for breakeven
        from scipy.optimize import fsolve
        breakeven_nsei = fsolve(
            lambda x: breakeven_equation(x[0]), 
            current_nsei
        )[0]
        
        # User input for prediction
        st.subheader("Portfolio Prediction")
        
        col1, col2 = st.columns(2)
        with col1:
            target_nsei = st.number_input(
                "Enter target NSEI level for prediction:",
                min_value=1000.0,
                max_value=50000.0,
                value=round(breakeven_nsei),
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
        )
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
            'LTP': '{:.2f}',
            'Invested value': '₹{:,.2f}',
            'Market Value': '₹{:,.2f}',
            'Unrealized P&L': '₹{:,.2f}',
            'Unrealized P&L (%)': '{:.2f}%'
        }), use_container_width=True)
    
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
else:
    st.info("Please upload your portfolio file to get started")
    st.image("https://via.placeholder.com/800x400?text=Upload+your+portfolio+CSV/Excel+file", use_column_width=True)

# Add some footer information
st.markdown("---")
st.caption("""
Note: This tool provides estimates based on simplified assumptions. 
Actual market performance may vary due to many factors including:
- Individual stock fundamentals
- Market sentiment
- Economic conditions
- Portfolio rebalancing
""")
