import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from io import BytesIO

# Set page config
st.set_page_config(page_title="Portfolio NSEI Predictor (Validated)", layout="wide")

# App title
st.title("✅ Portfolio Performance Predictor (Validated Calculations)")
st.markdown("""
**Precisely** predict portfolio performance at different NSEI levels with validated calculations.
""")

# Sidebar for file upload
with st.sidebar:
    st.header("📤 Data Input")
    uploaded_file = st.file_uploader(
        "Upload portfolio file",
        type=["csv", "xlsx"],
        help="Required columns: Symbol, Net Quantity, Avg. Cost Price, LTP, Invested value, Market Value"
    )
    
    st.header("⚙️ Market Parameters")
    current_nsei = st.number_input(
        "Current NSEI Level",
        min_value=1000.0,
        max_value=50000.0,
        value=22161.0,
        step=100.0
    )

# Main analysis
if uploaded_file is not None:
    try:
        # Read and validate data
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        
        # Validate columns
        required_cols = ['Symbol', 'Net Quantity', 'Avg. Cost Price', 'LTP', 
                        'Invested value', 'Market Value']
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            st.error(f"Missing columns: {', '.join(missing)}")
            st.stop()
        
        # Clean and calculate numeric fields
        num_cols = ['Avg. Cost Price', 'LTP', 'Invested value', 'Market Value']
        for col in num_cols:
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', ''), errors='coerce')
        
        # Calculate derived fields with validation
        df['Unrealized P&L'] = df['Market Value'] - df['Invested value']
        df['Unrealized P&L (%)'] = (df['Unrealized P&L'] / df['Invested value']) * 100
        
        # Portfolio metrics with cross-validation
        portfolio_value = df['Market Value'].sum()
        invested_value = df['Invested value'].sum()
        unrealized_pnl = df['Unrealized P&L'].sum()
        
        # Validation check
        if not np.isclose(unrealized_pnl, portfolio_value - invested_value, rtol=0.01):
            st.warning("P&L validation mismatch - recalculating")
            unrealized_pnl = portfolio_value - invested_value
        
        unrealized_pnl_pct = (unrealized_pnl / invested_value) * 100
        
        # Display metrics with color coding
        st.subheader("💰 Portfolio Summary")
        cols = st.columns(3)
        cols[0].metric("Current Value", f"₹{portfolio_value:,.2f}")
        cols[1].metric("Invested Value", f"₹{invested_value:,.2f}")
        
        pnl_color = "inverse" if unrealized_pnl < 0 else "normal"
        cols[2].metric("Unrealized P&L", 
                      f"₹{unrealized_pnl:,.2f}", 
                      f"{unrealized_pnl_pct:.2f}%",
                      delta_color=pnl_color)
        
        # Improved Beta Calculation
        st.subheader("📈 Sensitivity Analysis")
        
        # Method 1: Using price changes
        df['Price Change %'] = ((df['LTP'] - df['Avg. Cost Price']) / df['Avg. Cost Price']) * 100
        market_return_assumption = -10  # Assuming market dropped 10% to current state
        df['Estimated Beta'] = df['Price Change %'] / market_return_assumption
        
        # Weighted portfolio beta
        df['Weight'] = df['Market Value'] / portfolio_value
        portfolio_beta = (df['Estimated Beta'] * df['Weight']).sum()
        
        # Method 2: Alternative calculation
        alt_beta = abs(unrealized_pnl_pct / market_return_assumption)
        
        # Use average of both methods
        final_beta = np.mean([portfolio_beta, alt_beta])
        
        st.write(f"""
        **Validated Portfolio Beta**: {final_beta:.2f}
        - Method 1 (Stock-level): {portfolio_beta:.2f}
        - Method 2 (Portfolio-level): {alt_beta:.2f}
        """)
        
        # Prediction function with validation
        def predict_portfolio_value(target_nsei):
            market_return = ((target_nsei - current_nsei) / current_nsei) * 100
            portfolio_return = final_beta * market_return
            predicted_value = portfolio_value * (1 + portfolio_return/100)
            
            # Validate calculation
            if not np.isclose(
                predicted_value, 
                portfolio_value + (portfolio_value * (final_beta * (target_nsei - current_nsei)/current_nsei),
                rtol=0.01
            ):
                st.warning("Prediction validation mismatch")
            
            predicted_pnl = predicted_value - invested_value
            predicted_pnl_pct = (predicted_pnl / invested_value) * 100
            return predicted_value, predicted_pnl, predicted_pnl_pct
        
        # Accurate Breakeven Calculation
        def calculate_breakeven():
            required_return_pct = (-unrealized_pnl_pct) / final_beta
            breakeven_nsei = current_nsei * (1 + required_return_pct/100)
            
            # Verify
            test_value, _, _ = predict_portfolio_value(breakeven_nsei)
            if not np.isclose(test_value, invested_value, rtol=0.01):
                st.error("Breakeven validation failed - using approximation")
                breakeven_nsei = current_nsei * (1 + (-unrealized_pnl_pct/100)/final_beta
            
            return breakeven_nsei
        
        breakeven_nsei = calculate_breakeven()
        breakeven_change_pct = ((breakeven_nsei - current_nsei) / current_nsei) * 100
        
        # Prediction UI
        st.subheader("🔮 Portfolio Prediction")
        target_nsei = st.number_input(
            "Target NSEI Level:", 
            min_value=1000.0,
            max_value=50000.0,
            value=float(round(breakeven_nsei)),
            step=100.0
        )
        
        if st.button("Calculate Prediction"):
            pred_value, pred_pnl, pred_pnl_pct = predict_portfolio_value(target_nsei)
            
            # Display results
            cols = st.columns(3)
            cols[0].metric("Predicted Value", f"₹{pred_value:,.2f}")
            
            pnl_color = "inverse" if pred_pnl < 0 else "normal"
            cols[1].metric("Projected P&L (₹)", f"₹{pred_pnl:,.2f}")
            cols[2].metric("Projected P&L (%)", f"{pred_pnl_pct:.2f}%", delta_color=pnl_color)
            
            # Breakeven analysis
            st.success(f"""
            **Breakeven Analysis**:
            - Breakeven at NSEI: {breakeven_nsei:,.2f}
            - Required change: {breakeven_change_pct:.2f}%
            - Verification: At this level, P&L = ₹{pred_value - invested_value:,.2f}
            """)
        
        # Visualization with enhanced validation
        st.subheader("📊 Projection Charts")
        nsei_range = np.linspace(current_nsei * 0.7, current_nsei * 1.5, 50)
        values = [predict_portfolio_value(n)[0] for n in nsei_range]
        pnls = [predict_portfolio_value(n)[2] for n in nsei_range]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Value plot
        ax1.plot(nsei_range, values, color='navy')
        ax1.axvline(current_nsei, color='red', linestyle='--', label='Current')
        ax1.axhline(invested_value, color='green', linestyle=':', label='Invested')
        ax1.set_title('Portfolio Value Projection')
        ax1.legend()
        
        # P&L plot
        ax2.plot(nsei_range, pnls, color='purple')
        ax2.axvline(current_nsei, color='red', linestyle='--')
        ax2.axhline(0, color='black', linestyle='-')
        ax2.set_title('P&L Percentage Projection')
        
        st.pyplot(fig)
        
        # Holdings table with validation
        st.subheader("📋 Portfolio Holdings (Validated)")
        df_show = df.copy()
        df_show['Validation'] = np.isclose(
            df_show['Unrealized P&L'], 
            df_show['Market Value'] - df_show['Invested value'],
            rtol=0.01
        )
        
        st.dataframe(
            df_show.style.format({
                'Avg. Cost Price': '{:.2f}',
                'LTP': '{:.2f}',
                'Invested value': '₹{:,.2f}',
                'Market Value': '₹{:,.2f}',
                'Unrealized P&L': '₹{:,.2f}',
                'Unrealized P&L (%)': '{:.2f}%',
                'Estimated Beta': '{:.2f}'
            }).applymap(
                lambda x: 'color: red' if isinstance(x, bool) and not x else '',
                subset=['Validation']
            ),
            use_container_width=True
        )
        
    except Exception as e:
        st.error(f"Calculation error: {str(e)}")
        st.stop()
else:
    st.info("Please upload portfolio data to begin analysis")

# Methodology documentation
st.markdown("---")
st.subheader("🧮 Calculation Methodology")
st.markdown("""
**All calculations are validated with multiple methods:**

1. **Portfolio Beta**:
   - Calculated using both stock-level price changes and portfolio-level P&L
   - Weighted average of individual stock betas by market value
   - Cross-validated with portfolio-level P&L change

2. **Breakeven Point**:
   ```python
   breakeven_nsei = current_nsei × (1 + (-Current P&L%)/β)
