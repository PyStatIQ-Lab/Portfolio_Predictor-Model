import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
from scipy.optimize import fsolve
from matplotlib.colors import LinearSegmentedColormap

# Set page config
st.set_page_config(page_title="Portfolio NSEI Predictor", layout="wide", page_icon="📊")

# Custom styling
st.markdown("""
<style>
    .stMetricValue {
        font-size: 18px !important;
    }
    .st-b7 {
        background-color: #f0f2f6;
    }
    .css-1aumxhk {
        background-color: #f0f2f6;
        background-image: none;
    }
    .positive {
        color: green !important;
        font-weight: bold;
    }
    .negative {
        color: red !important;
        font-weight: bold;
    }
    .breakeven {
        color: orange !important;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# App title
st.title("📊 Portfolio Performance Predictor Based on NSEI Levels")
st.markdown("""
Predict how your portfolio will perform at different NSEI (Nifty 50) index levels.
Upload your portfolio file and enter target NSEI values to see predictions.
""")

# Sidebar for file upload
with st.sidebar:
    st.header("📤 Upload Portfolio Data")
    uploaded_file = st.file_uploader(
        "Choose Excel/CSV file with portfolio data",
        type=["csv", "xlsx"],
        help="File should contain columns: Symbol, Net Quantity, Avg. Cost Price, LTP, Invested value, Market Value, Unrealized P&L, Unrealized P&L (%)"
    )
    
    st.header("⚙️ Settings")
    current_nsei = st.number_input(
        "Current NSEI Level",
        min_value=1000.0,
        max_value=50000.0,
        value=22161.0,
        step=100.0,
        help="Enter the current NSEI index level"
    )
    
    # Sample data download
    st.markdown("### 🧪 Need sample data?")
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
        label="📥 Download Sample CSV",
        data=csv,
        file_name="sample_portfolio.csv",
        mime="text/csv",
        help="Download a sample portfolio file to test the app"
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
            st.error(f"❌ Missing required columns: {', '.join(missing)}")
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
        
        # Display portfolio summary with color coding
        st.subheader("📊 Portfolio Summary")
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Current Portfolio Value", f"₹{portfolio_value:,.2f}")
        
        # Color code based on profit/loss
        pnl_color_class = "negative" if unrealized_pnl < 0 else "positive"
        pnl_icon = "🔴" if unrealized_pnl < 0 else "🟢"
        
        col2.metric("Invested Value", f"₹{invested_value:,.2f}")
        col3.metric("Unrealized P&L", 
                   f"{pnl_icon} ₹{unrealized_pnl:,.2f}", 
                   f"{unrealized_pnl_pct:.2f}%",
                   delta_color="inverse")
        
        # Calculate portfolio beta (improved approach)
        st.subheader("🔍 Portfolio Sensitivity Analysis")
        
        # Improved beta calculation using individual stock betas (simulated)
        # In a real app, you would get these from an API
        simulated_betas = {
            'STAR.NS': 1.2,
            'ORCHPHARMA.NS': 0.8,
            'APARINDS.NS': 1.5
        }
        
        # Assign betas to stocks (using simulated values for demo)
        df['Beta'] = df['Symbol'].map(simulated_betas).fillna(1.0)  # Default to 1 if not found
        
        # Calculate weighted portfolio beta
        df['Weight'] = df['Market Value'] / portfolio_value
        portfolio_beta = (df['Beta'] * df['Weight']).sum()
        
        st.write(f"📈 Estimated Portfolio Beta: {portfolio_beta:.2f}")
        st.caption("""
        Beta measures your portfolio's sensitivity to NSEI movements. 
        - β < 1: Less volatile than market
        - β = 1: Moves with market
        - β > 1: More volatile than market
        """)
        
        # Prediction function
        def predict_portfolio_value(target_nsei):
            target_nsei = float(target_nsei)
            nsei_return_pct = ((target_nsei - current_nsei) / current_nsei) * 100
            portfolio_return_pct = portfolio_beta * nsei_return_pct
            predicted_portfolio_value = portfolio_value * (1 + portfolio_return_pct/100)
            predicted_pnl = predicted_portfolio_value - invested_value
            predicted_pnl_pct = (predicted_pnl / invested_value) * 100
            return float(predicted_portfolio_value), float(predicted_pnl), float(predicted_pnl_pct)
        
        # Find breakeven point
        def breakeven_equation(nsei_level):
            port_value, _, _ = predict_portfolio_value(float(nsei_level[0]))
            return port_value - invested_value
        
        # Numerical solution for breakeven
        breakeven_nsei = float(fsolve(breakeven_equation, current_nsei)[0])
        breakeven_change_pct = ((breakeven_nsei - current_nsei) / current_nsei) * 100
        
        # User input for prediction
        st.subheader("🔮 Portfolio Prediction")
        
        col1, col2 = st.columns([2, 1])
        with col1:
            target_nsei = st.number_input(
                "Enter target NSEI level for prediction:",
                min_value=1000.0,
                max_value=50000.0,
                value=float(round(breakeven_nsei)),
                step=100.0
            )
        
        if st.button("🚀 Predict Portfolio Value", help="Calculate projected portfolio value at target NSEI level"):
            pred_value, pred_pnl, pred_pnl_pct = predict_portfolio_value(target_nsei)
            
            # Determine color classes
            value_change_pct = (pred_value - portfolio_value)/portfolio_value*100
            value_color = "negative" if value_change_pct < 0 else "positive"
            pnl_color = "negative" if pred_pnl < 0 else "positive"
            
            # Format numbers with color
            value_change_str = f"<span class='{value_color}'>{value_change_pct:.2f}%</span>"
            pnl_str = f"<span class='{pnl_color}'>₹{pred_pnl:,.2f}</span>"
            pnl_pct_str = f"<span class='{pnl_color}'>{pred_pnl_pct:.2f}%</span>"
            
            # Display metrics with HTML for color
            st.markdown(f"""
            <div style="background-color:#f8f9fa;padding:20px;border-radius:10px;">
                <h4>Prediction at NSEI {target_nsei:,.0f}</h4>
                <p><strong>Predicted Portfolio Value:</strong> ₹{pred_value:,.2f} (Change: {value_change_str})</p>
                <p><strong>Predicted Unrealized P&L:</strong> {pnl_str} ({pnl_pct_str})</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Breakeven information with color
            breakeven_color = "breakeven"
            st.markdown(f"""
            <div style="margin-top:20px;">
                <h4>Breakeven Analysis</h4>
                <p>Your portfolio will <strong><span class='{breakeven_color}'>break even</span></strong> at:</p>
                <p><strong>NSEI Level:</strong> <span class='{breakeven_color}'>{breakeven_nsei:,.2f}</span></p>
                <p><strong>Required Change:</strong> <span class='{breakeven_color}'>{breakeven_change_pct:.2f}%</span> from current</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Generate predictions for visualization
        st.subheader("📈 Portfolio Projections")
        
        # Create a range of NSEI values for visualization
        nsei_range = np.linspace(
            max(current_nsei * 0.5, 1000),  # Don't go below 1000
            current_nsei * 1.5, 
            50
        ).astype(float)
        
        predicted_values = [predict_portfolio_value(n)[0] for n in nsei_range]
        predicted_pnls = [predict_portfolio_value(n)[1] for n in nsei_range]
        predicted_pnl_pcts = [predict_portfolio_value(n)[2] for n in nsei_range]
        
        # Create figure with custom styling
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Custom colormap for positive/negative regions
        cmap = LinearSegmentedColormap.from_list('pnl_cmap', ['red', 'white', 'green'])
        
        # Portfolio value plot
        ax1.plot(nsei_range, predicted_values, color='blue', linewidth=2.5, label='Portfolio Value')
        ax1.axvline(x=current_nsei, color='red', linestyle='--', linewidth=1.5, label='Current NSEI')
        ax1.axvline(x=breakeven_nsei, color='orange', linestyle='--', linewidth=1.5, label='Breakeven NSEI')
        ax1.axhline(y=invested_value, color='green', linestyle='--', linewidth=1.5, label='Invested Value')
        
        # Fill between for profit/loss regions
        ax1.fill_between(nsei_range, predicted_values, invested_value, 
                        where=np.array(predicted_values) >= invested_value,
                        color='green', alpha=0.1, label='Profit Region')
        ax1.fill_between(nsei_range, predicted_values, invested_value, 
                        where=np.array(predicted_values) < invested_value,
                        color='red', alpha=0.1, label='Loss Region')
        
        ax1.set_xlabel('NSEI Level', fontsize=12)
        ax1.set_ylabel('Portfolio Value (₹)', fontsize=12)
        ax1.set_title('Portfolio Value vs NSEI Level', fontsize=14, pad=20)
        ax1.legend(fontsize=10)
        ax1.grid(True, linestyle='--', alpha=0.7)
        
        # P&L percentage plot with color gradient
        sc = ax2.scatter(nsei_range, predicted_pnl_pcts, c=predicted_pnl_pcts, 
                        cmap=cmap, vmin=-100, vmax=100, s=50)
        ax2.plot(nsei_range, predicted_pnl_pcts, color='grey', alpha=0.3)
        ax2.axvline(x=current_nsei, color='red', linestyle='--', linewidth=1.5, label='Current NSEI')
        ax2.axvline(x=breakeven_nsei, color='orange', linestyle='--', linewidth=1.5, label='Breakeven NSEI')
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.5)
        
        # Add colorbar
        cbar = plt.colorbar(sc, ax=ax2)
        cbar.set_label('P&L %', rotation=270, labelpad=15)
        
        ax2.set_xlabel('NSEI Level', fontsize=12)
        ax2.set_ylabel('Unrealized P&L (%)', fontsize=12)
        ax2.set_title('Portfolio P&L % vs NSEI Level', fontsize=14, pad=20)
        ax2.legend(fontsize=10)
        ax2.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Show portfolio holdings with color formatting
        st.subheader("📋 Your Portfolio Holdings")
        
        # Format the DataFrame display
        def color_pnl(val):
            color = 'red' if val < 0 else 'green'
            return f'color: {color}'
        
        styled_df = df.style \
            .format({
                'Avg. Cost Price': '{:.2f}',
                'LTP': '{:.2f}',
                'Invested value': '₹{:,.2f}',
                'Market Value': '₹{:,.2f}',
                'Unrealized P&L': '₹{:,.2f}',
                'Unrealized P&L (%)': '{:.2f}%',
                'Beta': '{:.2f}'
            }) \
            .applymap(color_pnl, subset=['Unrealized P&L', 'Unrealized P&L (%)']) \
            .background_gradient(cmap=cmap, subset=['Unrealized P&L (%)'], vmin=-100, vmax=100)
        
        st.dataframe(styled_df, use_container_width=True)
        
        # Add export button for predictions
        st.markdown("---")
        st.subheader("📤 Export Predictions")
        
        # Create prediction table
        prediction_table = pd.DataFrame({
            'NSEI Level': nsei_range,
            'Predicted Portfolio Value': predicted_values,
            'Predicted P&L (₹)': predicted_pnls,
            'Predicted P&L (%)': predicted_pnl_pcts
        })
        
        # Export options
        col1, col2 = st.columns(2)
        with col1:
            csv = prediction_table.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="💾 Download Predictions as CSV",
                data=csv,
                file_name="portfolio_predictions.csv",
                mime="text/csv"
            )
        
        with col2:
            excel_buffer = BytesIO()
            with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
                prediction_table.to_excel(writer, index=False, sheet_name='Predictions')
                df.to_excel(writer, index=False, sheet_name='Portfolio')
            excel_bytes = excel_buffer.getvalue()
            st.download_button(
                label="💾 Download Full Report (Excel)",
                data=excel_bytes,
                file_name="portfolio_analysis.xlsx",
                mime="application/vnd.ms-excel"
            )
    
    except Exception as e:
        st.error(f"❌ Error processing file: {str(e)}")
        st.exception(e)
else:
    st.info("ℹ️ Please upload your portfolio file to get started")
    st.image("https://via.placeholder.com/800x400?text=Upload+your+portfolio+CSV/Excel+file", use_column_width=True)

# Add some footer information
st.markdown("---")
st.caption("""
📝 **Note**: This tool provides estimates based on simplified assumptions. 
Actual market performance may vary due to many factors including:
- Individual stock fundamentals
- Market sentiment
- Economic conditions
- Portfolio rebalancing

🔍 **Calculation Methodology**:
1. Portfolio beta is calculated as the weighted average of individual stock betas
2. Predicted returns are calculated as: Portfolio Return = Beta × (NSEI Return)
3. Breakeven point is calculated numerically as the NSEI level where portfolio value equals invested value
""")
