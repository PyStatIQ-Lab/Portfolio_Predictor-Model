import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
from scipy.optimize import fsolve
from matplotlib.colors import LinearSegmentedColormap

# Set page config
st.set_page_config(page_title="Portfolio NSEI Predictor Pro", layout="wide", page_icon="📊")

# Custom styling
st.markdown("""
<style>
    .stMetricValue {
        font-size: 18px !important;
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
    .css-1aumxhk {
        background-color: #f0f2f6;
    }
    .st-b7 {
        background-color: #f0f2f6;
    }
    .stAlert {
        border-left: 4px solid #ff4b4b;
    }
</style>
""", unsafe_allow_html=True)

# App title
st.title("📊 Portfolio Performance Predictor Pro")
st.markdown("""
Predict how your portfolio will perform at different NSEI (Nifty 50) index levels with accurate breakeven analysis.
""")

# Sidebar for file upload
with st.sidebar:
    st.header("📤 Upload Portfolio Data")
    uploaded_file = st.file_uploader(
        "Choose Excel/CSV file with portfolio data",
        type=["csv", "xlsx"],
        help="File should contain: Symbol, Net Quantity, Avg. Cost Price, LTP, Invested value, Market Value, P&L"
    )
    
    st.header("⚙️ Market Parameters")
    current_nsei = st.number_input(
        "Current NSEI Level",
        min_value=1000.0,
        max_value=50000.0,
        value=22161.0,
        step=100.0
    )
    
    # Sample data
    if st.checkbox("Show sample data format"):
        sample_data = pd.DataFrame({
            'Symbol': ['RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS'],
            'Net Quantity': [10, 15, 5],
            'Avg. Cost Price': [2450.50, 3250.75, 1450.25],
            'LTP': [2600.25, 3100.50, 1500.75],
            'Invested value': [24505.00, 48761.25, 7251.25],
            'Market Value': [26002.50, 46507.50, 7503.75],
            'Unrealized P&L': [1497.50, -2253.75, 252.50],
            'Unrealized P&L (%)': [6.11, -4.62, 3.48]
        })
        st.dataframe(sample_data)

# Main analysis function
def perform_analysis(df, current_nsei):
    # Calculate portfolio metrics
    portfolio_value = float(df['Market Value'].sum())
    invested_value = float(df['Invested value'].sum())
    unrealized_pnl = float(df['Unrealized P&L'].sum())
    unrealized_pnl_pct = (unrealized_pnl / invested_value) * 100
    
    # Display portfolio summary
    st.subheader("📊 Portfolio Summary")
    col1, col2, col3 = st.columns(3)
    col1.metric("Current Value", f"₹{portfolio_value:,.2f}")
    col2.metric("Invested Value", f"₹{invested_value:,.2f}")
    
    pnl_color = "negative" if unrealized_pnl < 0 else "positive"
    col3.metric("Unrealized P&L", 
               f"₹{unrealized_pnl:,.2f}", 
               f"{unrealized_pnl_pct:.2f}%",
               delta_color="inverse")
    
    # Beta calculation (using mock data - replace with real beta API in production)
    beta_map = {
        'RELIANCE.NS': 1.2,
        'TCS.NS': 0.9,
        'HDFCBANK.NS': 1.1,
        'INFY.NS': 0.8,
        'HINDUNILVR.NS': 0.7
    }
    df['Beta'] = df['Symbol'].map(beta_map).fillna(1.0)
    df['Weight'] = df['Market Value'] / portfolio_value
    portfolio_beta = (df['Beta'] * df['Weight']).sum()
    
    st.markdown(f"""
    ### 🔍 Portfolio Beta: {portfolio_beta:.2f}
    - β < 1: Less volatile than market  
    - β = 1: Moves with market  
    - β > 1: More volatile than market
    """)
    
    # Prediction function with proper compounding
    def predict_portfolio_value(target_nsei):
        nsei_return = (target_nsei - current_nsei) / current_nsei
        portfolio_return = portfolio_beta * nsei_return
        predicted_value = portfolio_value * (1 + portfolio_return)
        predicted_pnl = predicted_value - invested_value
        predicted_pnl_pct = (predicted_pnl / invested_value) * 100
        return predicted_value, predicted_pnl, predicted_pnl_pct
    
    # Accurate breakeven calculation
    def calculate_breakeven():
        try:
            # Direct formula solution
            required_return = (-unrealized_pnl_pct/100) / portfolio_beta
            breakeven_nsei = current_nsei * (1 + required_return)
            
            # Verify
            test_value, _, _ = predict_portfolio_value(breakeven_nsei)
            if not np.isclose(test_value, invested_value, rtol=0.01):
                st.warning("Secondary verification failed - using direct formula")
            
            breakeven_change = (breakeven_nsei - current_nsei) / current_nsei * 100
            return breakeven_nsei, breakeven_change
        except Exception as e:
            st.error(f"Breakeven calculation error: {str(e)}")
            return current_nsei, 0.0
    
    breakeven_nsei, breakeven_change = calculate_breakeven()
    
    # Prediction UI
    st.subheader("🔮 Portfolio Prediction")
    target_nsei = st.number_input(
        "Enter target NSEI level:", 
        min_value=1000.0,
        max_value=50000.0,
        value=float(round(breakeven_nsei)),
        step=100.0
    )
    
    if st.button("Calculate Prediction"):
        pred_value, pred_pnl, pred_pnl_pct = predict_portfolio_value(target_nsei)
        
        # Format results with color
        value_change = (pred_value - portfolio_value)/portfolio_value*100
        value_color = "negative" if value_change < 0 else "positive"
        pnl_color = "negative" if pred_pnl < 0 else "positive"
        
        st.markdown(f"""
        <div style="background-color:#f8f9fa;padding:20px;border-radius:10px;margin-bottom:20px;">
            <h4>📌 Prediction at NSEI {target_nsei:,.0f}</h4>
            <p><strong>Predicted Value:</strong> ₹{pred_value:,.2f} 
            (<span class="{value_color}">{value_change:.2f}%</span> from current)</p>
            <p><strong>Projected P&L:</strong> <span class="{pnl_color}">₹{pred_pnl:,.2f} ({pred_pnl_pct:.2f}%)</span></p>
        </div>
        
        <div style="background-color:#fff4e6;padding:20px;border-radius:10px;">
            <h4>⚖️ Breakeven Analysis</h4>
            <p><strong>Breakeven NSEI Level:</strong> <span class="breakeven">{breakeven_nsei:,.2f}</span></p>
            <p><strong>Required Market Change:</strong> <span class="breakeven">{breakeven_change:.2f}%</span></p>
            <p><em>This is where your portfolio value equals your invested amount.</em></p>
        </div>
        """, unsafe_allow_html=True)
    
    # Visualization
    st.subheader("📈 Portfolio Projections")
    nsei_range = np.linspace(
        max(current_nsei * 0.7, 1000),
        current_nsei * 1.3,
        100
    )
    pred_values = [predict_portfolio_value(n)[0] for n in nsei_range]
    pred_pnls = [predict_portfolio_value(n)[1] for n in nsei_range]
    
    # Create visualization
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # Main value line
    ax.plot(nsei_range, pred_values, label='Portfolio Value', color='#1f77b4', linewidth=2.5)
    
    # Breakeven markers
    ax.axvline(x=breakeven_nsei, color='orange', linestyle='--', label='Breakeven Point')
    ax.axhline(y=invested_value, color='green', linestyle=':', label='Invested Value')
    
    # Current market markers
    ax.axvline(x=current_nsei, color='red', linestyle='--', alpha=0.5, label='Current NSEI')
    ax.scatter(current_nsei, portfolio_value, color='red', s=100, zorder=5)
    
    # Formatting
    ax.set_xlabel('NSEI Index Level', fontsize=12)
    ax.set_ylabel('Portfolio Value (₹)', fontsize=12)
    ax.set_title('Portfolio Value Projection', fontsize=14, pad=20)
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Profit/loss regions
    ax.fill_between(nsei_range, pred_values, invested_value, 
                   where=np.array(pred_values)>=invested_value,
                   color='green', alpha=0.1, label='Profit Zone')
    ax.fill_between(nsei_range, pred_values, invested_value,
                   where=np.array(pred_values)<invested_value,
                   color='red', alpha=0.1, label='Loss Zone')
    
    st.pyplot(fig)
    
    # Holdings table
    st.subheader("📋 Portfolio Holdings")
    
    def color_pnl(val):
        color = 'red' if val < 0 else 'green'
        return f'color: {color}'
    
    st.dataframe(
        df.style.format({
            'Avg. Cost Price': '{:.2f}',
            'LTP': '{:.2f}',
            'Invested value': '₹{:,.2f}',
            'Market Value': '₹{:,.2f}',
            'Unrealized P&L': '₹{:,.2f}',
            'Unrealized P&L (%)': '{:.2f}%',
            'Beta': '{:.2f}'
        }).applymap(color_pnl, subset=['Unrealized P&L', 'Unrealized P&L (%)']),
        use_container_width=True
    )

# Main app logic
if uploaded_file is not None:
    try:
        # Read file
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        
        # Validate columns
        required_cols = ['Symbol', 'Net Quantity', 'Avg. Cost Price', 'LTP',
                        'Invested value', 'Market Value', 'Unrealized P&L']
        missing = [col for col in required_cols if col not in df.columns]
        
        if missing:
            st.error(f"Missing columns: {', '.join(missing)}")
        else:
            # Clean data
            num_cols = ['Avg. Cost Price', 'LTP', 'Invested value', 'Market Value', 'Unrealized P&L']
            for col in num_cols:
                if df[col].dtype == 'object':
                    df[col] = df[col].astype(str).str.replace(',', '').astype(float)
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Calculate P% if not present
            if 'Unrealized P&L (%)' not in df.columns:
                df['Unrealized P&L (%)'] = (df['Unrealized P&L'] / df['Invested value']) * 100
            
            perform_analysis(df, current_nsei)
            
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
else:
    st.info("ℹ️ Please upload your portfolio file to begin analysis")
    st.image("https://via.placeholder.com/800x400?text=Upload+your+portfolio+CSV/Excel+file", use_column_width=True)

# Footer
st.markdown("---")
st.markdown("""
**Methodology Notes:**
1. Portfolio beta calculated as weighted average of individual stock betas
2. Breakeven point calculated using: NSEI_BE = Current × (1 + (-Current P&L%)/β)
3. All projections assume linear relationship between NSEI and portfolio returns
""")
