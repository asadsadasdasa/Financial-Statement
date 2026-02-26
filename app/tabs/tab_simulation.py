import streamlit as st
import pandas as pd
from data_engine import apply_fiscal_calendar

def render_simulation():
    st.info("💡 Use this area to download standard templates, fill them with new scenarios, and inject them into the dashboard.")
    col_t1, col_t2 = st.columns(2)
    
    # 1. Template de Forecast
    df_template_fc = pd.DataFrame(columns=['Company Code', 'Company', 'Profit Center', 'WBS element', 'Revenue Type', 'Direct/Ind./Interco', 'Sales Rep', 'Currency', 'Backlog/Pipeline', 'Posting Date', 'Amount in local currency', 'Scenario'])
    csv_fc = df_template_fc.to_csv(index=False).encode('utf-8')
    
    with col_t1:
        st.markdown("#### 1. Revenue / Forecast Template")
        st.download_button("📥 Download Forecast Template", data=csv_fc, file_name="Template_Simulation_Forecast.csv", mime="text/csv")
        up_fc = st.file_uploader("Upload Forecast Scenario", type=['csv', 'xlsx'], key='up_fc')
        if up_fc:
            try:
                df_up_fc = pd.read_csv(up_fc, encoding='latin1') if up_fc.name.endswith('.csv') else pd.read_excel(up_fc)
                df_up_fc = apply_fiscal_calendar(df_up_fc)
                if st.button("Inject Forecast Data"):
                    st.session_state['uploaded_fc'] = df_up_fc 
                    st.success("New Revenue Scenario Injected!")
                    st.rerun()
            except Exception as e: st.error(f"Error: {e}")

    # 2. Template de Capacity
    df_template_cap = pd.DataFrame(columns=['Employee ID', 'Employee Name', 'Company', 'Position Title', 'Profit Center', 'Posting Date', 'Target Capacity EUR'])
    csv_cap = df_template_cap.to_csv(index=False).encode('utf-8')
    
    with col_t2:
        st.markdown("#### 2. Capacity / HR Template")
        st.download_button("📥 Download Capacity Template", data=csv_cap, file_name="Template_Simulation_Capacity.csv", mime="text/csv")
        up_cap = st.file_uploader("Upload Capacity Scenario", type=['csv', 'xlsx'], key='up_cap')
        if up_cap:
            try:
                df_up_cap = pd.read_csv(up_cap, encoding='latin1') if up_cap.name.endswith('.csv') else pd.read_excel(up_cap)
                df_up_cap = apply_fiscal_calendar(df_up_cap)
                if st.button("Inject Capacity Data"):
                    st.session_state['uploaded_cap'] = df_up_cap 
                    st.success("New Capacity Scenario Injected!")
                    st.rerun()
            except Exception as e: st.error(f"Error: {e}")
            
    if not st.session_state['uploaded_fc'].empty or not st.session_state['uploaded_cap'].empty:
        st.markdown("---")
        if st.button("🗑️ Clear All Injected Scenarios"):
            st.session_state['uploaded_fc'] = pd.DataFrame()
            st.session_state['uploaded_cap'] = pd.DataFrame()
            st.rerun()