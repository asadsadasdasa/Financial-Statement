import streamlit as st
import pandas as pd
from utils import inject_custom_css, ordem_meses
from data_engine import load_master_data, filter_and_convert

# Importação dos Módulos das Abas
from tabs.tab_pl import render_pl
from tabs.tab_capacity import render_capacity
from tabs.tab_simulation import render_simulation

# --- CONFIGURAÇÃO ---
st.set_page_config(page_title="HR Path - CFO Cockpit", page_icon="📈", layout="wide")
inject_custom_css()

st.title("📈 CFO Intelligence Cockpit")
st.markdown("##### North America Region | FP&A and Delivery Risk Analysis")

# --- CARREGAR DADOS ---
df_base, df_forecast, df_capacity, df_budget, df_currency = load_master_data()

# INTEGRAÇÃO DE UPLOADS E SIMULAÇÃO NA MEMÓRIA
if 'uploaded_fc' not in st.session_state: st.session_state['uploaded_fc'] = pd.DataFrame()
if 'uploaded_cap' not in st.session_state: st.session_state['uploaded_cap'] = pd.DataFrame()

if not st.session_state['uploaded_fc'].empty: 
    df_forecast = pd.concat([df_forecast, st.session_state['uploaded_fc']], ignore_index=True)
if not st.session_state['uploaded_cap'].empty:
    df_capacity = pd.concat([df_capacity, st.session_state['uploaded_cap']], ignore_index=True)

# --- FILTROS SUPERIORES ---
st.markdown("---")
col_f1, col_f2, col_f3, col_f4, col_f5 = st.columns(5)

get_unq = lambda df, col: df[col].dropna().unique().tolist() if not df.empty and col in df.columns else []

empresas = ['All'] + sorted(list(set(get_unq(df_base, 'Company Code') + get_unq(df_forecast, 'Company Code') + get_unq(df_capacity, 'Company'))))
emp_sel = col_f1.selectbox("🏢 Entity / Company", empresas)

centros = ['All'] + sorted(list(set(get_unq(df_base, 'Profit Center') + get_unq(df_forecast, 'Profit Center') + get_unq(df_capacity, 'Profit Center'))))
pc_sel = col_f2.selectbox("🎯 Practice (LOB)", centros)

moedas = get_unq(df_currency, 'Moeda_Destino') if not df_currency.empty else ['EUR', 'USD', 'CAD']
moeda_sel = col_f3.selectbox("💱 Currency", moedas)

anos = sorted(list(set([y for y in get_unq(df_base, 'Fiscal Year') + get_unq(df_forecast, 'Fiscal Year') if y != 'Unknown'])))
fy_sel = col_f4.selectbox("📅 Fiscal Year", anos) if anos else "FY25/26"

mes_sel = col_f5.selectbox("📆 Fiscal Period", ['All', 'YTD'] + ordem_meses)
st.markdown("---")

# --- APLICAR FILTROS GLOBAIS ---
df_act_fil = filter_and_convert(df_base, df_currency, fy_sel, mes_sel, emp_sel, pc_sel, moeda_sel)
df_fc_fil = filter_and_convert(df_forecast, df_currency, fy_sel, mes_sel, emp_sel, pc_sel, moeda_sel)
df_cap_fil = filter_and_convert(df_capacity, df_currency, fy_sel, mes_sel, emp_sel, pc_sel, moeda_sel, val_col='Target Capacity EUR', eur_base=True)

# --- ABAS (CHAMANDO OS MÓDULOS) ---
tab1, tab2, tab3 = st.tabs(["📊 P&L & Profitability", "⚖️ Capacity vs. Delivery Risk", "📤 Upload Data (Simulation)"])

with tab1: render_pl(df_act_fil, moeda_sel)
with tab2: render_capacity(df_fc_fil, df_cap_fil, df_act_fil, moeda_sel)
with tab3: render_simulation()