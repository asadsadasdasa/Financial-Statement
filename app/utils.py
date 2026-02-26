import streamlit as st

# Dicionários globais de tempo
meses_fiscais = {4:'01-Apr', 5:'02-May', 6:'03-Jun', 7:'04-Jul', 8:'05-Aug', 9:'06-Sep', 10:'07-Oct', 11:'08-Nov', 12:'09-Dec', 1:'10-Jan', 2:'11-Feb', 3:'12-Mar'}
ordem_meses = ['01-Apr', '02-May', '03-Jun', '04-Jul', '05-Aug', '06-Sep', '07-Oct', '08-Nov', '09-Dec', '10-Jan', '11-Feb', '12-Mar']

def inject_custom_css():
    st.markdown("""
    <style>
        div[data-testid="metric-container"] { background-color: #ffffff; border-left: 5px solid #002a4d; border-radius: 5px; padding: 15px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
        .stSelectbox { margin-bottom: -15px; }
        h1, h2, h3, h4 { color: #002a4d; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }
    </style>
    """, unsafe_allow_html=True)

def format_currency(val, currency_symbol):
    """Formata valores financeiros com K e M dinamicamente"""
    if abs(val) >= 1e6:
        return f"{currency_symbol} {val/1e6:,.2f}M"
    else:
        return f"{currency_symbol} {val/1e3:,.1f}k"