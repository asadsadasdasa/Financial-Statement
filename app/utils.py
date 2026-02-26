import streamlit as st
import pandas as pd

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

def format_currency(val, currency_symbol='USD'):
    """Format a numeric value using K/M suffixes and a currency symbol.

    Handles NaN values gracefully and returns a string, preserving the sign.
    """
    if pd.isna(val):
        return f"{currency_symbol} 0"
    try:
        val = float(val)
    except Exception:
        return f"{currency_symbol} {val}"
    
    sign = '-' if val < 0 else ''
    abs_val = abs(val)
    
    if abs_val >= 1e6:
        return f"{sign}{currency_symbol} {abs_val/1e6:,.2f}M"
    elif abs_val >= 1e3:
        return f"{sign}{currency_symbol} {abs_val/1e3:,.1f}k"
    else:
        return f"{sign}{currency_symbol} {abs_val:,.0f}"

def format_variance(actual, target, currency_symbol='USD', as_percentage=False):
    """Format a variance between actual and target values."""
    if pd.isna(actual) or pd.isna(target):
        return "N/A"
    delta = float(actual) - float(target)
    if as_percentage:
        if target == 0:
            return "N/A"
        pct = (delta / float(target)) * 100
        return f"{pct:+.1f}%"
    else:
        formatted = format_currency(delta, currency_symbol)
        if delta > 0 and not formatted.startswith('+'):
            formatted = '+' + formatted
        return formatted

def format_percentage(val, decimal_places=1):
    """Format a decimal value as a percentage."""
    if pd.isna(val):
        return "0.0%"
    try:
        return f"{float(val)*100:.{decimal_places}f}%"
    except Exception:
        return "N/A"

def variance_basis_points(actual, target):
    """Calculate variance in basis points (1 bps = 0.01%)."""
    if pd.isna(actual) or pd.isna(target):
        return 0
    return (float(actual) - float(target)) * 10000