import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from utils import format_currency

def render_pl(df_act_fil, moeda_sel):
    st.markdown("### Historical Financial Performance (Full P&L)")
    
    if df_act_fil.empty:
        st.info("No realized financial data available for the applied filters.")
        return
        
    df_act_fil['Amount Final'] *= -1 
    
    # Busca dinâmica na árvore do P&L (Mapeamento do SAP)
    def get_pl(lvl): return df_act_fil[df_act_fil['P&L LVL 5'] == lvl]['Amount Final'].sum() if 'P&L LVL 5' in df_act_fil.columns else 0

    # KPIs Completos Restaurados
    rev = get_pl('Net Sales')
    opex = get_pl('Operating Expenses')  # Cost of Delivery / Direct Costs
    gm = rev + opex
    sga = get_pl('General and Admin')
    ebitda = gm + sga
    depre = get_pl('Depreciation and Amortization')
    ebit = ebitda + depre

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Billed Revenue", format_currency(rev, moeda_sel))
    k2.metric("Gross Margin", format_currency(gm, moeda_sel), f"{(gm/rev*100) if rev else 0:.1f}%")
    k3.metric("EBITDA", format_currency(ebitda, moeda_sel), f"{(ebitda/rev*100) if rev else 0:.1f}%")
    k4.metric("EBIT (Operating Profit)", format_currency(ebit, moeda_sel), f"{(ebit/rev*100) if rev else 0:.1f}%")
    
    st.markdown("---")
    
    c1, c2 = st.columns([1, 1.5])
    with c1:
        st.markdown("#### P&L Statement Details")
        pl_df = pd.DataFrame([
            ("1. Net Sales (Revenue)", rev), 
            ("2. Cost of Delivery (Direct)", opex), 
            ("GROSS MARGIN", gm),
            ("3. SG&A (Indirect Costs)", sga), 
            ("EBITDA", ebitda),
            ("4. Depreciation & Amortization", depre),
            ("EBIT (Operating Profit)", ebit)
        ], columns=["Account", "Amount"])
        pl_df["% Margin"] = (pl_df["Amount"] / rev * 100).fillna(0).map("{:.1f}%".format)
        pl_df["Amount"] = pl_df["Amount"].map(lambda x: format_currency(x, moeda_sel))
        st.dataframe(pl_df, use_container_width=True, hide_index=True)
        
    with c2:
        st.markdown("#### Full Profitability Waterfall")
        fig_wf = go.Figure(go.Waterfall(
            measure=["relative", "relative", "total", "relative", "total", "relative", "total"],
            x=["Revenue", "Delivery Cost", "Gross Mrg", "SG&A", "EBITDA", "D&A", "EBIT"],
            y=[rev, opex, gm, sga, ebitda, depre, ebit],
            text=[format_currency(v, moeda_sel) for v in [rev, opex, gm, sga, ebitda, depre, ebit]], 
            textposition="outside",
            connector={"line":{"color":"rgb(63, 63, 63)"}},
            decreasing={"marker":{"color":"#ef553b"}}, 
            increasing={"marker":{"color":"#002a4d"}}, 
            totals={"marker":{"color":"#00cc96"}} 
        ))
        fig_wf.update_layout(showlegend=False, plot_bgcolor='white', margin=dict(t=20, b=20))
        st.plotly_chart(fig_wf, use_container_width=True)