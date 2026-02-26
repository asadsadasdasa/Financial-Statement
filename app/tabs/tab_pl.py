import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from utils import format_currency, format_variance, variance_basis_points, ordem_meses

# Corporate colour palette
COLOUR_PRIMARY = '#002a4d'
COLOUR_GRAY = '#7f8c8d'
COLOUR_SOFT_BLUE = '#3498db'
COLOUR_GREEN = '#2ecc71'
COLOUR_RED = '#e74c3c'

def render_pl(df_act_fil, df_forecast, df_budget, moeda_sel):
    """
    TAB 1: P&L & Variance Analysis
    Delivers executive narratives on YTD performance, margin variance, OpEx burn, and FY landing.
    """
    st.markdown("### P&L & Variance Intelligence")
    
    if df_act_fil.empty:
        st.info("No realized financial data available for the applied filters.")
        return
    
    # Use Amount Normalized if available (sign-corrected), else Amount Final
    amount_col = 'Amount Normalized' if 'Amount Normalized' in df_act_fil.columns else 'Amount Final'
    
    # === HELPER: Extract P&L line items ===
    def get_pl(lvl, df=None):
        """Get P&L line sum by level (Revenue, Operating Expenses, etc.)"""
        src = df if df is not None else df_act_fil
        if src.empty or 'P&L LVL 5' not in src.columns:
            return 0
        result = src[src['P&L LVL 5'] == lvl][amount_col].sum()
        return result if not pd.isna(result) else 0
    
    # === ROW 1: YTD EXECUTIVE KPIs vs Budget ===
    st.markdown("#### YTD Summary (Year-to-Date Performance)")
    
    rev_ytd = get_pl('Net Sales')
    opex_ytd = get_pl('Operating Expenses')
    gm_ytd = rev_ytd + opex_ytd
    sga_ytd = get_pl('General and Admin')
    ebitda_ytd = gm_ytd + sga_ytd
    
    # Budget comparison (if budget data available)
    budget_rev = get_pl('Net Sales', df_budget) if not df_budget.empty else rev_ytd
    budget_gm = (get_pl('Net Sales', df_budget) + get_pl('Operating Expenses', df_budget)) if not df_budget.empty else gm_ytd
    
    gm_pct_ytd = (gm_ytd / rev_ytd * 100) if rev_ytd != 0 else 0
    gm_pct_budget = (budget_gm / budget_rev * 100) if budget_rev != 0 else 0
    margin_var_bps = (gm_pct_ytd - gm_pct_budget) * 100  # Convert % to basis points
    rev_var_pct = ((rev_ytd - budget_rev) / budget_rev * 100) if budget_rev != 0 else 0
    
    # FY Forecast (estimate based on YTD pace)
    current_month = len([x for x in ordem_meses if df_act_fil['Fiscal Period'].isin([x]).any()])
    fy_forecast_rev = (rev_ytd / current_month * 12) if current_month > 0 else rev_ytd
    
    k1, k2, k3, k4 = st.columns(4)
    
    with k1:
        st.metric(
            "YTD Revenue",
            format_currency(rev_ytd, moeda_sel),
            f"{rev_var_pct:+.1f}% vs Budget",
            delta_color="normal" if rev_var_pct >= 0 else "inverse"
        )
    
    with k2:
        st.metric(
            "Gross Margin %",
            f"{gm_pct_ytd:.1f}%",
            f"{margin_var_bps:+.0f} bps",
            delta_color="normal" if margin_var_bps >= 0 else "inverse"
        )
    
    with k3:
        st.metric(
            "OpEx Burn Rate",
            f"{(opex_ytd / rev_ytd * 100) if rev_ytd != 0 else 0:.1f}%",
            "vs Revenue",
            delta_color="inverse"  # Lower is better
        )
    
    with k4:
        st.metric(
            "FY Forecast Landing",
            format_currency(fy_forecast_rev, moeda_sel),
            f"Annualized",
            delta_color="off"
        )
    
    st.markdown("---")
    
    # === ROW 2: OPEX BURN RATE TREND ===
    st.markdown("#### OpEx Burn Rate Trend")
    
    burn_by_period = []
    for period in ordem_meses:
        period_data = df_act_fil[df_act_fil['Fiscal Period'] == period]
        p_rev = get_pl('Net Sales', period_data)
        p_opex = get_pl('Operating Expenses', period_data)
        p_burn = (p_opex / p_rev * 100) if p_rev > 0 else 0
        burn_by_period.append({'Period': period, 'Burn Rate %': p_burn, 'Revenue': p_rev, 'OpEx': p_opex})
    
    burn_df = pd.DataFrame(burn_by_period)
    
    fig_burn = go.Figure()
    fig_burn.add_trace(go.Scatter(
        x=burn_df['Period'], y=burn_df['Burn Rate %'], mode='lines+markers',
        name='OpEx Burn Rate', line=dict(color=COLOUR_RED, width=2),
        marker=dict(size=6, symbol='circle'),
        hovertemplate='<b>%{x}</b><br>Burn Rate: %{y:.1f}%<extra></extra>'
    ))
    fig_burn.add_hline(
        y=30, line_dash="dash", line_color=COLOUR_GRAY, 
        annotation_text="Target Burn (30%)", annotation_position="right"
    )
    fig_burn.update_layout(
        hovermode='x unified',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(showgrid=False, zeroline=False),
        yaxis=dict(showgrid=False, zeroline=False),
        margin=dict(t=30, b=30, l=60, r=80),
        height=300
    )
    st.plotly_chart(fig_burn, use_container_width=True)
    
    st.markdown("---")
    
    # === ROW 3: ADVANCED WATERFALL + MONTHLY REVENUE COMBO ===
    col_wf, col_combo = st.columns([1, 1.5])
    
    with col_wf:
        st.markdown("#### P&L Waterfall (YTD)")
        
        # Waterfall structure: Revenue → Opex → Gross Margin → SGA → EBITDA → D&A → EBIT
        depre_ytd = get_pl('Depreciation and Amortization')
        ebit_ytd = ebitda_ytd + depre_ytd
        
        fig_wf = go.Figure(go.Waterfall(
            measure=["relative", "relative", "total", "relative", "total", "relative", "total"],
            x=["Revenue", "Delivery\nCost", "Gross\nMargin", "SG&A", "EBITDA", "D&A", "EBIT"],
            y=[rev_ytd, opex_ytd, gm_ytd, sga_ytd, ebitda_ytd, depre_ytd, ebit_ytd],
            text=[
                format_currency(v, moeda_sel) for v in [rev_ytd, opex_ytd, gm_ytd, sga_ytd, ebitda_ytd, depre_ytd, ebit_ytd]
            ],
            textposition="outside",
            connector={"line": {"color": COLOUR_GRAY}},
            decreasing={"marker": {"color": COLOUR_RED}},
            increasing={"marker": {"color": COLOUR_PRIMARY}},
            totals={"marker": {"color": COLOUR_GREEN}}
        ))
        fig_wf.update_layout(
            showlegend=False,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=False),
            margin=dict(t=20, b=20, l=50, r=50),
            height=400
        )
        st.plotly_chart(fig_wf, use_container_width=True)
    
    with col_combo:
        st.markdown("#### Monthly Revenue vs Forecast & Budget")
        
        # Monthly decomposition for combo chart
        monthly_data = []
        for period in ordem_meses:
            actuals = get_pl('Net Sales', df_act_fil[df_act_fil['Fiscal Period'] == period])
            forecast = get_pl('Net Sales', df_forecast[df_forecast['Fiscal Period'] == period]) if not df_forecast.empty else actuals
            budget = get_pl('Net Sales', df_budget[df_budget['Fiscal Period'] == period]) if not df_budget.empty else actuals
            monthly_data.append({
                'Period': period,
                'Actuals': actuals,
                'Forecast': forecast,
                'Budget': budget
            })
        
        monthly_df = pd.DataFrame(monthly_data)
        
        fig_combo = go.Figure()
        
        # Bar chart for Actuals
        fig_combo.add_trace(go.Bar(
            x=monthly_df['Period'], y=monthly_df['Actuals'], name='Actuals',
            marker_color=COLOUR_PRIMARY, opacity=0.8,
            hovertemplate='<b>Actuals</b><br>%{y:,.0f}<extra></extra>'
        ))
        
        # Step-line for Budget target
        fig_combo.add_trace(go.Scatter(
            x=monthly_df['Period'], y=monthly_df['Budget'], name='Budget Target',
            mode='lines', line=dict(color=COLOUR_SOFT_BLUE, width=2, dash='solid'),
            hovertemplate='<b>Budget</b><br>%{y:,.0f}<extra></extra>'
        ))
        
        # Step-line for Forecast (if different)
        if not (monthly_df['Forecast'] == monthly_df['Budget']).all():
            fig_combo.add_trace(go.Scatter(
                x=monthly_df['Period'], y=monthly_df['Forecast'], name='Forecast',
                mode='lines', line=dict(color=COLOUR_GREEN, width=2, dash='dash'),
                hovertemplate='<b>Forecast</b><br>%{y:,.0f}<extra></extra>'
            ))
        
        fig_combo.update_layout(
            barmode='overlay',
            hovermode='x unified',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(showgrid=False, zeroline=False),
            yaxis=dict(showgrid=False, zeroline=False),
            legend=dict(orientation='h', yanchor='bottom', y=1.01, xanchor='left', x=0),
            margin=dict(t=30, b=30, l=60, r=50),
            height=400
        )
        st.plotly_chart(fig_combo, use_container_width=True)
    
    st.markdown("---")
    
    # === ROW 4: DETAILED P&L STATEMENT ===
    st.markdown("#### Detailed P&L Statement (YTD)")
    
    pl_detail = pd.DataFrame([
        ("1. Net Sales (Revenue)", rev_ytd),
        ("2. Cost of Delivery (Direct Costs)", opex_ytd),
        ("= GROSS MARGIN", gm_ytd),
        ("3. SG&A (Indirect Costs)", sga_ytd),
        ("= EBITDA", ebitda_ytd),
        ("4. Depreciation & Amortization", depre_ytd),
        ("= EBIT (Operating Profit)", ebit_ytd)
    ], columns=["Line Item", "Amount YTD"])
    
    pl_detail["Margin %"] = (pl_detail["Amount YTD"] / rev_ytd * 100).fillna(0).apply(lambda x: f"{x:.1f}%")
    pl_detail["Amount YTD"] = pl_detail["Amount YTD"].apply(lambda x: format_currency(x, moeda_sel))
    
    st.dataframe(pl_detail, use_container_width=True, hide_index=True, column_config={
        "Line Item": st.column_config.TextColumn(width=None),
        "Amount YTD": st.column_config.TextColumn(width=150),
        "Margin %": st.column_config.TextColumn(width=100)
    })