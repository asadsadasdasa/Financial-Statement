import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from utils import format_currency, ordem_meses

# Corporate colour palette
COLOUR_PRIMARY = '#002a4d'
COLOUR_GRAY = '#7f8c8d'
COLOUR_SOFT_BLUE = '#3498db'
COLOUR_GREEN = '#2ecc71'
COLOUR_RED = '#e74c3c'

def render_revenue_pipeline(df_forecast, df_actuals, moeda_sel):
    """
    TAB 3: Revenue Pipeline & Backlog Health
    Delivers narratives on pipeline coverage, at-risk accounts, revenue mix, and forward visibility.
    """
    st.markdown("### Revenue Pipeline: Backlog & Forward Visibility")
    
    if df_forecast.empty:
        st.info("No forecast/pipeline data available for the applied filters.")
        return
    
    # Use Amount Final (already converted/filtered by filter_and_convert)  
    amount_col = 'Amount Final' if 'Amount Final' in df_forecast.columns else 'Amount in local currency'
    
    # === DATA PREPARATION ===
    df_act_rev = pd.DataFrame()
    if not df_actuals.empty and 'WBS element' in df_actuals.columns:
        df_act_rev = df_actuals[
            (df_actuals['P&L LVL 5'] == 'Net Sales') & 
            (df_actuals['WBS element'].notna()) & 
            (df_actuals['WBS element'] != '')
        ].copy()
        if 'Amount Normalized' in df_act_rev.columns:
            df_act_rev['Amount Final'] = df_act_rev['Amount Normalized'].abs()
        else:
            df_act_rev['Amount Final'] = df_act_rev['Amount Final'].abs()
    
    # Separate backlog and pipeline
    df_backlog = df_forecast[df_forecast['Backlog/Pipeline'] == 'Backlog'].copy() if not df_forecast.empty else pd.DataFrame()
    df_pipeline = df_forecast[df_forecast['Backlog/Pipeline'] == 'Pipeline'].copy() if not df_forecast.empty else pd.DataFrame()
    
    tot_act = df_act_rev['Amount Final'].sum() if not df_act_rev.empty else 0
    tot_backlog = df_backlog[amount_col].sum() if not df_backlog.empty else 0
    tot_pipeline = df_pipeline[amount_col].sum() if not df_pipeline.empty else 0
    
    total_revenue_forward = tot_act + tot_backlog + tot_pipeline
    backlog_months = (tot_backlog / (tot_act / 12)) if tot_act > 0 else 0
    pipeline_coverage = (tot_pipeline / tot_backlog) if tot_backlog > 0 else 0
    backlog_pct = (tot_backlog / total_revenue_forward * 100) if total_revenue_forward > 0 else 0
    
    # === ROW 1: PIPELINE HEALTH KPIs ===
    st.markdown("#### Pipeline Health Summary")
    p1, p2, p3, p4 = st.columns(4)
    
    with p1:
        st.metric(
            "Total Backlog",
            format_currency(tot_backlog, moeda_sel),
            f"{backlog_pct:.1f}% of Total Revenue",
            delta_color="off"
        )
    
    with p2:
        st.metric(
            "Total Pipeline",
            format_currency(tot_pipeline, moeda_sel),
            f"Unfunded Opportunity",
            delta_color="normal" if tot_pipeline > 0 else "off"
        )
    
    with p3:
        coverage_color = "normal" if pipeline_coverage >= 1.0 else "inverse"
        st.metric(
            "Pipeline Coverage Ratio",
            f"{pipeline_coverage:.2f}x",
            "Pipeline / Backlog",
            delta_color=coverage_color
        )
    
    with p4:
        months_label = f"{backlog_months:.1f} months" if backlog_months > 0 else "Depleted"
        st.metric(
            "Backlog Runway",
            months_label,
            "at current pace",
            delta_color="inverse" if backlog_months < 3 else "normal"
        )
    
    st.markdown("---")
    
    # === ROW 2: MONTHLY REVENUE WATERFALL WITH OPACITY ===
    st.markdown("#### 12-Month Revenue Waterfall (Stacked with Fade)")
    
    monthly_rev = []
    for period in ordem_meses:
        act = df_act_rev[df_act_rev['Fiscal Period'] == period]['Amount Final'].sum() if not df_act_rev.empty else 0
        back = df_backlog[df_backlog['Fiscal Period'] == period][amount_col].sum() if not df_backlog.empty else 0
        pipe = df_pipeline[df_pipeline['Fiscal Period'] == period][amount_col].sum() if not df_pipeline.empty else 0
        monthly_rev.append({
            'Period': period,
            'Actuals': act,
            'Backlog': back,
            'Pipeline': pipe
        })
    
    monthly_df = pd.DataFrame(monthly_rev)
    
    fig_waterfall = go.Figure()
    
    # Actuals: solid opacity
    fig_waterfall.add_trace(go.Bar(
        x=monthly_df['Period'], y=monthly_df['Actuals'], name='Actuals',
        marker_color=COLOUR_PRIMARY, opacity=1.0,
        hovertemplate='<b>Actuals</b><br>%{y:,.0f}<extra></extra>'
    ))
    
    # Backlog: 0.8 opacity
    fig_waterfall.add_trace(go.Bar(
        x=monthly_df['Period'], y=monthly_df['Backlog'], name='Backlog',
        marker_color=COLOUR_SOFT_BLUE, opacity=0.8,
        hovertemplate='<b>Backlog</b><br>%{y:,.0f}<extra></extra>'
    ))
    
    # Pipeline: 0.6 opacity
    fig_waterfall.add_trace(go.Bar(
        x=monthly_df['Period'], y=monthly_df['Pipeline'], name='Pipeline',
        marker_color=COLOUR_GREEN, opacity=0.6,
        hovertemplate='<b>Pipeline</b><br>%{y:,.0f}<extra></extra>'
    ))
    
    fig_waterfall.update_layout(
        barmode='stack',
        hovermode='x unified',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(showgrid=False, zeroline=False),
        yaxis=dict(showgrid=False, zeroline=False),
        legend=dict(orientation='h', yanchor='bottom', y=1.01, xanchor='left', x=0),
        margin=dict(t=30, b=30, l=60, r=50),
        height=350
    )
    st.plotly_chart(fig_waterfall, use_container_width=True)
    
    st.markdown("---")
    
    # === ROW 3: REVENUE MIX & AT-RISK ACCOUNTS ===
    col_mix, col_risk = st.columns([1, 1])
    
    with col_mix:
        st.markdown("#### Revenue Mix by Practice (LOB)")
        
        if 'Profit Center' in df_forecast.columns or 'Practice' in df_forecast.columns:
            practice_col = 'Profit Center' if 'Profit Center' in df_forecast.columns else 'Practice'
            combined_rev = pd.concat([
                df_act_rev.groupby(practice_col)['Amount Final'].sum() if not df_act_rev.empty else pd.Series(),
                df_backlog.groupby(practice_col)[amount_col].sum() if not df_backlog.empty else pd.Series(),
                df_pipeline.groupby(practice_col)[amount_col].sum() if not df_pipeline.empty else pd.Series()
            ]).groupby(level=0).sum().reset_index()
            combined_rev.columns = [practice_col, 'Total Revenue']
            combined_rev = combined_rev[combined_rev['Total Revenue'] > 0].sort_values('Total Revenue', ascending=False)
            
            if not combined_rev.empty:
                fig_mix = px.pie(
                    combined_rev, values='Total Revenue', names=practice_col,
                    color_discrete_sequence=px.colors.qualitative.Plotly,
                    title=None
                )
                fig_mix.update_traces(
                    textposition='inside', textinfo='percent+label',
                    hovertemplate='<b>%{label}</b><br>Revenue: %{value:,.0f}<extra></extra>'
                )
                fig_mix.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    margin=dict(t=0, b=0, l=0, r=0),
                    height=350,
                    showlegend=False
                )
                st.plotly_chart(fig_mix, use_container_width=True)
            else:
                st.info("No LOB breakdown available in forecast data.")
        else:
            st.warning("Practice/LOB dimension not available in Forecast data.")
    
    with col_risk:
        st.markdown("#### At-Risk Accounts (Low Coverage)")
        
        # Analyze account-level coverage
        if 'Account' in df_forecast.columns or 'Company' in df_forecast.columns:
            account_col = 'Account' if 'Account' in df_forecast.columns else 'Company'
            
            account_backlog = df_backlog.groupby(account_col)[amount_col].sum() if not df_backlog.empty else pd.Series()
            account_pipeline = df_pipeline.groupby(account_col)[amount_col].sum() if not df_pipeline.empty else pd.Series()
            
            account_coverage = pd.DataFrame({
                'Account': account_backlog.index,
                'Backlog': account_backlog.values,
                'Pipeline': account_pipeline.reindex(account_backlog.index, fill_value=0).values
            })
            
            account_coverage['Coverage Ratio'] = (account_coverage['Pipeline'] / account_coverage['Backlog']).fillna(0)
            account_coverage = account_coverage[account_coverage['Backlog'] > 0].sort_values('Coverage Ratio')
            
            if not account_coverage.empty:
                # Highlight top 10 at-risk (lowest coverage)
                at_risk = account_coverage.head(10).copy()
                at_risk['Risk Level'] = at_risk['Coverage Ratio'].apply(
                    lambda x: '🔴 Critical' if x < 0.5 else ('🟡 Warning' if x < 1.0 else '🟢 Healthy')
                )
                
                fig_risk = go.Figure(data=[
                    go.Bar(
                        y=at_risk['Account'],
                        x=at_risk['Backlog'],
                        name='Backlog at Risk',
                        marker_color=COLOUR_RED,
                        orientation='h',
                        hovertemplate='<b>%{y}</b><br>Backlog: %{x:,.0f}<extra></extra>'
                    ),
                    go.Bar(
                        y=at_risk['Account'],
                        x=at_risk['Pipeline'],
                        name='Pipeline Coverage',
                        marker_color=COLOUR_GREEN,
                        orientation='h',
                        hovertemplate='<b>%{y}</b><br>Pipeline: %{x:,.0f}<extra></extra>'
                    )
                ])
                
                fig_risk.update_layout(
                    barmode='stack',
                    hovermode='y unified',
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    xaxis=dict(showgrid=False, zeroline=False),
                    yaxis=dict(showgrid=False, zeroline=False),
                    legend=dict(orientation='v', yanchor='top', y=0.99, xanchor='right', x=0.99),
                    margin=dict(t=0, b=30, l=200, r=50),
                    height=350
                )
                st.plotly_chart(fig_risk, use_container_width=True)
            else:
                st.info("No account-level pipeline data available.")
        else:
            st.warning("Account dimension not available in Forecast data.")
    
    st.markdown("---")
    
    # === ROW 4: SUMMARY STATISTICS TABLE ===
    st.markdown("#### Pipeline Snapshot by Fiscal Period")
    
    pipeline_summary = pd.DataFrame({
        'Fiscal Period': ordem_meses,
        'Actuals': [
            df_act_rev[df_act_rev['Fiscal Period'] == p]['Amount Final'].sum() if not df_act_rev.empty else 0
            for p in ordem_meses
        ],
        'Backlog': [
            df_backlog[df_backlog['Fiscal Period'] == p][amount_col].sum() if not df_backlog.empty else 0
            for p in ordem_meses
        ],
        'Pipeline': [
            df_pipeline[df_pipeline['Fiscal Period'] == p][amount_col].sum() if not df_pipeline.empty else 0
            for p in ordem_meses
        ]
    })
    
    pipeline_summary['Total Forward Revenue'] = (pipeline_summary['Actuals'] + pipeline_summary['Backlog'] + pipeline_summary['Pipeline'])
    pipeline_summary['Backlog %'] = (pipeline_summary['Backlog'] / pipeline_summary['Total Forward Revenue'] * 100).round(1)
    pipeline_summary['Pipeline %'] = (pipeline_summary['Pipeline'] / pipeline_summary['Total Forward Revenue'] * 100).round(1)
    
    st.dataframe(pipeline_summary, use_container_width=True, hide_index=True, column_config={
        'Actuals': st.column_config.NumberColumn(format='%d'),
        'Backlog': st.column_config.NumberColumn(format='%d'),
        'Pipeline': st.column_config.NumberColumn(format='%d'),
        'Total Forward Revenue': st.column_config.NumberColumn(format='%d'),
        'Backlog %': st.column_config.NumberColumn(format='%.1f%%'),
        'Pipeline %': st.column_config.NumberColumn(format='%.1f%%')
    })
