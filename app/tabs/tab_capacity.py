import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from utils import format_currency, ordem_meses

def render_capacity(df_fc_cl, df_cap_cl, df_base_filtered, moeda_sel):
    st.markdown("### Executive Summary: Supply vs Demand")
    
    df_act_rev = pd.DataFrame()
    if not df_base_filtered.empty and 'WBS element' in df_base_filtered.columns:
        df_act_rev = df_base_filtered[(df_base_filtered['P&L LVL 5'] == 'Net Sales') & (df_base_filtered['WBS element'].notna()) & (df_base_filtered['WBS element'] != '')].copy()
        df_act_rev['Amount Final'] *= -1

    tot_act = df_act_rev['Amount Final'].sum() if not df_act_rev.empty else 0
    tot_bck = df_fc_cl[df_fc_cl['Backlog/Pipeline'] == 'Backlog']['Amount Final'].sum() if not df_fc_cl.empty else 0
    tot_pip = df_fc_cl[df_fc_cl['Backlog/Pipeline'] == 'Pipeline']['Amount Final'].sum() if not df_fc_cl.empty else 0
    tot_cap = df_cap_cl['Amount Final'].sum() if not df_cap_cl.empty else 0
    
    secured = tot_act + tot_bck
    bench_cost = tot_cap - secured
    
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("1. Target Capacity (Supply)", format_currency(tot_cap, moeda_sel))
    m2.metric("2. Secured Revenue (Actual+Backlog)", format_currency(secured, moeda_sel))
    
    if bench_cost > 0: m3.metric("3. Unused Capacity (Bench Risk)", format_currency(bench_cost, moeda_sel), "- Idle Cost", delta_color="inverse")
    else: m3.metric("3. Overbooked (Delivery Risk)", format_currency(abs(bench_cost), moeda_sel), "+ Need Subcontractors", delta_color="off")
        
    m4.metric("4. Potential Pipeline", format_currency(tot_pip, moeda_sel))
    
    st.markdown("---")
    col_chart1, col_chart2 = st.columns([2, 1])
    
    with col_chart1:
        st.markdown("#### Delivery Evolution vs Capacity Ceiling")
        df_chart = pd.DataFrame({'Fiscal Period': ordem_meses})
        
        def get_agg(df, col_filter, val_filter):
            if df.empty or col_filter not in df.columns: return pd.Series(0, index=ordem_meses)
            return df[df[col_filter] == val_filter].groupby('Fiscal Period')['Amount Final'].sum().reindex(ordem_meses, fill_value=0)

        df_chart['Actuals'] = df_act_rev.groupby('Fiscal Period')['Amount Final'].sum().reindex(ordem_meses, fill_value=0).values if not df_act_rev.empty else 0
        df_chart['Backlog'] = get_agg(df_fc_cl, 'Backlog/Pipeline', 'Backlog').values
        df_chart['Pipeline'] = get_agg(df_fc_cl, 'Backlog/Pipeline', 'Pipeline').values
        df_chart['Capacity'] = df_cap_cl.groupby('Fiscal Period')['Amount Final'].sum().reindex(ordem_meses, fill_value=0).values if not df_cap_cl.empty else 0

        fig_trend = go.Figure()
        fig_trend.add_trace(go.Bar(x=df_chart['Fiscal Period'], y=df_chart['Actuals'], name='Actuals', marker_color='#2c3e50'))
        fig_trend.add_trace(go.Bar(x=df_chart['Fiscal Period'], y=df_chart['Backlog'], name='Backlog', marker_color='#005A9C'))
        fig_trend.add_trace(go.Bar(x=df_chart['Fiscal Period'], y=df_chart['Pipeline'], name='Pipeline', marker_color='#00cc96'))
        fig_trend.add_trace(go.Scatter(x=df_chart['Fiscal Period'], y=df_chart['Capacity'], name='Max Capacity', mode='lines+markers', line=dict(color='#d32f2f', width=3, dash='dash')))
        fig_trend.update_layout(barmode='stack', hovermode='x unified', plot_bgcolor='white', legend=dict(orientation="h", y=1.1))
        st.plotly_chart(fig_trend, use_container_width=True)

    with col_chart2:
        st.markdown("#### Bench by Position")
        if not df_cap_cl.empty and 'Position Title' in df_cap_cl.columns:
            cap_by_pos = df_cap_cl.groupby('Position Title')['Amount Final'].sum().reset_index()
            cap_by_pos = cap_by_pos[cap_by_pos['Amount Final'] > 0].sort_values(by='Amount Final', ascending=False)
            fig_pos = px.pie(cap_by_pos, values='Amount Final', names='Position Title', hole=0.5, color_discrete_sequence=px.colors.sequential.Blues_r)
            fig_pos.update_traces(textposition='inside', textinfo='percent')
            fig_pos.update_layout(showlegend=False, margin=dict(t=0, b=0, l=0, r=0))
            st.plotly_chart(fig_pos, use_container_width=True)