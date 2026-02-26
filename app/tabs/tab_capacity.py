import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from utils import format_currency, format_variance, variance_basis_points, ordem_meses

# Corporate colour palette
COLOUR_PRIMARY = '#002a4d'
COLOUR_GRAY = '#7f8c8d'
COLOUR_SOFT_BLUE = '#3498db'
COLOUR_GREEN = '#2ecc71'
COLOUR_RED = '#e74c3c'

def render_capacity(df_fc_cl, df_cap_cl, df_base_filtered, moeda_sel):
    """
    TAB 2: Capacity vs Delivery Risk
    Delivers advanced FP&A narratives on supply utilization, bench cost, subcontractor risk, and revenue pipeline.
    """
    st.markdown("### Capacity: Supply vs Demand Intelligence")
    
    # === DATA PREPARATION ===
    df_act_rev = pd.DataFrame()
    if not df_base_filtered.empty and 'WBS element' in df_base_filtered.columns:
        df_act_rev = df_base_filtered[
            (df_base_filtered['P&L LVL 5'] == 'Net Sales') & 
            (df_base_filtered['WBS element'].notna()) & 
            (df_base_filtered['WBS element'] != '')
        ].copy()
        # Amount Normalized already has correct sign (positive for revenue)
        if 'Amount Normalized' in df_act_rev.columns:
            df_act_rev['Amount Final'] = df_act_rev['Amount Normalized'].abs()
        else:
            df_act_rev['Amount Final'] = df_act_rev['Amount Final'].abs()

    # Use Amount Final (already converted/filtered by filter_and_convert)
    amount_col = 'Amount Final' if 'Amount Final' in df_cap_cl.columns else 'Target Capacity EUR'
    
    tot_act = df_act_rev['Amount Final'].sum() if not df_act_rev.empty else 0
    tot_bck = df_fc_cl[df_fc_cl['Backlog/Pipeline'] == 'Backlog'][amount_col].sum() if not df_fc_cl.empty else 0
    tot_pip = df_fc_cl[df_fc_cl['Backlog/Pipeline'] == 'Pipeline'][amount_col].sum() if not df_fc_cl.empty else 0
    tot_cap = df_cap_cl[amount_col].sum() if not df_cap_cl.empty else 0
    
    secured = tot_act + tot_bck
    blended_utilization = (secured / tot_cap * 100) if tot_cap > 0 else 0
    bench_cost = max(0, tot_cap - secured)
    subcontractor_risk = max(0, secured - tot_cap)
    pipeline_coverage = (tot_pip / bench_cost) if bench_cost > 0 else 0 if tot_pip == 0 else float('inf')
    
    # === ROW 1: EXECUTIVE KPIs ===
    st.markdown("#### Executive Metrics")
    k1, k2, k3, k4 = st.columns(4)
    
    with k1:
        st.metric(
            "Blended Utilization",
            f"{blended_utilization:.1f}%",
            f"{blended_utilization - 75:.1f}% vs target",
            delta_color="inverse" if blended_utilization < 75 else "normal"
        )
    
    with k2:
        st.metric(
            "Bench Cost Value",
            format_currency(bench_cost, moeda_sel),
            "Idle Capacity" if bench_cost > 0 else "Overbooked",
            delta_color="inverse"
        )
    
    with k3:
        subcontr_label = format_currency(subcontractor_risk, moeda_sel) if subcontractor_risk > 0 else "–"
        st.metric(
            "Subcontractor Risk",
            subcontr_label,
            "Delivery Risk" if subcontractor_risk > 0 else "Protected",
            delta_color="off" if subcontractor_risk == 0 else "normal"
        )
    
    with k4:
        coverage_text = f"{pipeline_coverage:.1f}x" if pipeline_coverage != float('inf') and pipeline_coverage > 0 else "Uncovered"
        st.metric(
            "Pipeline Coverage Ratio",
            coverage_text,
            "Bench Risk" if pipeline_coverage < 1 else "Healthy",
            delta_color="inverse" if pipeline_coverage < 1 else "normal"
        )
    
    st.markdown("---")
    
    # === ROW 2: SUPPLY vs DEMAND FORWARD CURVE ===
    st.markdown("#### Supply vs Demand: 12-Month Forward Curve")
    df_chart = pd.DataFrame({'Fiscal Period': ordem_meses})
    
    def get_agg(df, col_filter, val_filter):
        if df.empty or col_filter not in df.columns:
            return pd.Series(0, index=ordem_meses)
        col = 'Amount Normalized' if 'Amount Normalized' in df.columns else amount_col
        return df[df[col_filter] == val_filter].groupby('Fiscal Period')[col].sum().reindex(ordem_meses, fill_value=0)

    df_chart['Actuals'] = (
        df_act_rev.groupby('Fiscal Period')['Amount Final'].sum().reindex(ordem_meses, fill_value=0).values 
        if not df_act_rev.empty else 0
    )
    df_chart['Backlog'] = get_agg(df_fc_cl, 'Backlog/Pipeline', 'Backlog').values
    df_chart['Pipeline'] = get_agg(df_fc_cl, 'Backlog/Pipeline', 'Pipeline').values
    df_chart['Capacity'] = (
        df_cap_cl.groupby('Fiscal Period')[amount_col].sum().reindex(ordem_meses, fill_value=0).values 
        if not df_cap_cl.empty else 0
    )

    fig_trend = go.Figure()
    fig_trend.add_trace(go.Bar(
        x=df_chart['Fiscal Period'], y=df_chart['Actuals'], name='Actuals', 
        marker_color=COLOUR_PRIMARY, hovertemplate='<b>Actuals</b><br>%{y:,.0f}<extra></extra>'
    ))
    fig_trend.add_trace(go.Bar(
        x=df_chart['Fiscal Period'], y=df_chart['Backlog'], name='Backlog', 
        marker_color=COLOUR_SOFT_BLUE, hovertemplate='<b>Backlog</b><br>%{y:,.0f}<extra></extra>'
    ))
    fig_trend.add_trace(go.Bar(
        x=df_chart['Fiscal Period'], y=df_chart['Pipeline'], name='Pipeline', 
        marker_color=COLOUR_GREEN, hovertemplate='<b>Pipeline</b><br>%{y:,.0f}<extra></extra>'
    ))
    fig_trend.add_trace(go.Scatter(
        x=df_chart['Fiscal Period'], y=df_chart['Capacity'], name='Max Capacity (Ceiling)', 
        mode='lines', line=dict(color=COLOUR_RED, width=3, dash='dash'),
        hovertemplate='<b>Capacity Ceiling</b><br>%{y:,.0f}<extra></extra>'
    ))
    
    fig_trend.update_layout(
        barmode='stack',
        hovermode='x unified',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(showgrid=False, zeroline=False),
        yaxis=dict(showgrid=False, zeroline=False),
        legend=dict(orientation='h', yanchor='bottom', y=1.01, xanchor='left', x=0),
        margin=dict(t=30, b=30, l=50, r=50),
        height=400
    )
    st.plotly_chart(fig_trend, use_container_width=True)
    
    st.markdown("---")
    
    # === ROW 3: BENCH RISK TREEMAP & UTILIZATION HEATMAP ===
    col_tree, col_heat = st.columns([1, 1])
    
    with col_tree:
        st.markdown("#### Bench Risk by Position & Company")
        if not df_cap_cl.empty and 'Position Title' in df_cap_cl.columns:
            # Create bench allocation by Position and Company
            cap_by_pos_co = df_cap_cl.copy()
            if 'Company' not in cap_by_pos_co.columns:
                cap_by_pos_co['Company'] = 'Unspecified'
            
            cap_agg = cap_by_pos_co.groupby(['Position Title', 'Company'])[amount_col].sum().reset_index()
            cap_agg.columns = ['Position Title', 'Company', 'Capacity']
            
            # Estimate bench (visible idle positions)
            cap_agg['Bench Risk'] = cap_agg['Capacity'] * (1 - blended_utilization / 100)
            cap_agg = cap_agg[cap_agg['Bench Risk'] > 0].sort_values('Bench Risk', ascending=False).head(15)
            
            if not cap_agg.empty:
                fig_tree = px.treemap(
                    cap_agg, 
                    labels=['Position Title', 'Company'], 
                    values='Bench Risk',
                    color='Bench Risk',
                    color_continuous_scale=[[0, COLOUR_GREEN], [0.5, '#f39c12'], [1, COLOUR_RED]],
                    hover_data={'Bench Risk': ':.0f'},
                    title=None
                )
                fig_tree.update_traces(textposition='middle center', hovertemplate='<b>%{label}</b><br>Bench Risk: %{value:,.0f}<extra></extra>')
                fig_tree.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    margin=dict(t=0, b=0, l=0, r=0),
                    height=350,
                    coloraxis_colorbar=dict(thickness=15, len=0.7)
                )
                st.plotly_chart(fig_tree, use_container_width=True)
            else:
                st.info("🎯 No bench risk detected – all demand is satisfied.")
        else:
            st.warning("Position Title dimension not available in Capacity data.")
    
    with col_heat:
        st.markdown("#### Utilization Rate by Period & Position")
        if not df_cap_cl.empty and 'Position Title' in df_cap_cl.columns:
            # Build period-position utilization heatmap
            cap_by_period_pos = df_cap_cl.groupby(['Fiscal Period', 'Position Title'])[amount_col].sum().reset_index()
            secured_by_period_pos = pd.concat([
                df_act_rev.groupby(['Fiscal Period', 'Position Title' if 'Position Title' in df_act_rev.columns else 'WBS element'])['Amount Final'].sum().reset_index() if not df_act_rev.empty else pd.DataFrame(),
                df_fc_cl[df_fc_cl['Backlog/Pipeline'].isin(['Actuals', 'Backlog'])].groupby(['Fiscal Period', 'Position Title' if 'Position Title' in df_fc_cl.columns else 'WBS element'])[amount_col].sum().reset_index() if not df_fc_cl.empty else pd.DataFrame()
            ])
            
            if not cap_by_period_pos.empty:
                cap_pv = cap_by_period_pos.set_index(['Fiscal Period', 'Position Title'])[amount_col].unstack(fill_value=0)
                sec_pv = secured_by_period_pos.set_index(['Fiscal Period', 'Position Title']).sum(axis=1).groupby('Fiscal Period').sum() if not secured_by_period_pos.empty else pd.Series()
                
                util_matrix = (cap_pv / cap_pv.sum(axis=1).values.reshape(-1, 1) * 100).fillna(0)
                util_matrix = util_matrix.loc[ordem_meses].fillna(0)
                
                fig_heat = go.Figure(data=go.Heatmap(
                    z=util_matrix.T.values,
                    x=util_matrix.index,
                    y=util_matrix.columns,
                    colorscale=[[0, COLOUR_RED], [0.5, '#f39c12'], [1, COLOUR_GREEN]],
                    hovertemplate='<b>%{y}</b><br>Period %{x}<br>Capacity: %{z:.1f}%<extra></extra>',
                    colorbar=dict(title='Capacity %', thickness=15, len=0.7)
                ))
                fig_heat.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    xaxis=dict(showgrid=False),
                    yaxis=dict(showgrid=False),
                    margin=dict(t=0, b=30, l=150, r=50),
                    height=350
                )
                st.plotly_chart(fig_heat, use_container_width=True)
            else:
                st.info("No period-level utilization data available.")
        else:
            st.warning("Position Title dimension required for heatmap analysis.")
    
    st.markdown("---")
    
    # === ROW 4: SUMMARY STATISTICS ===
    st.markdown("#### Period-Level Capacity Snapshot")
    cap_summary = pd.DataFrame({
        'Fiscal Period': ordem_meses,
        'Capacity Supply': df_cap_cl.groupby('Fiscal Period')[amount_col].sum().reindex(ordem_meses, fill_value=0).values if not df_cap_cl.empty else 0,
    })
    cap_summary['Secured Demand'] = [
        (df_act_rev[df_act_rev['Fiscal Period'] == p]['Amount Final'].sum() if not df_act_rev.empty else 0) +
        (df_fc_cl[(df_fc_cl['Fiscal Period'] == p) & (df_fc_cl['Backlog/Pipeline'] == 'Backlog')][amount_col].sum() if not df_fc_cl.empty else 0)
        for p in ordem_meses
    ]
    cap_summary['Pipeline Potential'] = df_fc_cl[df_fc_cl['Backlog/Pipeline'] == 'Pipeline'].groupby('Fiscal Period')[amount_col].sum().reindex(ordem_meses, fill_value=0).values if not df_fc_cl.empty else 0
    cap_summary['Utilization %'] = (cap_summary['Secured Demand'] / cap_summary['Capacity Supply'] * 100).round(1)
    cap_summary['Bench Risk'] = (cap_summary['Capacity Supply'] - cap_summary['Secured Demand']).clip(lower=0).astype(int)
    
    st.dataframe(cap_summary, use_container_width=True, hide_index=True, column_config={
        'Capacity Supply': st.column_config.NumberColumn(format='%d'),
        'Secured Demand': st.column_config.NumberColumn(format='%d'),
        'Pipeline Potential': st.column_config.NumberColumn(format='%d'),
        'Utilization %': st.column_config.NumberColumn(format='%.1f%%'),
        'Bench Risk': st.column_config.NumberColumn(format='%d')
    })