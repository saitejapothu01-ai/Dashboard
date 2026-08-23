import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')

# ── PAGE CONFIG ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Renewable Energy & CO₂ Dashboard",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CUSTOM CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2rem; font-weight: 700; color: #185FA5;
        border-bottom: 3px solid #185FA5; padding-bottom: 0.5rem; margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.1rem; color: #444; margin-bottom: 1.5rem;
    }
    .metric-card {
        background: #f8f9fa; border-radius: 10px; padding: 1rem;
        border-left: 4px solid #185FA5; margin-bottom: 1rem;
    }
    .finding-box {
        background: #E6F1FB; border-radius: 8px; padding: 1rem;
        border-left: 4px solid #185FA5; margin: 0.5rem 0;
    }
    .warning-box {
        background: #FAEEDA; border-radius: 8px; padding: 1rem;
        border-left: 4px solid #BA7517; margin: 0.5rem 0;
    }
    .success-box {
        background: #EAF3DE; border-radius: 8px; padding: 1rem;
        border-left: 4px solid #27500A; margin: 0.5rem 0;
    }
    .null-box {
        background: #F1EFE8; border-radius: 8px; padding: 1rem;
        border-left: 4px solid #993C1D; margin: 0.5rem 0;
    }
    .stSelectbox label { font-weight: 600; }
    .stMultiSelect label { font-weight: 600; }
</style>
""", unsafe_allow_html=True)

# ── INCOME COLOURS ───────────────────────────────────────────────────────────
INCOME_COLORS = {
    'High income':         '#185FA5',
    'Upper middle income': '#533AB7',
    'Lower middle income': '#BA7517',
    'Low income':          '#993C1D'
}
INCOME_ORDER = ['High income','Upper middle income','Lower middle income','Low income']

# ── LOAD DATA ────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv('clean_dataset_for_dashboard.csv')
    df['log_co2'] = np.log(df['co2_emissions'] + 1)
    df['log_gdp'] = np.log(df['gdp_per_capita'] + 1)
    df['renew_x_gdp']    = df['renewable_share'] * df['log_gdp']
    df['renew_x_growth'] = df['renewable_share'] * df['gdp_growth']
    df['log_gdp_sq']     = df['log_gdp'] ** 2
    return df

df_all = load_data()
df_model = df_all[df_all['Year'] < 2020].copy()

# ── SIDEBAR ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/wind-turbine.png", width=60)
    st.markdown("## 🌱 Dashboard Controls")
    st.markdown("---")

    page = st.radio("Navigate to", [
        "🏠 Overview",
        "📊 Exploratory Analysis",
        "📈 Regression Results",
        "🌍 EKC Analysis",
        "💡 Income Group Diagnostic",
        "🔍 Country Explorer"
    ])

    st.markdown("---")
    st.markdown("### Filters")

    include_2020 = st.checkbox("Include 2020 in visualisations", value=False,
                                help="2020 excluded from modelling due to COVID-19 distortions")

    selected_groups = st.multiselect(
        "Income Groups",
        options=INCOME_ORDER,
        default=INCOME_ORDER
    )

    year_range = st.slider(
        "Year Range",
        min_value=2000, max_value=2020,
        value=(2000, 2019)
    )

    st.markdown("---")
    st.markdown("**Dissertation:**")
    st.markdown("*The Moderating Role of Economic Development in Renewable Energy & CO₂ Emissions*")
    st.markdown("**Dataset:** 3,072 obs · 159 countries · 2000–2019")

# ── FILTER DATA ──────────────────────────────────────────────────────────────
max_year = 2020 if include_2020 else 2019
df = df_all[
    (df_all['Year'] >= year_range[0]) &
    (df_all['Year'] <= min(year_range[1], max_year)) &
    (df_all['income_group'].isin(selected_groups))
].copy()

# ═══════════════════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════
if page == "🏠 Overview":
    st.markdown('<div class="main-header">🌱 Renewable Energy & CO₂ Emissions</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">The Moderating Role of Economic Development — Global Panel Analysis 2000–2019</div>', unsafe_allow_html=True)

    # KPI metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Countries Analysed", f"{df['Entity'].nunique():,}", delta=None)
    with col2:
        st.metric("Observations", f"{len(df_model):,}", delta=None)
    with col3:
        avg_ren = df_model['renewable_share'].mean()
        st.metric("Avg Renewable Share", f"{avg_ren:.1f}%", delta=None)
    with col4:
        avg_co2 = df_model['co2_emissions'].mean()/1000
        st.metric("Avg CO₂ (thousand kt)", f"{avg_co2:,.0f}", delta=None)

    st.markdown("---")

    # Key findings summary
    st.markdown("### 🔑 Key Research Findings")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class="success-box">
        <b>Direct Relationship Confirmed</b><br>
        Renewable energy share significantly reduces CO₂ emissions globally.<br>
        <b>β = −0.0289 (p&lt;0.001)</b> — every 1% increase in renewable share reduces CO₂ by ~2.89%
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="success-box">
        <b>GDP per Capita Moderates the Effect</b><br>
        The emissions-reducing effect of renewables weakens as countries get wealthier.<br>
        <b>Interaction β = +0.0018 (p&lt;0.001)</b> — statistically significant moderation
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="null-box">
        <b>GDP Growth Does NOT Moderate</b><br>
        Annual GDP growth rate does not significantly change the renewable–emissions relationship.<br>
        <b>Interaction β = −0.0000 (p = 0.5677)</b> — not significant
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="finding-box">
        <b>📈 EKC Confirmed — Turning Point at $50,490</b><br>
        Inverted U-shape confirmed. Countries above $50,490 GDP/capita see falling emissions.<br>
        <b>log_gdp² β = −0.0271 (p&lt;0.001)</b>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🌍 Income Group — Diagnostic Finding")
    st.markdown("""
    <div class="warning-box">
    <b>⭐ Surprising Finding: Low-income countries benefit MOST from renewable energy</b><br>
    Each 1% increase in renewable share reduces CO₂ by <b>4.65%</b> in low-income countries vs only <b>2.62%</b> in high-income countries.
    This challenges the assumption in existing literature that wealthier nations benefit more.
    </div>
    """, unsafe_allow_html=True)

    # Quick chart
    st.markdown("### Global Trends at a Glance")
    col1, col2 = st.columns(2)
    with col1:
        co2_t = df_model.groupby('Year')['co2_emissions'].mean().reset_index()
        fig = px.line(co2_t, x='Year', y='co2_emissions',
                      title='Average CO₂ Emissions Per Country (kt)',
                      color_discrete_sequence=['#E24B4A'])
        fig.update_layout(showlegend=False, height=300, margin=dict(t=40,b=20))
        fig.update_yaxes(title='CO₂ Emissions (kt)')
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        ren_t = df_model.groupby('Year')['renewable_share'].mean().reset_index()
        fig = px.line(ren_t, x='Year', y='renewable_share',
                      title='Average Renewable Energy Share (%)',
                      color_discrete_sequence=['#0F6E56'])
        fig.update_layout(showlegend=False, height=300, margin=dict(t=40,b=20))
        fig.update_yaxes(title='Renewable Share (%)')
        st.plotly_chart(fig, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════
# PAGE 2 — EXPLORATORY ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════
elif page == "📊 Exploratory Analysis":
    st.markdown('<div class="main-header">📊 Exploratory Data Analysis</div>', unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs(["Trends Over Time", "Income Group Comparison", "Scatter Analysis", "Descriptive Statistics"])

    with tab1:
        st.markdown("### CO₂ Emissions and Renewable Share Trends by Income Group")
        col1, col2 = st.columns(2)

        with col1:
            fig_data = []
            for grp in selected_groups:
                sub = df[df['income_group']==grp].groupby('Year')['co2_emissions'].mean().reset_index()
                fig_data.append(go.Scatter(x=sub['Year'], y=sub['co2_emissions']/1000,
                                           mode='lines+markers', name=grp,
                                           line=dict(color=INCOME_COLORS.get(grp,'grey'), width=2),
                                           marker=dict(size=5)))
            fig = go.Figure(fig_data)
            fig.update_layout(title='CO₂ Emissions by Income Group', height=380,
                              yaxis_title='Avg CO₂ (thousand kt)', xaxis_title='Year',
                              legend=dict(orientation='h', yanchor='bottom', y=-0.3))
            if not include_2020:
                fig.add_vline(x=2019.5, line_dash='dash', line_color='grey',
                              annotation_text='2020 excluded')
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig_data = []
            for grp in selected_groups:
                sub = df[df['income_group']==grp].groupby('Year')['renewable_share'].mean().reset_index()
                fig_data.append(go.Scatter(x=sub['Year'], y=sub['renewable_share'],
                                           mode='lines+markers', name=grp,
                                           line=dict(color=INCOME_COLORS.get(grp,'grey'), width=2),
                                           marker=dict(size=5)))
            fig = go.Figure(fig_data)
            fig.update_layout(title='Renewable Share by Income Group', height=380,
                              yaxis_title='Renewable Share (%)', xaxis_title='Year',
                              legend=dict(orientation='h', yanchor='bottom', y=-0.3))
            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.markdown("### Income Group Averages")
        grp_stats = df[df['Year']<2020].groupby('income_group').agg(
            avg_co2=('co2_emissions','mean'),
            avg_renewable=('renewable_share','mean'),
            avg_gdp=('gdp_per_capita','mean'),
            avg_growth=('gdp_growth','mean'),
            countries=('Entity','nunique')
        ).reset_index()
        grp_stats.columns = ['Income Group','Avg CO₂ (kt)','Avg Renewable (%)','Avg GDP/Capita','Avg Growth (%)','Countries']
        grp_stats = grp_stats[grp_stats['Income Group'].isin(selected_groups)]
        grp_stats[['Avg CO₂ (kt)','Avg GDP/Capita']] = grp_stats[['Avg CO₂ (kt)','Avg GDP/Capita']].round(0)
        grp_stats[['Avg Renewable (%)','Avg Growth (%)']] = grp_stats[['Avg Renewable (%)','Avg Growth (%)']].round(2)
        st.dataframe(grp_stats, use_container_width=True, hide_index=True)

        col1, col2 = st.columns(2)
        with col1:
            fig = px.bar(grp_stats, x='Income Group', y='Avg Renewable (%)',
                         color='Income Group', color_discrete_map=INCOME_COLORS,
                         title='Average Renewable Share by Income Group')
            fig.update_layout(showlegend=False, height=350)
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig = px.bar(grp_stats, x='Income Group', y='Avg CO₂ (kt)',
                         color='Income Group', color_discrete_map=INCOME_COLORS,
                         title='Average CO₂ Emissions by Income Group')
            fig.update_layout(showlegend=False, height=350)
            st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.markdown("### Bivariate Relationships")
        col1, col2 = st.columns(2)
        with col1:
            plot_df = df[df['income_group'].isin(selected_groups)].dropna(subset=['renewable_share','log_co2'])
            fig = px.scatter(plot_df, x='renewable_share', y='log_co2',
                             color='income_group', color_discrete_map=INCOME_COLORS,
                             category_orders={'income_group': INCOME_ORDER},
                             opacity=0.5, title='Renewable Share vs Log CO₂ by Income Group',
                             labels={'renewable_share':'Renewable Energy Share (%)','log_co2':'Log CO₂ Emissions','income_group':'Income Group'})
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            plot_df2 = df[df['income_group'].isin(selected_groups)].dropna(subset=['log_gdp','log_co2'])
            fig = px.scatter(plot_df2, x='log_gdp', y='log_co2',
                             color='income_group', color_discrete_map=INCOME_COLORS,
                             category_orders={'income_group': INCOME_ORDER},
                             opacity=0.5, title='Log GDP vs Log CO₂ by Income Group',
                             labels={'log_gdp':'Log GDP per Capita','log_co2':'Log CO₂ Emissions','income_group':'Income Group'})
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

    with tab4:
        st.markdown("### Descriptive Statistics (2000–2019)")
        desc = df[df['Year']<2020][['co2_emissions','renewable_share','gdp_per_capita',
                                     'gdp_growth','energy_intensity']].describe().T.round(2)
        desc.index = ['CO₂ Emissions (kt)','Renewable Share (%)','GDP per Capita',
                      'GDP Growth (%)','Energy Intensity']
        st.dataframe(desc, use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            fig = px.histogram(df[df['Year']<2020], x='renewable_share',
                               color='income_group', color_discrete_map=INCOME_COLORS,
                               nbins=40, title='Distribution of Renewable Energy Share',
                               labels={'renewable_share':'Renewable Share (%)','income_group':'Income Group'},
                               barmode='overlay', opacity=0.6)
            fig.update_layout(height=350)
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig = px.box(df[df['Year']<2020], x='income_group', y='renewable_share',
                         color='income_group', color_discrete_map=INCOME_COLORS,
                         category_orders={'income_group': INCOME_ORDER},
                         title='Renewable Share Distribution by Income Group',
                         labels={'income_group':'Income Group','renewable_share':'Renewable Share (%)'})
            fig.update_layout(showlegend=False, height=350)
            st.plotly_chart(fig, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════
# PAGE 3 — REGRESSION RESULTS
# ═══════════════════════════════════════════════════════════════════════════
elif page == "📈 Regression Results":
    st.markdown('<div class="main-header">📈 Regression Analysis Results</div>', unsafe_allow_html=True)
    st.info("All models use fixed effects panel regression with country and year dummies. HC3 robust standard errors. Dependent variable: Log CO₂ Emissions.")

    tab1, tab2, tab3 = st.tabs(["Model Summary Table", "Model 1 — Direct", "Model 2 — Moderation"])

    with tab1:
        st.markdown("### Table 4.3: Fixed Effects Regression Results — All Four Models")

        results_data = {
            'Variable': ['Renewable Share','Log GDP per Capita','Renew × Log GDP',
                         'GDP Growth Rate','Renew × GDP Growth','Log GDP² (EKC)',
                         'Energy Intensity','Fossil Electricity','Electricity Access',
                         'R²','Adj. R²','N'],
            'Model 1 — Direct': ['-0.0289***','0.1595***','—','0.0003','—','—',
                                   '0.0190***','0.0001***','0.0077***','0.9958','0.9955','3,072'],
            'Model 2 — GDP Mod': ['-0.0430***','0.1090***','0.0018***','0.0000','—','—',
                                    '0.0204***','0.0001***','0.0060***','0.9959','0.9956','3,072'],
            'Model 3 — Growth Mod': ['-0.0287***','0.1594***','—','0.0009','-0.0000','—',
                                       '0.0191***','0.0001***','0.0077***','0.9958','0.9955','3,072'],
            'Model 4 — EKC': ['-0.0272***','0.5870***','—','0.0001','—','-0.0271***',
                               '0.0247***','0.0001***','0.0061***','0.9959','0.9956','3,072'],
        }
        results_df = pd.DataFrame(results_data)
        st.dataframe(results_df, use_container_width=True, hide_index=True)
        st.caption("*** p<0.001, ** p<0.01, * p<0.05 | All models include country + year fixed effects")

        st.markdown("---")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown("""<div class="success-box"><b>Model 1</b><br>
            renewable_share: <b>-0.0289***</b><br>R² = 0.9958</div>""", unsafe_allow_html=True)
        with col2:
            st.markdown("""<div class="success-box"><b>Model 2</b><br>
            renew_x_gdp: <b>+0.0018***</b><br>R² = 0.9959</div>""", unsafe_allow_html=True)
        with col3:
            st.markdown("""<div class="null-box"><b>Model 3</b><br>
            renew_x_growth: <b>-0.0000 n.s.</b><br>p = 0.5677</div>""", unsafe_allow_html=True)
        with col4:
            st.markdown("""<div class="finding-box"><b>Model 4 EKC</b><br>
            log_gdp²: <b>-0.0271***</b><br>Turning pt: $50,490</div>""", unsafe_allow_html=True)

    with tab2:
        st.markdown("### Model 1: Direct Relationship Between Renewable Energy and CO₂")
        st.markdown("""
        <div class="success-box">
        <b>Result:</b> Renewable energy share has a significant negative effect on CO₂ emissions.<br>
        <b>β = −0.0289 (SE = 0.0012, t = −24.78, p&lt;0.001)</b><br>
        For every 1 percentage point increase in renewable energy share, CO₂ emissions decrease by approximately <b>2.89%</b>, 
        holding all other variables constant.
        </div>
        """, unsafe_allow_html=True)

        m1_data = {
            'Variable': ['const','renewable_share','log_gdp','gdp_growth','energy_intensity','fossil_electricity','electricity_access'],
            'Coefficient': [7.3654,-0.0289,0.1595,0.0003,0.0190,0.0001,0.0077],
            'Std Error': [0.1557,0.0012,0.0211,0.0009,0.0048,0.0000,0.0007],
            't-statistic': [47.303,-24.780,7.566,0.355,4.004,6.126,10.739],
            'p-value': [0.0000,0.0000,0.0000,0.7225,0.0001,0.0000,0.0000],
            'Significance': ['***','***','***','','***','***','***']
        }
        st.dataframe(pd.DataFrame(m1_data), use_container_width=True, hide_index=True)
        st.markdown("**R² = 0.9958 | Adj. R² = 0.9955 | N = 3,072 | F-stat = 18,758.01 (p<0.001)**")

        fig = px.bar(
            x=['renewable_share','log_gdp','energy_intensity','fossil_elec','electricity_access'],
            y=[-0.0289, 0.1595, 0.0190, 0.0001, 0.0077],
            color=[-0.0289, 0.1595, 0.0190, 0.0001, 0.0077],
            color_continuous_scale=['#E24B4A','#E24B4A','white','#0F6E56','#0F6E56'],
            title='Model 1 — Key Coefficients (excluding fixed effects)',
            labels={'x':'Variable','y':'Coefficient'}
        )
        fig.update_layout(showlegend=False, height=350, coloraxis_showscale=False)
        fig.add_hline(y=0, line_dash='dash', line_color='black', line_width=1)
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.markdown("### Model 2: Moderation by GDP per Capita")
        st.markdown("""
        <div class="success-box">
        <b>Result:</b> GDP per capita DOES significantly moderate the renewable–emissions relationship.<br>
        <b>Interaction β = +0.0018 (SE = 0.0003, t = 6.280, p&lt;0.001)</b><br>
        The positive interaction term means the emissions-reducing effect of renewables <b>weakens as countries get wealthier</b>.
        This is the primary finding of this dissertation.
        </div>
        """, unsafe_allow_html=True)

        m2_data = {
            'Variable': ['const','renewable_share','log_gdp','renew_x_gdp','gdp_growth','energy_intensity','fossil_electricity','electricity_access'],
            'Coefficient': [7.8516,-0.0430,0.1090,0.0018,0.0000,0.0204,0.0001,0.0060],
            'Std Error': [0.1712,0.0026,0.0219,0.0003,0.0009,0.0046,0.0000,0.0007],
            't-statistic': [45.850,-16.477,4.975,6.280,0.004,4.424,7.689,8.282],
            'p-value': [0.0000,0.0000,0.0000,0.0000,0.9964,0.0000,0.0000,0.0000],
            'Significance': ['***','***','***','***','','***','***','***']
        }
        st.dataframe(pd.DataFrame(m2_data), use_container_width=True, hide_index=True)
        st.markdown("**R² = 0.9959 | Adj. R² = 0.9956 | N = 3,072**")

        st.markdown("#### Marginal Effect of Renewable Energy at Different GDP Levels")
        gdp_vals = np.linspace(4.5, 12.5, 100)
        marginal_effect = -0.0430 + 0.0018 * gdp_vals
        gdp_actual = np.exp(gdp_vals)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=gdp_actual, y=marginal_effect,
                                  mode='lines', line=dict(color='#185FA5', width=2.5),
                                  name='Marginal Effect'))
        fig.add_hline(y=0, line_dash='dash', line_color='red',
                      annotation_text='Zero effect threshold')
        fig.add_vline(x=50490, line_dash='dot', line_color='#BA7517',
                      annotation_text='EKC Turning Point ($50,490)')
        fig.update_layout(
            title='Marginal Effect of Renewable Share on Log CO₂ by GDP per Capita',
            xaxis_title='GDP per Capita (USD, log scale)',
            yaxis_title='Marginal Effect on Log CO₂',
            xaxis_type='log', height=400
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption("The marginal effect becomes less negative as GDP per capita rises, showing the moderating role of economic development.")

# ═══════════════════════════════════════════════════════════════════════════
# PAGE 4 — EKC ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════
elif page == "🌍 EKC Analysis":
    st.markdown('<div class="main-header">🌍 Environmental Kuznets Curve Analysis</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="finding-box">
    <b>EKC Hypothesis:</b> As countries develop economically, emissions first rise (industrialisation phase) then fall 
    (post-industrial phase) — forming an inverted U-shape.<br><br>
    <b>Result:</b> EKC CONFIRMED — Turning point at GDP per capita ≈ <b>$50,490</b><br>
    log_gdp² coefficient = <b>−0.0271 (p&lt;0.001)</b>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([2,1])

    with col1:
        b1, b2 = 0.5870, -0.0271
        gdp_range = np.linspace(4.5, 12.5, 300)
        co2_ekc = 5.7924 + b1*gdp_range + b2*(gdp_range**2)
        tp_log = 10.830
        tp_gdp = np.exp(tp_log)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=gdp_range[gdp_range<=tp_log], y=co2_ekc[gdp_range<=tp_log],
            fill='tozeroy', fillcolor='rgba(226,75,74,0.1)',
            line=dict(color='#533AB7', width=3), name='Rising Phase', mode='lines'
        ))
        fig.add_trace(go.Scatter(
            x=gdp_range[gdp_range>=tp_log], y=co2_ekc[gdp_range>=tp_log],
            fill='tozeroy', fillcolor='rgba(15,110,86,0.1)',
            line=dict(color='#533AB7', width=3), name='Falling Phase', mode='lines',
            showlegend=True
        ))
        fig.add_vline(x=tp_log, line_dash='dash', line_color='#E24B4A', line_width=2,
                      annotation_text=f'Turning Point<br>log GDP = {tp_log}<br>≈ ${tp_gdp:,.0f}',
                      annotation_position='top left')

        for grp in INCOME_ORDER:
            sub = df_model[df_model['income_group']==grp]
            avg_log_gdp = sub['log_gdp'].mean()
            avg_log_co2 = sub['log_co2'].mean()
            fig.add_trace(go.Scatter(
                x=[avg_log_gdp], y=[avg_log_co2],
                mode='markers+text', name=grp,
                marker=dict(size=14, color=INCOME_COLORS[grp], symbol='diamond'),
                text=[grp.split(' ')[0]], textposition='top center',
                textfont=dict(size=9)
            ))

        fig.update_layout(
            title='Environmental Kuznets Curve — Model 4 Results',
            xaxis_title='Log GDP per Capita',
            yaxis_title='Predicted Log CO₂ Emissions',
            height=500, legend=dict(orientation='h', yanchor='bottom', y=-0.3)
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("### EKC Model 4 Results")
        st.markdown("""
        | Variable | Coef | p-value |
        |---|---|---|
        | renewable_share | -0.0272*** | <0.001 |
        | log_gdp | +0.5870*** | <0.001 |
        | **log_gdp²** | **-0.0271***  | **<0.001** |
        | energy_intensity | +0.0247*** | <0.001 |
        | R² | 0.9959 | — |
        | N | 3,072 | — |
        """)

        st.markdown("---")
        st.markdown("### Turning Point")
        st.metric("EKC Turning Point", "$50,490", delta="GDP per capita")
        st.markdown("""
        **Formula:** −β₁ / (2β₂)  
        = −0.5870 / (2 × −0.0271)  
        = 10.830 (log scale)  
        = exp(10.830)  
        = **$50,490**
        """)

        st.markdown("---")
        st.markdown("### What this means")
        st.markdown("""
        🔴 **Below $50,490** → Emissions RISE as income grows (most of the world)
        
        🟢 **Above $50,490** → Emissions FALL as income grows (only wealthiest nations)
        
        Only ~15 countries have crossed this threshold.
        """)

    st.markdown("---")
    st.markdown("### Income Groups and EKC Position")
    ekc_pos = pd.DataFrame({
        'Income Group': INCOME_ORDER,
        'Avg GDP/Capita': [31828, 5164, 1681, 521],
        'EKC Phase': ['⚠️ Near turning point','🔴 Rising phase','🔴 Rising phase','🔴 Rising phase'],
        'Distance from Turning Point': ['$18,662 below','$45,326 below','$48,809 below','$49,969 below']
    })
    st.dataframe(ekc_pos, use_container_width=True, hide_index=True)

# ═══════════════════════════════════════════════════════════════════════════
# PAGE 5 — INCOME GROUP DIAGNOSTIC
# ═══════════════════════════════════════════════════════════════════════════
elif page == "💡 Income Group Diagnostic":
    st.markdown('<div class="main-header">💡 Income Group Diagnostic Analysis</div>', unsafe_allow_html=True)
    st.info("Sub-group regressions used as a contextual diagnostic tool to explain patterns in the global interaction model (Objective 6).")

    st.markdown("""
    <div class="warning-box">
    <b>⭐ Key Finding:</b> Low-income countries experience the STRONGEST emissions reduction from renewable energy,
    while upper-middle income countries experience the WEAKEST. This is the opposite of what most literature assumes.
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([1,1])

    with col1:
        st.markdown("### Regression Coefficients by Income Group")
        ig_results = pd.DataFrame({
            'Income Group': INCOME_ORDER,
            'Renewable Coeff.': [-0.0262,-0.0125,-0.0296,-0.0465],
            'p-value': [0.0000,0.0000,0.0000,0.0000],
            'Significance': ['***','***','***','***'],
            'R²': [0.9967,0.9969,0.9966,0.9813],
            'N (obs)': [1063,926,706,343]
        })
        st.dataframe(ig_results, use_container_width=True, hide_index=True)
        st.caption("*** p<0.001. Country + year fixed effects in all models.")

        st.markdown("### Interpretation")
        st.markdown("""
        - **Low income (β = −0.0465):** Each 1% increase in renewable share reduces CO₂ by **4.65%**
        - **Lower middle (β = −0.0296):** Reduces CO₂ by **2.96%**
        - **High income (β = −0.0262):** Reduces CO₂ by **2.62%**
        - **Upper middle (β = −0.0125):** Reduces CO₂ by only **1.25%** — weakest effect
        
        **Why?** Upper-middle income countries (like China, Brazil) are simultaneously 
        industrialising rapidly, meaning energy demand growth offsets the renewable benefit.
        Low-income countries displace more carbon-intensive fuels proportionally.
        """)

    with col2:
        coefs = [-0.0262,-0.0125,-0.0296,-0.0465]
        groups_short = ['High income','Upper middle','Lower middle','Low income']

        fig = go.Figure()
        for i, (grp, coef, full) in enumerate(zip(groups_short, coefs, INCOME_ORDER)):
            fig.add_trace(go.Bar(
                x=[grp], y=[coef],
                name=full,
                marker_color=INCOME_COLORS[full],
                text=[f'{coef:.4f}***'],
                textposition='inside',
                textfont=dict(color='white', size=13, family='Arial Bold')
            ))
        fig.add_hline(y=0, line_dash='dash', line_color='black', line_width=1)
        fig.update_layout(
            title='Effect of Renewable Energy Share on Log CO₂<br>by World Bank Income Group (all p<0.001)',
            yaxis_title='Regression Coefficient (Log CO₂)',
            xaxis_title='Income Group',
            showlegend=False, height=450,
            plot_bgcolor='white',
            yaxis=dict(gridcolor='lightgrey')
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.markdown("### How this contextualises the Global Interaction Model")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="finding-box">
        <b>Global Model 2 found:</b><br>
        Positive interaction (renew × GDP) → effect weakens as GDP rises
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="warning-box">
        <b>But upper-middle breaks the pattern:</b><br>
        Their effect (−0.0125) is weaker than high income (−0.0262) — rapid industrialisation explains this anomaly
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="success-box">
        <b>Policy implication:</b><br>
        Climate finance should prioritise low-income countries where renewable investment delivers the MOST emissions reduction per dollar
        </div>""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════
# PAGE 6 — COUNTRY EXPLORER
# ═══════════════════════════════════════════════════════════════════════════
elif page == "🔍 Country Explorer":
    st.markdown('<div class="main-header">🔍 Country Explorer</div>', unsafe_allow_html=True)

    available_countries = sorted(df['Entity'].dropna().unique().tolist())
    selected_countries = st.multiselect(
        "Select countries to compare (max 6)",
        options=available_countries,
        default=['United Kingdom','India','China','Nigeria','Brazil','Germany'][:min(6,len(available_countries))]
    )

    if selected_countries:
        country_df = df[df['Entity'].isin(selected_countries)].copy()

        col1, col2 = st.columns(2)
        with col1:
            fig = px.line(country_df, x='Year', y='co2_emissions',
                          color='Entity', title='CO₂ Emissions Over Time',
                          labels={'co2_emissions':'CO₂ Emissions (kt)','Entity':'Country'})
            fig.update_layout(height=380)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig = px.line(country_df, x='Year', y='renewable_share',
                          color='Entity', title='Renewable Energy Share Over Time',
                          labels={'renewable_share':'Renewable Share (%)','Entity':'Country'})
            fig.update_layout(height=380)
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("### Country Comparison Table (Latest Available Year)")
        latest = country_df.sort_values('Year').groupby('Entity').last().reset_index()
        summary = latest[['Entity','income_group','co2_emissions','renewable_share',
                           'gdp_per_capita','gdp_growth']].copy()
        summary.columns = ['Country','Income Group','CO₂ (kt)','Renewable (%)','GDP/Capita','GDP Growth (%)']
        summary[['CO₂ (kt)','GDP/Capita']] = summary[['CO₂ (kt)','GDP/Capita']].round(0)
        summary[['Renewable (%)','GDP Growth (%)']] = summary[['Renewable (%)','GDP Growth (%)']].round(2)
        st.dataframe(summary, use_container_width=True, hide_index=True)

        st.markdown("### Scatter: GDP per Capita vs CO₂ Emissions")
        fig = px.scatter(country_df, x='gdp_per_capita', y='co2_emissions',
                         color='Entity', size='renewable_share',
                         animation_frame='Year',
                         title='GDP per Capita vs CO₂ Emissions (bubble size = renewable share)',
                         labels={'gdp_per_capita':'GDP per Capita (USD)',
                                 'co2_emissions':'CO₂ Emissions (kt)',
                                 'renewable_share':'Renewable Share (%)'})
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)

    else:
        st.info("Please select at least one country from the dropdown above.")

# ── FOOTER ───────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style='text-align:center; color:#888; font-size:0.85rem;'>
📊 Dissertation Dashboard — The Moderating Role of Economic Development in Renewable Energy & CO₂ Emissions |
Data: Kaggle Global Sustainable Energy + World Bank Income Classifications | Period: 2000–2019
</div>
""", unsafe_allow_html=True)
