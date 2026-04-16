import streamlit as st
import pandas as pd
import plotly.express as px

# 1. Page Configuration
st.set_page_config(page_title="Malta 4DW Impact Analysis", layout="wide")

# 2. Coordinate Database (Comprehensive Malta & Gozo)
COORDS = {
    'Birkirkara': [35.8972, 14.4611], 'Il-Gżira': [35.9058, 14.4961], 'Il-Ħamrun': [35.8847, 14.4844],
    'L-Imsida': [35.8967, 14.4892], 'Pembroke': [35.9306, 14.4764], 'Tal-Pietà': [35.8944, 14.4944],
    'Ħal Qormi': [35.8794, 14.4722], 'San Ġiljan': [35.9183, 14.4883], 'San Ġwann': [35.9094, 14.4786],
    'Santa Venera': [35.8903, 14.4736], 'Tas-Sliema': [35.9122, 14.5042], 'Is-Swieqi': [35.9225, 14.4800],
    'Il-Mellieħa': [35.9333, 14.3667], 'Il-Mosta': [35.9097, 14.4261], 'In-Naxxar': [35.9150, 14.4447],
    'San Pawl il-Baħar': [35.9483, 14.4017], 'Il-Belt Valletta': [35.8989, 14.5145], 'Ħaż-Żabbar': [35.8772, 14.5381],
    'Victoria': [36.0444, 14.2397], 'Ix-Xagħra': [36.0500, 14.2644], 'In-Nadur': [36.0375, 14.2883],
    'Iż-Żebbuġ': [36.0720, 14.2410], 'Ix-Xewkija': [36.0328, 14.2581]
}

# 3. Data Loading
@st.cache_data
def load_data():
    df_base = pd.read_csv("synthetic_trips.csv")
    df_4dw = pd.read_csv("4dw_full_population.csv")
    df_base['scenario'] = '5-Day (Baseline)'
    df_4dw['scenario'] = '4-Day Week'
    return df_base, df_4dw, pd.concat([df_base, df_4dw], ignore_index=True)

df_base, df_4dw, df_all = load_data()

# --- CALCULATE KPI METRICS ---
total_base = len(df_base)
total_4dw = len(df_4dw)
decrease_raw = total_base - total_4dw
pct_decrease = (decrease_raw / total_base) * 100

# --- TOP HEADER (METRICS) ---
st.title("🚗 Malta 4-Day Work Week Impact Study")
m1, m2, m3 = st.columns(3)
m1.metric("Trips (5-Day Baseline)", f"{total_base:,}")
m2.metric("Trips (4-Day Week)", f"{total_4dw:,}", delta=f"-{decrease_raw:,} trips")
m3.metric("Reduction in Traffic Volume", f"{pct_decrease:.1f}%", delta_color="normal")

st.divider()

# 4. SIDEBAR NAVIGATION
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to:", ["Geographic Heatmap", "Data Insights & Graphs"])

st.sidebar.divider()
st.sidebar.subheader("Scenario Selection")
scen_choice = st.sidebar.selectbox("Select View for Visuals:", ["5-Day (Baseline)", "4-Day Week"])
active_df = df_base if scen_choice == "5-Day (Baseline)" else df_4dw

# -------------------------------------------------------------------------
# PAGE 1: GEOGRAPHIC HEATMAP
# -------------------------------------------------------------------------
if page == "Geographic Heatmap":
    st.subheader(f"📍 Trip Density Heatmap ({scen_choice})")
    
    h_df = active_df['predicted_origin'].value_counts().reset_index()
    h_df.columns = ['locality', 'trips']
    h_df = h_df[h_df['locality'].isin(COORDS.keys())]
    h_df['lat'] = h_df['locality'].apply(lambda x: COORDS[x][0])
    h_df['lon'] = h_df['locality'].apply(lambda x: COORDS[x][1])

    fig_map = px.density_mapbox(
        h_df, lat='lat', lon='lon', z='trips', 
        radius=30, zoom=10, height=700,
        center={"lat": 35.95, "lon": 14.40},
        mapbox_style="carto-positron",
        color_continuous_scale="Plasma",
        title="Heat Intensity: Darker = More Congestion"
    )
    st.plotly_chart(fig_map, use_container_width=True)

# -------------------------------------------------------------------------
# PAGE 2: DATA INSIGHTS & GRAPHS
# -------------------------------------------------------------------------
elif page == "Data Insights & Graphs":
    st.subheader("📊 Statistical Comparison")
    
    # 1. TOP 10 BUSIEST AREAS GRAPH
    st.markdown("### 🏙️ Top 10 Busiest Localities")
    top_towns = active_df['predicted_origin'].value_counts().nlargest(10).reset_index()
    top_towns.columns = ['Town', 'Total Trips']
    
    fig_busiest = px.bar(top_towns, x='Total Trips', y='Town', orientation='h',
                         color='Total Trips', color_continuous_scale='Viridis',
                         text_auto='.2s')
    fig_busiest.update_layout(yaxis={'categoryorder':'total ascending'})
    st.plotly_chart(fig_busiest, use_container_width=True)

    st.divider()

    # 2. THE TRAFFIC CLOCK (Line Graph)
    st.markdown("### ⏰ The Traffic Clock (Congestion over Time)")
    time_order = ['00:00 - 02:59', '03:00 - 05:59', '06:00 - 08:59', '09:00 - 11:59', 
                  '12:00 - 14:59', '15:00 - 17:59', '18:00 - 20:59', '21:00 - 23:59']
    time_stats = df_all.groupby(['time_bin', 'scenario']).size().reset_index(name='Count')
    time_stats['time_bin'] = pd.Categorical(time_stats['time_bin'], categories=time_order, ordered=True)
    
    fig_line = px.line(time_stats.sort_values('time_bin'), x='time_bin', y='Count', color='scenario', 
                       markers=True, title="Hourly Volume Comparison")
    st.plotly_chart(fig_line, use_container_width=True)

    st.divider()

    # 3. MODE & PURPOSE
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 🚌 Transport Mode Split")
        st.plotly_chart(px.pie(active_df, names='mode', hole=0.5), use_container_width=True)
    with col2:
        st.markdown("### 🎯 Trip Purpose Analysis")
        purpose_stats = df_all.groupby(['purpose', 'scenario']).size().reset_index(name='Count')
        st.plotly_chart(px.bar(purpose_stats, x='purpose', y='Count', color='scenario', barmode='group'), use_container_width=True)