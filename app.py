import pandas as pd
from dash import Dash, dcc, html, Input, Output, State, no_update, callback_context
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import re


# ---------------------------------------------------------
# 1. DATA LOADING (Parquet Only)
# ---------------------------------------------------------
def load_data():
    try:
        # Load pre-optimized Parquet file
        df = pd.read_parquet('Final_Data.parquet', engine='pyarrow')
        print(f"✅ Data loaded successfully: {len(df)} rows")
        return df
    except Exception as e:
        print(f"❌ Error reading Parquet file: {e}")
        # Minimal fallback
        return pd.DataFrame({'BOROUGH': ['BROOKLYN'], 'CRASH DATETIME': [pd.Timestamp('2023-01-01')]})


df = load_data()

# ---------------------------------------------------------
# 2. FEATURE ENGINEERING
# ---------------------------------------------------------
try:
    if 'CRASH DATETIME' in df.columns and not df.empty:
        df['Year'] = df['CRASH DATETIME'].dt.year.fillna(2023).astype(int)
        df['Hour'] = df['CRASH DATETIME'].dt.hour.fillna(12).astype(int)
        df['DayOfWeek'] = df['CRASH DATETIME'].dt.day_name().fillna('Monday')

        # SLIDER LOGIC
        start_date = df['CRASH DATETIME'].min()
        end_date = df['CRASH DATETIME'].max()
        df['days_from_start'] = (df['CRASH DATETIME'] - start_date).dt.days
        min_day = 0
        max_day = int(df['days_from_start'].max())

        # Generate Slider Marks (Years Only)
        slider_marks = {}
        unique_years = sorted(df['Year'].unique())
        for y in unique_years:
            if y == start_date.year:
                slider_marks[0] = {'label': str(y), 'style': {'font-weight': 'bold'}}
            else:
                jan_1 = pd.Timestamp(f"{y}-01-01")
                day_index = (jan_1 - start_date).days
                if 0 <= day_index <= max_day:
                    slider_marks[day_index] = {'label': str(y), 'style': {'font-weight': 'bold'}}
    else:
        raise ValueError("Empty DataFrame")

except Exception as e:
    print(f"Feature engineering error: {e}")
    df['days_from_start'] = 0
    min_day, max_day = 0, 1
    slider_marks = {0: "Start", 1: "End"}
    start_date = pd.Timestamp.now()
    end_date = pd.Timestamp.now()


# ---------------------------------------------------------
# 3. HELPER FUNCTIONS
# ---------------------------------------------------------
def create_safe_options(column_name, df, limit=500):
    try:
        if column_name not in df.columns: return []
        top_vals = df[column_name].value_counts().head(limit).index.tolist()
        cleaned = sorted([str(x) for x in top_vals if pd.notna(x) and str(x).strip() != ''], key=lambda s: s.lower())
        options = [{'label': x, 'value': x} for x in cleaned]
        options.insert(0, {'label': '✔ Select All / Reset', 'value': 'ALL'})
        return options
    except:
        return []


def create_combined_options(df, col1, col2, limit=500):
    try:
        v1 = df[col1].value_counts().head(limit).index.tolist() if col1 in df.columns else []
        v2 = df[col2].value_counts().head(limit).index.tolist() if col2 in df.columns else []
        combined = set(v1) | set(v2)
        cleaned = sorted([str(x).strip() for x in combined if str(x).strip() != ''], key=lambda s: s.lower())
        options = [{'label': x, 'value': x} for x in cleaned]
        options.insert(0, {'label': '✔ Select All / Reset', 'value': 'ALL'})
        return options
    except:
        return []


def create_sex_options(df):
    mapping = {'M': 'Male', 'F': 'Female', 'U': 'Unknown'}
    try:
        unique_vals = df['PERSON_SEX'].unique().dropna()
        if hasattr(unique_vals, 'tolist'): unique_vals = unique_vals.tolist()
        options = []
        for val in unique_vals:
            label = mapping.get(val, str(val))
            options.append({'label': label, 'value': val})
        options = sorted(options, key=lambda x: x['label'])
        options.insert(0, {'label': '✔ Select All / Reset', 'value': 'ALL'})
        return options
    except:
        return []


# --- GENERATE OPTIONS ---
borough_options = create_safe_options('BOROUGH', df)
zip_options = create_safe_options('ZIP CODE', df)
on_street_options = create_safe_options('ON STREET NAME', df, limit=1000)
cross_street_options = create_safe_options('CROSS STREET NAME', df, limit=1000)
vehicle_options = create_combined_options(df, 'VEHICLE TYPE CODE 1', 'VEHICLE TYPE CODE 2')
factor_options = create_combined_options(df, 'CONTRIBUTING FACTOR VEHICLE 1', 'CONTRIBUTING FACTOR VEHICLE 2')
p_type_options = create_safe_options('PERSON_TYPE', df)
injury_options = create_safe_options('PERSON_INJURY', df)
sex_options = create_sex_options(df)
age_options = create_safe_options('PERSON_AGE', df)
safety_options = create_safe_options('SAFETY_EQUIPMENT', df)
ejection_options = create_safe_options('EJECTION', df)
emotional_options = create_safe_options('EMOTIONAL_STATUS', df)
bodily_options = create_safe_options('BODILY_INJURY', df)
complaint_options = create_safe_options('COMPLAINT', df)
ped_role_options = create_safe_options('PED_ROLE', df)
ped_action_options = create_safe_options('PED_ACTION', df)
n_inj_options = create_safe_options('NUMBER OF PERSONS INJURED', df)
n_kill_options = create_safe_options('NUMBER OF PERSONS KILLED', df)
n_ped_inj_options = create_safe_options('NUMBER OF PEDESTRIANS INJURED', df)
n_ped_kill_options = create_safe_options('NUMBER OF PEDESTRIANS KILLED', df)

# ---------------------------------------------------------
# 4. APP LAYOUT
# ---------------------------------------------------------
app = Dash(__name__, external_stylesheets=[dbc.themes.FLATLY])

app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>NYC Traffic Dashboard</title>
        <link rel="icon" href="data:image/svg+xml,<svg xmlns=%22http://www.w3.org/2000/svg%22 viewBox=%220 0 100 100%22><text y=%22.9em%22 font-size=%2290%22>🗽</text></svg>">
        {%css%}
        <style>
            body {
                background-image: url('https://images.unsplash.com/photo-1496442226666-8d4d0e62e6e9?q=80&w=2070&auto=format&fit=crop');
                background-size: cover;
                background-attachment: fixed;
                background-position: center;
                background-repeat: no-repeat;
            }
            .card {
                background-color: rgba(255, 255, 255, 0.95) !important;
                border: none;
                box-shadow: 0 4px 15px rgba(0,0,0,0.2);
            }
            .main-header {
                background-color: rgba(44, 62, 80, 0.9);
                color: white;
                padding: 20px;
                border-radius: 10px;
                margin-top: 20px;
                box-shadow: 0 4px 15px rgba(0,0,0,0.3);
            }
            .rc-slider-track { background-color: #3498db; }
            .rc-slider-handle { border: solid 2px #3498db; background-color: #fff; }
            .rc-slider-tooltip { display: none !important; }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>{%config%}{%scripts%}{%renderer%}</footer>
    </body>
</html>
'''

app.layout = dbc.Container([

    # Header
    dbc.Row([
        dbc.Col([
            html.Div([
                html.H1("🗽 NYC Traffic Analysis Dashboard", className="text-center fw-bold mb-2"),
                html.P("Advanced Filtering & Timeline Analysis", className="text-center mb-0",
                       style={'opacity': '0.8'}),
            ], className="main-header mb-4")
        ], width=12)
    ]),

    # Filters Section
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("🔎 Comprehensive Filters", className="bg-primary text-white fw-bold"),
                dbc.CardBody([

                    # Smart Search
                    dbc.Label("⚡ Smart Search (Type & Press Enter)"),
                    dbc.InputGroup([
                        dbc.Input(id='search-input', placeholder='Try: "Male driver in Taxi on Broadway 2023"',
                                  type='text', n_submit=0),
                        dbc.Button("Apply Smart Search", id='search-button', color="secondary", n_clicks=0)
                    ], className="mb-3"),

                    # FILTERS ACCORDION
                    dbc.Accordion([

                        # Group 1: Location & Time
                        dbc.AccordionItem([
                            dbc.Row([
                                dbc.Col([
                                    dbc.Label("📅 Filter by Timeline"),
                                    dcc.RangeSlider(
                                        id='date-slider',
                                        min=min_day,
                                        max=max_day,
                                        value=[min_day, max_day],
                                        marks=slider_marks,
                                        step=1,
                                        updatemode='drag',
                                        tooltip=None
                                    ),
                                    dbc.Row([
                                        dbc.Col([
                                            dbc.Label("Start Date", className="text-muted small mb-0"),
                                            dbc.Input(id='date-picker-start', type='date', value=start_date.date(),
                                                      min=start_date.date(), max=end_date.date(),
                                                      style={'fontWeight': 'bold', 'color': '#3498db',
                                                             'textAlign': 'center'})
                                        ], width=6, className="pe-1"),
                                        dbc.Col([
                                            dbc.Label("End Date", className="text-muted small mb-0"),
                                            dbc.Input(id='date-picker-end', type='date', value=end_date.date(),
                                                      min=start_date.date(), max=end_date.date(),
                                                      style={'fontWeight': 'bold', 'color': '#3498db',
                                                             'textAlign': 'center'})
                                        ], width=6, className="ps-1"),
                                    ], className="mt-4")
                                ], width=12, className="mb-4 px-4"),

                                dbc.Col([dbc.Label("Borough"), dcc.Dropdown(id='borough-dd', options=borough_options,
                                                                            placeholder="Select Boroughs...",
                                                                            multi=True, className="mb-2")], width=6),
                                dbc.Col([dbc.Label("Zip Code"),
                                         dcc.Dropdown(id='zip-dd', options=zip_options, placeholder="Select Zips...",
                                                      multi=True, className="mb-2")], width=6),
                                dbc.Col([dbc.Label("On Street"),
                                         dcc.Dropdown(id='on-street-dd', options=on_street_options,
                                                      placeholder="Select Streets...", multi=True, className="mb-2")],
                                        width=6),
                                dbc.Col([dbc.Label("Cross Street"),
                                         dcc.Dropdown(id='cross-street-dd', options=cross_street_options,
                                                      placeholder="Select Streets...", multi=True, className="mb-2")],
                                        width=6),
                            ])
                        ], title="📍 Location & Time Timeline"),

                        # Group 2: Vehicle & Factors
                        dbc.AccordionItem([
                            dbc.Row([
                                dbc.Col([
                                    dbc.Label("🚗 Vehicle Type (Any involved)"),
                                    dcc.Dropdown(id='vehicle-dd', options=vehicle_options,
                                                 placeholder="Select Vehicle Types...", multi=True, className="mb-2")
                                ], width=12),

                                dbc.Col([
                                    dbc.Label("⚠️ Contributing Factors (Any involved)"),
                                    dcc.Dropdown(id='factor-dd', options=factor_options,
                                                 placeholder="Select Factors...", multi=True, className="mb-2")
                                ], width=12),
                            ])
                        ], title="🚗 Vehicles & Factors"),

                        # Group 3: Person Details
                        dbc.AccordionItem([
                            dbc.Row([
                                dbc.Col([dbc.Label("Person Type"),
                                         dcc.Dropdown(id='ptype-dd', options=p_type_options, placeholder="All",
                                                      multi=True, className="mb-2")], width=4),
                                dbc.Col([dbc.Label("Sex"),
                                         dcc.Dropdown(id='sex-dd', options=sex_options, placeholder="All", multi=True,
                                                      className="mb-2")], width=4),
                                dbc.Col([dbc.Label("Age"),
                                         dcc.Dropdown(id='age-dd', options=age_options, placeholder="All", multi=True,
                                                      className="mb-2")], width=4),
                                dbc.Col([dbc.Label("Injury"),
                                         dcc.Dropdown(id='injury-dd', options=injury_options, placeholder="All",
                                                      multi=True, className="mb-2")], width=6),
                                dbc.Col([dbc.Label("Safety Equip"),
                                         dcc.Dropdown(id='safety-dd', options=safety_options, placeholder="All",
                                                      multi=True, className="mb-2")], width=6),
                                dbc.Col([dbc.Label("Ejection"),
                                         dcc.Dropdown(id='eject-dd', options=ejection_options, placeholder="All",
                                                      multi=True, className="mb-2")], width=4),
                                dbc.Col([dbc.Label("Emotional"),
                                         dcc.Dropdown(id='emo-dd', options=emotional_options, placeholder="All",
                                                      multi=True, className="mb-2")], width=4),
                                dbc.Col([dbc.Label("Bodily Injury"),
                                         dcc.Dropdown(id='body-dd', options=bodily_options, placeholder="All",
                                                      multi=True, className="mb-2")], width=4),
                                dbc.Col([dbc.Label("Ped. Action"),
                                         dcc.Dropdown(id='ped-act-dd', options=ped_action_options, placeholder="All",
                                                      multi=True, className="mb-2")], width=6),
                                dbc.Col([dbc.Label("Ped. Role"),
                                         dcc.Dropdown(id='ped-role-dd', options=ped_role_options, placeholder="All",
                                                      multi=True, className="mb-2")], width=6),
                            ])
                        ], title="👥 Person Details"),

                        # Group 4: Casualty Counts
                        dbc.AccordionItem([
                            dbc.Row([
                                dbc.Col([dbc.Label("Total Injured"),
                                         dcc.Dropdown(id='n-inj-dd', options=n_inj_options, placeholder="All",
                                                      className="mb-2")], width=3),
                                dbc.Col([dbc.Label("Total Killed"),
                                         dcc.Dropdown(id='n-kill-dd', options=n_kill_options, placeholder="All",
                                                      className="mb-2")], width=3),
                                dbc.Col([dbc.Label("Ped. Injured"),
                                         dcc.Dropdown(id='n-ped-inj-dd', options=n_ped_inj_options, placeholder="All",
                                                      className="mb-2")], width=3),
                                dbc.Col([dbc.Label("Ped. Killed"),
                                         dcc.Dropdown(id='n-ped-kill-dd', options=n_ped_kill_options, placeholder="All",
                                                      className="mb-2")], width=3),
                            ])
                        ], title="🚑 Casualty Counts"),

                    ], start_collapsed=False, className="mb-3"),

                    dbc.Button("📊 GENERATE / REFRESH REPORT", id='generate-btn', color="success", size="lg",
                               className="w-100 fw-bold shadow-sm"),

                    # HIDDEN TRIGGER STORE
                    dcc.Store(id='smart-search-trigger', data=0)
                ])
            ], className="mb-4")
        ], width=12)
    ]),

    # Content Area
    dcc.Loading(id='report-loading', type='cube', color="#ecf0f1", children=[
        html.Div(id='report-content')
    ])

], fluid=True, className="pb-5")


# ---------------------------------------------------------
# 5. VISUALIZATION FUNCTIONS
# ---------------------------------------------------------
def get_clean_layout(title_text):
    return dict(
        title=dict(text=f"<b>{title_text}</b>", font=dict(size=20, family="Arial, sans-serif", color="#2c3e50"), x=0.5,
                   xanchor='center'),
        template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#2c3e50"), margin=dict(t=60, l=20, r=20, b=20)
    )


def create_map(data):
    """
    FULL DATA MAP (Optimized):
    1. Rounds coords to 4 decimals (approx 11m precision) to reduce JSON size.
    2. Uses minimal columns for the map dataframe.
    3. DISABLES HOVER info to prevent browser crash with 600k+ points.
    4. NO SAMPLING - Shows all valid injury/fatal points.
    """
    if 'LATITUDE' not in data.columns or 'LONGITUDE' not in data.columns:
        return go.Figure().add_annotation(text="No Coordinate Data")

    # 1. Minimal Columns & Filtering
    # Only keep strict necessary columns to save memory
    cols = ['LATITUDE', 'LONGITUDE']
    if 'PERSON_INJURY' in data.columns: cols.append('PERSON_INJURY')

    map_df = data[cols].dropna(subset=['LATITUDE', 'LONGITUDE']).copy()
    map_df = map_df[(map_df['LATITUDE'] != 0) & (map_df['LONGITUDE'] != 0)]

    # 2. Filter for Casualties ONLY
    if 'PERSON_INJURY' in map_df.columns:
        map_df = map_df[map_df['PERSON_INJURY'].isin(['Injured', 'Killed'])]

    if map_df.empty:
        return go.Figure().add_annotation(text="No Injuries or Fatalities found", showarrow=False)

    # 3. DATA OPTIMIZATION (Crucial for speed)
    # Rounding reduces JSON float size significantly
    map_df['LATITUDE'] = map_df['LATITUDE'].round(4)
    map_df['LONGITUDE'] = map_df['LONGITUDE'].round(4)

    # 4. Sort so KILLED (Red) is on TOP
    map_df['sort_val'] = map_df['PERSON_INJURY'].map({'Injured': 1, 'Killed': 2})
    map_df = map_df.sort_values('sort_val')

    count = len(map_df)

    # 5. Plot with scatter_map (GPU Accelerated)
    fig = px.scatter_map(
        map_df,
        lat='LATITUDE',
        lon='LONGITUDE',
        color='PERSON_INJURY',
        color_discrete_map={'Killed': '#e74c3c', 'Injured': '#f39c12'},
        zoom=10,
        center={"lat": 40.7128, "lon": -74.0060},  # NYC Center
        map_style="open-street-map",
        opacity=0.7,
        title=f"<b>🗺 Casualties Map ({count:,} records)</b>"
    )

    # 6. PERFORMANCE FIX: DISABLE HOVER for large datasets
    # This is what prevents the "laggy/heavy" feel on the frontend
    fig.update_traces(hoverinfo='skip', hovertemplate=None)

    fig.update_traces(marker=dict(size=5))  # Small dots
    fig.update_layout(height=600, margin={"r": 0, "t": 50, "l": 0, "b": 0}, font=dict(size=16, color="#2c3e50"))
    return fig


def create_charts(dff):
    charts = []
    # 1. Borough Bar
    if 'BOROUGH' in dff.columns:
        c = px.bar(dff['BOROUGH'].value_counts().reset_index().head(10), x='BOROUGH', y='count', color='count',
                   color_continuous_scale='Teal')
        c.update_layout(**get_clean_layout("📍 Crashes by Borough"))
        charts.append(c)
    else:
        charts.append(go.Figure())
    # 2. Trend Line
    if 'Hour' in dff.columns:
        trend = dff['Hour'].value_counts().sort_index().reset_index()
        c = px.area(trend, x='Hour', y='count', markers=True)
        c.update_layout(xaxis=dict(tickmode='linear', dtick=2), **get_clean_layout("📈 Hourly Trend"))
        charts.append(c)
    else:
        charts.append(go.Figure())
    # 3. Injury Pie
    if 'PERSON_INJURY' in dff.columns:
        c = px.pie(dff['PERSON_INJURY'].value_counts().reset_index(), values='count', names='PERSON_INJURY', hole=0.4)
        c.update_layout(**get_clean_layout("🚑 Injury Severity"))
        charts.append(c)
    else:
        charts.append(go.Figure())
    # 4. Age Histogram
    if 'PERSON_AGE' in dff.columns:
        age_df = dff[dff['PERSON_AGE'].between(0, 100)]
        c = px.histogram(age_df, x='PERSON_AGE', nbins=20, color_discrete_sequence=['#27ae60'])
        c.update_layout(bargap=0.1, **get_clean_layout("👥 Age Distribution"))
        charts.append(c)
    else:
        charts.append(go.Figure())
    # 5. Sunburst
    if 'BOROUGH' in dff.columns and 'VEHICLE TYPE CODE 1' in dff.columns:
        v_counts = dff.groupby(['BOROUGH', 'VEHICLE TYPE CODE 1'], observed=True).size().reset_index(name='count')
        v_counts = v_counts[v_counts['count'] > 0]
        c = px.sunburst(v_counts, path=['BOROUGH', 'VEHICLE TYPE CODE 1'], values='count')
        c.update_layout(height=500, **get_clean_layout("🍩 Borough → Vehicle Hierarchy"))
        charts.append(c)
    else:
        charts.append(go.Figure())
    # 6. Map
    charts.append(create_map(dff))
    # 7. Contributing Factors
    if 'CONTRIBUTING FACTOR VEHICLE 1' in dff.columns:
        fact_counts = dff['CONTRIBUTING FACTOR VEHICLE 1'].value_counts().head(10).reset_index()
        c = px.bar(fact_counts, y='CONTRIBUTING FACTOR VEHICLE 1', x='count', orientation='h',
                   title="⚠️ Top Contributing Factors", color='count', color_continuous_scale='Reds')
        c.update_layout(yaxis=dict(autorange="reversed"), **get_clean_layout("⚠️ Top Contributing Factors"))
        charts.append(c)
    else:
        charts.append(go.Figure())
    # 8. Top Streets
    if 'ON STREET NAME' in dff.columns:
        st_counts = dff['ON STREET NAME'].value_counts().head(10).reset_index()
        c = px.bar(st_counts, x='ON STREET NAME', y='count', title="🛣️ Most Dangerous Streets",
                   color_discrete_sequence=['#e67e22'])
        c.update_layout(**get_clean_layout("🛣️ Most Dangerous Streets"))
        charts.append(c)
    else:
        charts.append(go.Figure())
    # 9. Casualties Breakdown
    try:
        cols = ['NUMBER OF PEDESTRIANS INJURED', 'NUMBER OF PEDESTRIANS KILLED', 'NUMBER OF CYCLIST INJURED',
                'NUMBER OF CYCLIST KILLED', 'NUMBER OF MOTORIST INJURED', 'NUMBER OF MOTORIST KILLED']
        valid_cols = [c for c in cols if c in dff.columns]
        if valid_cols:
            sums = dff[valid_cols].sum().reset_index()
            sums.columns = ['Type', 'Count']
            sums['Category'] = sums['Type'].apply(
                lambda x: 'Pedestrian' if 'PEDESTRIANS' in x else ('Cyclist' if 'CYCLIST' in x else 'Motorist'))
            sums['Status'] = sums['Type'].apply(lambda x: 'Killed' if 'KILLED' in x else 'Injured')
            c = px.bar(sums, x='Category', y='Count', color='Status', barmode='group', title="🚑 Casualty Breakdown",
                       color_discrete_map={'Injured': '#f1c40f', 'Killed': '#c0392b'})
            c.update_layout(**get_clean_layout("🚑 Casualty Breakdown"))
            charts.append(c)
        else:
            charts.append(go.Figure())
    except:
        charts.append(go.Figure())

    # 10. Gender Donut (Mapped Labels)
    if 'PERSON_SEX' in dff.columns:
        g_counts = dff['PERSON_SEX'].value_counts().reset_index()
        g_counts.columns = ['Sex', 'Count']
        g_counts['Label'] = g_counts['Sex'].astype(str).map({'M': 'Male', 'F': 'Female', 'U': 'Unknown'}).fillna(
            g_counts['Sex'].astype(str))
        c = px.pie(g_counts, values='Count', names='Label', hole=0.6, title="👫 Gender Distribution")
        c.update_layout(**get_clean_layout("👫 Gender Distribution"))
        charts.append(c)
    else:
        charts.append(go.Figure())

    # 11. Safety Equipment
    if 'SAFETY_EQUIPMENT' in dff.columns:
        c = px.bar(dff['SAFETY_EQUIPMENT'].value_counts().head(8).reset_index(), x='count', y='SAFETY_EQUIPMENT',
                   orientation='h', title="🛡️ Safety Equipment")
        c.update_layout(**get_clean_layout("🛡️ Safety Equipment"))
        charts.append(c)
    else:
        charts.append(go.Figure())
    # 12. Bodily Injury
    if 'BODILY_INJURY' in dff.columns:
        c = px.bar(dff['BODILY_INJURY'].value_counts().head(8).reset_index(), x='BODILY_INJURY', y='count',
                   title="🏥 Bodily Injury Type")
        c.update_layout(**get_clean_layout("🏥 Bodily Injury Type"))
        charts.append(c)
    else:
        charts.append(go.Figure())
    return charts


# ---------------------------------------------------------
# 6. CALLBACKS
# ---------------------------------------------------------

# 1. CLIENTSIDE: AUTO-CLEAR "SELECT ALL"
app.clientside_callback(
    """
    function(borough, zip, street, cross, vehicle, factor, ptype, sex, age, inj, safe, eject, emo, body, act, role) {
        var args = Array.from(arguments);
        var outputs = new Array(args.length).fill(window.dash_clientside.no_update);

        for (var i = 0; i < args.length; i++) {
            if (args[i] && args[i].includes('ALL')) {
                outputs[i] = []; 
            }
        }
        return outputs;
    }
    """,
    [Output('borough-dd', 'value'), Output('zip-dd', 'value'), Output('on-street-dd', 'value'),
     Output('cross-street-dd', 'value'),
     Output('vehicle-dd', 'value'), Output('factor-dd', 'value'),
     Output('ptype-dd', 'value'), Output('sex-dd', 'value'), Output('age-dd', 'value'), Output('injury-dd', 'value'),
     Output('safety-dd', 'value'), Output('eject-dd', 'value'), Output('emo-dd', 'value'), Output('body-dd', 'value'),
     Output('ped-act-dd', 'value'), Output('ped-role-dd', 'value')],
    [Input('borough-dd', 'value'), Input('zip-dd', 'value'), Input('on-street-dd', 'value'),
     Input('cross-street-dd', 'value'),
     Input('vehicle-dd', 'value'), Input('factor-dd', 'value'),
     Input('ptype-dd', 'value'), Input('sex-dd', 'value'), Input('age-dd', 'value'), Input('injury-dd', 'value'),
     Input('safety-dd', 'value'), Input('eject-dd', 'value'), Input('emo-dd', 'value'), Input('body-dd', 'value'),
     Input('ped-act-dd', 'value'), Input('ped-role-dd', 'value')]
)


# 2. SYNC SLIDER <-> NATIVE PICKERS
@app.callback(
    [Output('date-slider', 'value'),
     Output('date-picker-start', 'value'),
     Output('date-picker-end', 'value')],
    [Input('date-slider', 'value'),
     Input('date-picker-start', 'value'),
     Input('date-picker-end', 'value')],
    prevent_initial_call=True
)
def sync_dates(slider_range, start_pick, end_pick):
    ctx = callback_context
    trigger = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else None

    if trigger == 'date-slider' and slider_range:
        d1 = start_date + pd.Timedelta(days=slider_range[0])
        d2 = start_date + pd.Timedelta(days=slider_range[1])
        return no_update, d1.date(), d2.date()
    elif (trigger == 'date-picker-start' or trigger == 'date-picker-end') and start_pick and end_pick:
        try:
            d1 = pd.to_datetime(start_pick)
            d2 = pd.to_datetime(end_pick)
            if d1 < start_date: d1 = start_date
            if d2 > end_date: d2 = end_date
            if d1 > d2: d1 = d2
            v1 = (d1 - start_date).days
            v2 = (d2 - start_date).days
            return [v1, v2], no_update, no_update
        except:
            return no_update, no_update, no_update
    return no_update, no_update, no_update


# 3. SMART SEARCH (TRIGGERS REPORT VIA STORE)
@app.callback(
    [Output('borough-dd', 'value', allow_duplicate=True), Output('zip-dd', 'value', allow_duplicate=True),
     Output('on-street-dd', 'value', allow_duplicate=True), Output('cross-street-dd', 'value', allow_duplicate=True),
     Output('vehicle-dd', 'value', allow_duplicate=True), Output('factor-dd', 'value', allow_duplicate=True),
     Output('ptype-dd', 'value', allow_duplicate=True), Output('sex-dd', 'value', allow_duplicate=True),
     Output('injury-dd', 'value', allow_duplicate=True), Output('safety-dd', 'value', allow_duplicate=True),
     Output('eject-dd', 'value', allow_duplicate=True), Output('emo-dd', 'value', allow_duplicate=True),
     Output('body-dd', 'value', allow_duplicate=True),
     Output('ped-act-dd', 'value', allow_duplicate=True), Output('ped-role-dd', 'value', allow_duplicate=True),
     Output('smart-search-trigger', 'data')],  # TRIGGERS REPORT AUTO-UPDATE
    [Input('search-button', 'n_clicks'), Input('search-input', 'n_submit')],
    State('search-input', 'value'),
    prevent_initial_call=True
)
def smart_search(n_clicks, n_submit, text):
    if not text: return [no_update] * 16
    text = text.lower()

    if "female" in text:
        sex_res = ['F']
        text = text.replace("female", "")
    elif "male" in text:
        sex_res = ['M']
        text = text.replace("male", "")
    else:
        sex_res = None

    def find(opts, t):
        for o in sorted(opts, key=lambda x: len(str(x['label'])), reverse=True):
            if str(o['label']).lower() in t: return [o['value']]
        return None

    zip_c = int(re.search(r'\b(\d{5})\b', text).group(1)) if re.search(r'\b(\d{5})\b', text) else None
    zip_res = [zip_c] if zip_c else None

    if not sex_res: sex_res = find(sex_options, text)

    # Increment trigger value to force update
    trigger_val = (n_clicks or 0) + (n_submit or 0)

    return (find(borough_options, text), zip_res, find(on_street_options, text), find(cross_street_options, text),
            find(vehicle_options, text), find(factor_options, text),
            find(p_type_options, text), sex_res, find(injury_options, text), find(safety_options, text),
            find(ejection_options, text), find(emotional_options, text), find(bodily_options, text),
            find(ped_action_options, text), find(ped_role_options, text), trigger_val)


# 4. MAIN REPORT (TRIGGERED BY BUTTON OR SMART SEARCH)
@app.callback(
    Output('report-content', 'children'),
    [Input('generate-btn', 'n_clicks'), Input('smart-search-trigger', 'data')],  # NEW INPUT
    [
        State('date-slider', 'value'),  # All filters are State (Passive)
        State('borough-dd', 'value'), State('zip-dd', 'value'),
        State('on-street-dd', 'value'), State('cross-street-dd', 'value'),
        State('vehicle-dd', 'value'), State('factor-dd', 'value'),
        State('ptype-dd', 'value'), State('sex-dd', 'value'), State('age-dd', 'value'),
        State('injury-dd', 'value'), State('safety-dd', 'value'), State('eject-dd', 'value'),
        State('emo-dd', 'value'), State('body-dd', 'value'), State('ped-act-dd', 'value'),
        State('ped-role-dd', 'value'),
        State('n-inj-dd', 'value'), State('n-kill-dd', 'value'), State('n-ped-inj-dd', 'value'),
        State('n-ped-kill-dd', 'value')]
)
def update_report(n_gen, n_smart, time_range,
                  borough, zip_c, on_st, cross_st,
                  vehicle, factor,
                  ptype, sex, age, inj, safe, eject, emo, body, pact, prole,
                  ninj, nkill, npinj, npkill):
    # Only run if either trigger has fired
    ctx = callback_context
    if not ctx.triggered:
        # Optional: Load initial report or show prompt
        pass

    dff = df.copy()

    if time_range: dff = dff[dff['days_from_start'].between(time_range[0], time_range[1])]

    # OPTIMIZED FILTERING using Categories
    if borough and 'ALL' not in borough: dff = dff[dff['BOROUGH'].isin(borough)]
    if zip_c and 'ALL' not in zip_c: dff = dff[dff['ZIP CODE'].isin(zip_c)]
    if on_st and 'ALL' not in on_st: dff = dff[dff['ON STREET NAME'].isin(on_st)]
    if cross_st and 'ALL' not in cross_st: dff = dff[dff['CROSS STREET NAME'].isin(cross_st)]

    if vehicle and 'ALL' not in vehicle:
        dff = dff[dff['VEHICLE TYPE CODE 1'].isin(vehicle) | dff['VEHICLE TYPE CODE 2'].isin(vehicle)]
    if factor and 'ALL' not in factor:
        dff = dff[dff['CONTRIBUTING FACTOR VEHICLE 1'].isin(factor) | dff['CONTRIBUTING FACTOR VEHICLE 2'].isin(factor)]

    if ptype and 'ALL' not in ptype: dff = dff[dff['PERSON_TYPE'].isin(ptype)]
    if sex and 'ALL' not in sex: dff = dff[dff['PERSON_SEX'].isin(sex)]
    if age and 'ALL' not in age: dff = dff[dff['PERSON_AGE'].isin(age)]
    if inj and 'ALL' not in inj: dff = dff[dff['PERSON_INJURY'].isin(inj)]
    if safe and 'ALL' not in safe: dff = dff[dff['SAFETY_EQUIPMENT'].isin(safe)]
    if eject and 'ALL' not in eject: dff = dff[dff['EJECTION'].isin(eject)]
    if emo and 'ALL' not in emo: dff = dff[dff['EMOTIONAL_STATUS'].isin(emo)]
    if body and 'ALL' not in body: dff = dff[dff['BODILY_INJURY'].isin(body)]
    if pact and 'ALL' not in pact: dff = dff[dff['PED_ACTION'].isin(pact)]
    if prole and 'ALL' not in prole: dff = dff[dff['PED_ROLE'].isin(prole)]

    if ninj: dff = dff[dff['NUMBER OF PERSONS INJURED'] == ninj]
    if nkill: dff = dff[dff['NUMBER OF PERSONS KILLED'] == nkill]
    if npinj: dff = dff[dff['NUMBER OF PEDESTRIANS INJURED'] == npinj]
    if npkill: dff = dff[dff['NUMBER OF PEDESTRIANS KILLED'] == npkill]

    if len(dff) == 0:
        return dbc.Alert("No records found matching your filters.", color="warning", className="mt-4")

    charts = create_charts(dff)

    return html.Div([
        dbc.Row([dbc.Col(dbc.Card(
            [dbc.CardBody([html.H2(f"{len(dff):,}", className="text-primary fw-bold"), html.P("Records Found")])],
            className="mb-4 text-center"), width=12)]),
        dbc.Row([dbc.Col(dbc.Card(dcc.Graph(figure=charts[0]), body=True), width=12, md=6),
                 dbc.Col(dbc.Card(dcc.Graph(figure=charts[1]), body=True), width=12, md=6)], className="mb-4"),
        dbc.Row([dbc.Col(dbc.Card(dcc.Graph(figure=charts[2]), body=True), width=12, md=4),
                 dbc.Col(dbc.Card(dcc.Graph(figure=charts[9]), body=True), width=12, md=4),
                 dbc.Col(dbc.Card(dcc.Graph(figure=charts[3]), body=True), width=12, md=4)], className="mb-4"),
        dbc.Row([dbc.Col(dbc.Card(dcc.Graph(figure=charts[8]), body=True), width=12, md=6),
                 dbc.Col(dbc.Card(dcc.Graph(figure=charts[11]), body=True), width=12, md=6)], className="mb-4"),
        dbc.Row([dbc.Col(dbc.Card(dcc.Graph(figure=charts[6]), body=True), width=12, md=6),
                 dbc.Col(dbc.Card(dcc.Graph(figure=charts[7]), body=True), width=12, md=6)], className="mb-4"),
        dbc.Row([dbc.Col(dbc.Card(dcc.Graph(figure=charts[10]), body=True), width=12, md=6),
                 dbc.Col(dbc.Card(dcc.Graph(figure=charts[4]), body=True), width=12, md=6)], className="mb-4"),
        dbc.Row([dbc.Col(dbc.Card(dcc.Graph(figure=charts[5]), body=True), width=12)], className="mb-4"),
    ])


if __name__ == '__main__':
    app.run(debug=True, port=8050)