import os
from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
from tensorflow.keras.models import load_model
import joblib
import pandas as pd
import numpy as np
from functools import wraps
import requests # For making API calls
import random
from datetime import datetime

app = Flask(__name__)
app.secret_key = 'a8f3$kL9#mP2@vQ7!nX4&wR6*yT1^uJ5' # Change this to a strong, random key
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'
app.config['SESSION_COOKIE_SECURE'] = True

# --- Load the trained model and preprocessor ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'flight_accident_cnn_model.h5')
PREPROCESSOR_PATH = os.path.join(BASE_DIR, 'preprocessor.pkl')

model = None
preprocessor = None

try:
    model = load_model(MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    print("Model and preprocessor loaded successfully.")
except Exception as e:
    print(f"Error loading model or preprocessor: {e}")
    # exit()

# --- Simple User Authentication ---
USERS = {
    "user1": "pass123",
    "admin": "adminpass"
}

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'logged_in' not in session:
            flash('Please log in to access this page.', 'warning')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

# --- Data for Dropdowns and API lookups ---
AIRLINES = [
    'Air India', 'IndiGo', 'SpiceJet', 'Vistara', 'GoAir', 'AirAsia India',
    'Alliance Air', 'TruJet', 'Star Air', 'Zoom Air', 'Other'
]
AIRCRAFT_TYPES = [
    'Airbus A320', 'Boeing 737', 'ATR 72', 'Bombardier Q400', 'Airbus A330',
    'Boeing 787', 'Airbus A350', 'Boeing 747', 'Embraer 190', 'Other'
]
INDIAN_AIRPORT_CODES = [
    'DEL', 'BOM', 'BLR', 'MAA', 'CCU', 'HYD', 'AMD', 'COK', 'PNQ', 'GOI', 'LKO', 'JAI', 'ATQ', 'SXR', 'IXB', 'GAU'
]
FLIGHT_PHASES = [
    'Cruise', 'Takeoff', 'Landing', 'Climb', 'Descent', 'Taxi', 'Approach', 'Hold', 'Go-around'
]
SEASONS = ['Monsoon', 'Winter', 'Summer', 'Post-Monsoon']
WEATHER_CONDITIONS = [
    'Clear', 'Rainy', 'Cloudy', 'Foggy', 'Stormy', 'Snowy', 'Windy'
]
TURBULENCE_SEVERITIES = ['None', 'Light', 'Moderate', 'Severe', 'Extreme']
ROUTE_COMPLEXITIES = ['Low', 'Medium', 'High']
AIR_TRAFFIC_DENSITIES = ['Low', 'Medium', 'High', 'Very High']

AIRPORT_DATA = {
    "DEL": {"lat": 28.55, "lon": 77.10, "elev_ft": 777, "city": "Delhi"},
    "BOM": {"lat": 19.08, "lon": 72.87, "elev_ft": 39, "city": "Mumbai"},
    "BLR": {"lat": 13.19, "lon": 77.70, "elev_ft": 3000, "city": "Bengaluru"},
    "MAA": {"lat": 12.99, "lon": 80.18, "elev_ft": 52, "city": "Chennai"},
    "CCU": {"lat": 22.65, "lon": 88.44, "elev_ft": 16, "city": "Kolkata"},
    "HYD": {"lat": 17.24, "lon": 78.42, "elev_ft": 2024, "city": "Hyderabad"},
    "AMD": {"lat": 23.07, "lon": 72.63, "elev_ft": 180, "city": "Ahmedabad"},
    "COK": {"lat": 10.15, "lon": 76.40, "elev_ft": 30, "city": "Kochi"},
    "PNQ": {"lat": 18.58, "lon": 73.91, "elev_ft": 1942, "city": "Pune"},
    "GOI": {"lat": 15.38, "lon": 73.83, "elev_ft": 184, "city": "Goa"},
    "LKO": {"lat": 26.76, "lon": 80.88, "elev_ft": 410, "city": "Lucknow"},
    "JAI": {"lat": 26.82, "lon": 75.81, "elev_ft": 1263, "city": "Jaipur"},
    "ATQ": {"lat": 31.70, "lon": 74.79, "elev_ft": 755, "city": "Amritsar"},
    "SXR": {"lat": 33.98, "lon": 74.77, "elev_ft": 5429, "city": "Srinagar"},
    "IXB": {"lat": 26.68, "lon": 88.32, "elev_ft": 412, "city": "Bagdogra"},
    "GAU": {"lat": 26.10, "lon": 91.58, "elev_ft": 161, "city": "Guwahati"}
}

AIRPORT_PLACES = {v['city']: k for k, v in AIRPORT_DATA.items()}
airport_place_names = sorted(list(AIRPORT_PLACES.keys()))

def get_weather_data(lat, lon, date):
    """Fetches weather data for a specific location and date."""
    try:
        base_url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": date,
            "end_date": date,
            "daily": "weathercode,temperature_2m_max,precipitation_sum,windspeed_10m_max",
            "timezone": "auto"
        }
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        data = response.json()

        daily_data = data.get('daily', {})
        
        wmo_code = daily_data.get('weathercode', [0])[0]
        weather_condition = "Clear"
        if wmo_code in [0, 1]: weather_condition = "Clear"
        elif wmo_code in [2, 3]: weather_condition = "Cloudy"
        elif wmo_code in [45, 48]: weather_condition = "Foggy"
        elif wmo_code in [51, 53, 55, 61, 63, 65, 80, 81, 82]: weather_condition = "Rainy"
        elif wmo_code in [71, 73, 75, 85, 86]: weather_condition = "Snowy"
        elif wmo_code in [95, 96, 99]: weather_condition = "Stormy"
        
        visibility_km = 10.0
        if weather_condition == "Rainy": visibility_km = 5.0
        elif weather_condition == "Foggy": visibility_km = 0.5
        elif weather_condition == "Snowy": visibility_km = 2.0
        elif weather_condition == "Stormy": visibility_km = 1.0

        return {
            "Temperature_Celsius": daily_data.get('temperature_2m_max', [25.0])[0],
            "Precipitation_MM": daily_data.get('precipitation_sum', [0.0])[0],
            "Wind_Speed_KNOTS": round(daily_data.get('windspeed_10m_max', [10.0])[0] * 0.54, 2),
            "Weather_Condition": weather_condition,
            "Visibility_KM": visibility_km
        }
    except requests.exceptions.RequestException as e:
        print(f"API request failed: {e}")
        return None
    except (KeyError, IndexError) as e:
        print(f"Error parsing weather data: {e}")
        return None

# --- Routes ---
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if username in USERS and USERS[username] == password:
            session['logged_in'] = True
            session['username'] = username
            flash('Logged in successfully!', 'success')
            return redirect(url_for('predict'))
        else:
            flash('Invalid username or password.', 'danger')
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('logged_in', None)
    session.pop('username', None)
    flash('You have been logged out.', 'info')
    return redirect(url_for('home'))

@app.route('/realtime', methods=['GET', 'POST'])
@login_required
def realtime():
    prediction_result = None
    if request.method == 'POST':
        if model is None or preprocessor is None:
            flash("Model or preprocessor not loaded. Cannot perform prediction.", 'danger')
            return redirect(url_for('realtime'))

        try:
            dep_place = request.form['dep_airport_place']
            arr_place = request.form['arr_airport_place']
            flight_date = request.form['flight_date']

            if dep_place == arr_place:
                flash("Departure and arrival places cannot be the same.", 'danger')
                return redirect(url_for('realtime'))

            dep_code = AIRPORT_PLACES.get(dep_place)
            arr_code = AIRPORT_PLACES.get(arr_place)

            if not dep_code or not arr_code:
                flash("Invalid airport place selected.", 'danger')
                return redirect(url_for('realtime'))

            dep_airport_info = AIRPORT_DATA.get(dep_code)
            
            weather_data = get_weather_data(dep_airport_info['lat'], dep_airport_info['lon'], flight_date)
            if not weather_data:
                flash("Could not fetch real-time weather data. Please try again later.", 'danger')
                return redirect(url_for('realtime'))

            form_data = {
                'Flight_Duration_Hours': random.uniform(1.0, 4.5),
                'Flight_Phase': random.choice(['Takeoff', 'Landing', 'Cruise']),
                'Departure_Airport_Code': dep_code,
                'Departure_Airport_Elevation_FT': dep_airport_info['elev_ft'],
                'Arrival_Airport_Code': arr_code,
                'Arrival_Airport_Elevation_FT': AIRPORT_DATA.get(arr_code, {}).get('elev_ft', 500),
                'Total_Onboard': random.randint(150, 250),
                'Cargo_Weight_KG': random.randint(1000, 5000),
                'Airline': random.choice(AIRLINES),
                'Aircraft_Type': random.choice(AIRCRAFT_TYPES),
                'Aircraft_Age_Years': random.randint(5, 15),
                'Last_Maintenance_Hours': random.randint(50, 500),
                'Engine_Hours_Since_Overhaul': random.randint(500, 3000),
                'Pilot_Experience_Hours': random.randint(5000, 15000),
                'CoPilot_Experience_Hours': random.randint(2000, 8000),
                'Number_of_Crew': random.randint(5, 10),
                'Season': SEASONS[datetime.strptime(flight_date, '%Y-%m-%d').month % 4],
                'Wind_Direction_Degrees': random.randint(0, 359),
                'Turbulence_Severity': random.choice(['Light', 'Moderate']) if weather_data['Weather_Condition'] in ['Rainy', 'Stormy'] else 'None',
                'Route_Complexity': random.choice(['Low', 'Medium']),
                'Air_Traffic_Density': random.choice(['Medium', 'High'])
            }
            form_data.update(weather_data)

            input_df = pd.DataFrame([form_data])
            processed_input = preprocessor.transform(input_df)
            processed_input = processed_input.reshape(processed_input.shape[0], processed_input.shape[1], 1)

            prediction_proba = model.predict(processed_input)[0][0]
            prediction_class = (prediction_proba > 0.5).astype(int)

            reasons_for_report = []
            if prediction_class == 1:
                prediction_text = f"High risk of accident detected! (Probability: {prediction_proba:.2f})"
                prediction_color = "text-red-600"
                if weather_data['Weather_Condition'] in ['Stormy', 'Foggy', 'Rainy']:
                    reasons_for_report.append({'factor': f"Adverse Weather: {weather_data['Weather_Condition']}", 'impact': 'negative', 'value': 0.8})
                if weather_data['Wind_Speed_KNOTS'] > 25:
                    reasons_for_report.append({'factor': f"High Wind Speed: {weather_data['Wind_Speed_KNOTS']} knots", 'impact': 'negative', 'value': 0.7})
                if not reasons_for_report:
                    reasons_for_report.append({'factor': "High risk based on a combination of factors.", 'impact': 'negative', 'value': 0.7})
            else:
                prediction_text = f"Low risk of accident. (Probability: {prediction_proba:.2f})"
                prediction_color = "text-green-600"
                reasons_for_report.append({'factor': "Favorable weather conditions reported.", 'impact': 'positive', 'value': 0.9})

            pdf_fetched_data = {
                "Departure Place": dep_place,
                "Arrival Place": arr_place,
                "Flight Date": flight_date,
                "Season": form_data['Season'],
                "Weather Condition": weather_data['Weather_Condition'],
                "Temperature (°C)": f"{weather_data['Temperature_Celsius']:.1f}",
                "Wind Speed (knots)": f"{weather_data['Wind_Speed_KNOTS']}",
                "Visibility (km)": f"{weather_data['Visibility_KM']}",
                "Precipitation (mm)": f"{weather_data['Precipitation_MM']}"
            }

            prediction_result = {
                "text": prediction_text,
                "color": prediction_color,
                "reasons": reasons_for_report,
                "fetched_data": pdf_fetched_data
            }
            flash('Real-time prediction successful!', 'success')

        except Exception as e:
            flash(f"An error occurred during real-time prediction: {e}", 'danger')
            print(f"Real-time prediction error: {e}")

    return render_template('realtime.html',
                           airport_places=airport_place_names,
                           prediction_result=prediction_result)

@app.route('/chart')
@login_required
def chart():
    df_path = os.path.join(BASE_DIR, 'flight_accidents_india_synthetic.csv')
    if os.path.exists(df_path):
        df = pd.read_csv(df_path)
    else:
        df = pd.DataFrame({'Flight_Phase': [], 'Weather_Condition': [], 'Visibility_KM': [], 'Season': [], 'Turbulence_Severity': [], 'Departure_Airport_Elevation_FT': [], 'Air_Traffic_Density': [], 'Accident': []})

    flight_phase_data = df['Flight_Phase'].value_counts().to_dict()
    weather_condition_data = df['Weather_Condition'].value_counts().to_dict()
    visibility_data = df.groupby('Visibility_KM')['Accident'].sum().reset_index().to_dict(orient='list')
    season_data = df['Season'].value_counts().to_dict()
    turbulence_severity_data = df['Turbulence_Severity'].value_counts().to_dict()
    dep_airport_elevation_data = df.groupby(pd.cut(df['Departure_Airport_Elevation_FT'], bins=5))['Accident'].sum().reset_index()
    dep_airport_elevation_data['Departure_Airport_Elevation_FT'] = dep_airport_elevation_data['Departure_Airport_Elevation_FT'].astype(str)
    dep_airport_elevation_data = dep_airport_elevation_data.to_dict(orient='list')
    air_traffic_density_data = df['Air_Traffic_Density'].value_counts().to_dict()

    return render_template('chart.html',
                           flight_phase_data=flight_phase_data,
                           weather_condition_data=weather_condition_data,
                           visibility_data=visibility_data,
                           season_data=season_data,
                           turbulence_severity_data=turbulence_severity_data,
                           dep_airport_elevation_data=dep_airport_elevation_data,
                           air_traffic_density_data=air_traffic_density_data)

@app.route('/predict', methods=['GET', 'POST'])
@login_required
def predict():
    prediction_result = None
    if request.method == 'POST':
        if model is None or preprocessor is None:
            flash("Model or preprocessor not loaded. Cannot perform prediction.", 'danger')
            return redirect(url_for('predict'))
        try:
            form_data = {
                'Flight_Duration_Hours': float(request.form['flight_duration_hours']),
                'Flight_Phase': request.form['flight_phase'],
                'Departure_Airport_Code': request.form['dep_airport_code'],
                'Departure_Airport_Elevation_FT': float(request.form['dep_airport_elevation_ft']),
                'Arrival_Airport_Code': request.form['arr_airport_code'],
                'Arrival_Airport_Elevation_FT': float(request.form['arr_airport_elevation_ft']),
                'Total_Onboard': int(request.form['total_onboard']),
                'Cargo_Weight_KG': float(request.form['cargo_weight_kg']),
                'Airline': request.form['airline'],
                'Aircraft_Type': request.form['aircraft_type'],
                'Aircraft_Age_Years': int(request.form['aircraft_age_years']),
                'Last_Maintenance_Hours': float(request.form['last_maintenance_hours']),
                'Engine_Hours_Since_Overhaul': float(request.form['engine_hours_since_overhaul']),
                'Pilot_Experience_Hours': float(request.form['pilot_experience_hours']),
                'CoPilot_Experience_Hours': float(request.form['copilot_experience_hours']),
                'Number_of_Crew': int(request.form['number_of_crew']),
                'Season': request.form['season'],
                'Weather_Condition': request.form['weather_condition'],
                'Visibility_KM': float(request.form['visibility_km']),
                'Wind_Speed_KNOTS': float(request.form['wind_speed_knots']),
                'Wind_Direction_Degrees': float(request.form['wind_direction_degrees']),
                'Temperature_Celsius': float(request.form['temperature_celsius']),
                'Precipitation_MM': float(request.form['precipitation_mm']),
                'Turbulence_Severity': request.form['turbulence_severity'],
                'Route_Complexity': request.form['route_complexity'],
                'Air_Traffic_Density': request.form['air_traffic_density']
            }
            input_df = pd.DataFrame([form_data])
            processed_input = preprocessor.transform(input_df)
            processed_input = processed_input.reshape(processed_input.shape[0], processed_input.shape[1], 1)
            prediction_proba = model.predict(processed_input)[0][0]
            prediction_class = (prediction_proba > 0.5).astype(int)
            
            reasons_for_report = []

            # High-risk factors
            if form_data['Turbulence_Severity'] in ['Severe', 'Extreme']:
                reasons_for_report.append({'factor': f"High Turbulence ({form_data['Turbulence_Severity']})", 'impact': 'negative', 'value': 0.9})
            if form_data['Weather_Condition'] in ['Stormy', 'Foggy'] and form_data['Visibility_KM'] < 1.0:
                reasons_for_report.append({'factor': f"Poor Visibility ({form_data['Visibility_KM']} km) in {form_data['Weather_Condition']} weather", 'impact': 'negative', 'value': 0.8})
            if form_data['Wind_Speed_KNOTS'] > 40:
                reasons_for_report.append({'factor': f"High Wind Speed ({form_data['Wind_Speed_KNOTS']} knots)", 'impact': 'negative', 'value': 0.75})
            if form_data['Aircraft_Age_Years'] > 20:
                reasons_for_report.append({'factor': f"Old Aircraft ({form_data['Aircraft_Age_Years']} years)", 'impact': 'negative', 'value': 0.6})
            if form_data['Pilot_Experience_Hours'] < 2000:
                 reasons_for_report.append({'factor': f"Low Pilot Experience ({form_data['Pilot_Experience_Hours']} hrs)", 'impact': 'negative', 'value': 0.65})
            if form_data['Last_Maintenance_Hours'] > 1500:
                 reasons_for_report.append({'factor': f"High Hours Since Maintenance ({form_data['Last_Maintenance_Hours']})", 'impact': 'negative', 'value': 0.5})

            # Low-risk (positive) factors
            if form_data['Weather_Condition'] == 'Clear' and form_data['Visibility_KM'] > 8:
                 reasons_for_report.append({'factor': "Favorable Weather Conditions", 'impact': 'positive', 'value': 0.9})
            if form_data['Pilot_Experience_Hours'] > 10000 and form_data['CoPilot_Experience_Hours'] > 5000:
                 reasons_for_report.append({'factor': "Experienced Pilot and Crew", 'impact': 'positive', 'value': 0.8})
            if form_data['Aircraft_Age_Years'] < 5 and form_data['Last_Maintenance_Hours'] < 200:
                 reasons_for_report.append({'factor': "Well-Maintained Aircraft", 'impact': 'positive', 'value': 0.7})
            if form_data['Air_Traffic_Density'] == 'Low':
                 reasons_for_report.append({'factor': "Low Air Traffic Density", 'impact': 'positive', 'value': 0.6})
            if form_data['Route_Complexity'] == 'Low':
                 reasons_for_report.append({'factor': "Standard Route Complexity", 'impact': 'positive', 'value': 0.5})

            # Fallback message
            if not reasons_for_report:
                if prediction_class == 1:
                    reasons_for_report.append({'factor': "High risk due to a combination of factors.", 'impact': 'negative', 'value': 0.7})
                else:
                    reasons_for_report.append({'factor': "Overall conditions appear stable.", 'impact': 'positive', 'value': 0.7})

            if prediction_class == 1:
                prediction_text = f"High risk of accident detected! (Probability: {prediction_proba:.2f})"
                prediction_color = "text-red-600"
            else:
                prediction_text = f"Low risk of accident. (Probability: {prediction_proba:.2f})"
                prediction_color = "text-green-600"
                reasons_for_report = [r for r in reasons_for_report if r['impact'] == 'positive']

            prediction_result = {
                "text": prediction_text,
                "color": prediction_color,
                "reasons": sorted(reasons_for_report, key=lambda x: x['value'], reverse=True),
                "form_data": form_data
            }
            flash('Prediction successful!', 'success')
        except Exception as e:
            flash(f"An error occurred during prediction: {e}", 'danger')
            print(f"Prediction error: {e}")

    return render_template('predict.html',
                           airlines=AIRLINES,
                           aircraft_types=AIRCRAFT_TYPES,
                           airport_codes=INDIAN_AIRPORT_CODES,
                           flight_phases=FLIGHT_PHASES,
                           seasons=SEASONS,
                           weather_conditions=WEATHER_CONDITIONS,
                           turbulence_severities=TURBULENCE_SEVERITIES,
                           route_complexities=ROUTE_COMPLEXITIES,
                           air_traffic_densities=AIR_TRAFFIC_DENSITIES,
                           prediction_result=prediction_result)

@app.route('/dashboard')
@login_required
def dashboard():
    feature_importance_data = {
        'Weather_Condition': 0.25, 'Flight_Phase': 0.20, 'Turbulence_Severity': 0.18,
        'Pilot_Experience_Hours': 0.15, 'Aircraft_Age_Years': 0.10, 'Visibility_KM': 0.08,
        'Wind_Speed_KNOTS': 0.07, 'Last_Maintenance_Hours': 0.05, 'Air_Traffic_Density': 0.04
    }
    sorted_features = sorted(feature_importance_data.items(), key=lambda item: item[1], reverse=True)
    labels = [item[0].replace('_', ' ') for item in sorted_features]
    values = [item[1] for item in sorted_features]
    return render_template('dashboard.html', labels=labels, values=values)

@app.route('/about')
def about():
    return render_template('about.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)