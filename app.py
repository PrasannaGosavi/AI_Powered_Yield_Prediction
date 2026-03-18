import pickle
import numpy as np
from flask import Flask, request, render_template, jsonify

app = Flask(__name__)

# ════════════════════════════════════════════════════════════
#  LOAD MODELS (trained separately)
# ════════════════════════════════════════════════════════════
with open('yield_model.pkl', 'rb') as f:
    yield_model = pickle.load(f)

with open('crop_recommendation_model.pkl', 'rb') as f:
    crop_model = pickle.load(f)


# ════════════════════════════════════════════════════════════
#  YIELD MODEL — options
#  Features: Farm_ID | Crop_Type | Farm_Area(acres) |
#            Irrigation_Type | Fertilizer_Used(tons) |
#            Pesticide_Used(kg) | Soil_Type | Season |
#            Water_Usage(cubic meters)   →  predicts: Yield(tons)
# ════════════════════════════════════════════════════════════

YIELD_CROP_TYPES = [
    'Wheat', 'Rice', 'Maize', 'Sugarcane', 'Cotton',
    'Soybean', 'Groundnut', 'Barley', 'Millet', 'Sorghum',
    'Potato', 'Tomato', 'Onion', 'Chickpea', 'Lentil',
    'Sunflower', 'Mustard', 'Jute', 'Coconut', 'Banana'
]

IRRIGATION_TYPES = ['Drip', 'Sprinkler', 'Flood', 'Furrow', 'Rainfed']

SOIL_TYPES = [
    'Loamy', 'Sandy', 'Clay', 'Silty', 'Peaty',
    'Chalky', 'Sandy Loam', 'Clay Loam', 'Black', 'Red'
]

YIELD_SEASONS = ['Kharif', 'Rabi', 'Zaid', 'Summer', 'Winter', 'Whole Year']


# ════════════════════════════════════════════════════════════
#  CROP RECOMMENDATION MODEL — options
#  Features: State_Name | District_Name | Crop_Year | Season |
#            Crop | Temperature | Humidity | Soil_Moisture |
#            Area | Production
# ════════════════════════════════════════════════════════════

STATES = [
    'Andhra Pradesh', 'Assam', 'Bihar', 'Chhattisgarh', 'Delhi',
    'Goa', 'Gujarat', 'Haryana', 'Himachal Pradesh', 'Jammu and Kashmir',
    'Jharkhand', 'Karnataka', 'Kerala', 'Madhya Pradesh', 'Maharashtra',
    'Manipur', 'Meghalaya', 'Mizoram', 'Nagaland', 'Odisha',
    'Puducherry', 'Punjab', 'Rajasthan', 'Sikkim', 'Tamil Nadu',
    'Telangana', 'Tripura', 'Uttar Pradesh', 'Uttarakhand', 'West Bengal'
]

DISTRICTS = [
    'Ahmednagar', 'Akola', 'Amravati', 'Aurangabad', 'Beed',
    'Buldhana', 'Chandrapur', 'Dhule', 'Gadchiroli', 'Gondia',
    'Hingoli', 'Jalgaon', 'Jalna', 'Kolhapur', 'Latur',
    'Nagpur', 'Nanded', 'Nashik', 'Osmanabad', 'Parbhani',
    'Pune', 'Raigad', 'Ratnagiri', 'Sangli', 'Satara',
    'Solapur', 'Thane', 'Wardha', 'Washim', 'Yavatmal'
]

CROP_SEASONS = ['Kharif', 'Rabi', 'Zaid', 'Whole Year', 'Summer', 'Winter']

CROPS = [
    'Arecanut', 'Arhar/Tur', 'Bajra', 'Banana', 'Barley',
    'Black pepper', 'Cardamom', 'Cashewnut', 'Castor seed', 'Coconut',
    'Coffee', 'Cotton(lint)', 'Cowpea(Lobia)', 'Dry chillies', 'Garlic',
    'Ginger', 'Gram', 'Grapes', 'Groundnut', 'Guar seed',
    'Horse-gram', 'Jowar', 'Jute', 'Khesari', 'Linseed',
    'Maize', 'Masoor', 'Mesta', 'Moong(Green Gram)', 'Moth',
    'Niger seed', 'Onion', 'Peas & beans (Pulses)', 'Potato',
    'Ragi', 'Rapeseed &Mustard', 'Rice', 'Safflower', 'Sesamum',
    'Small millets', 'Soyabean', 'Sugarcane', 'Sunflower',
    'Sweet potato', 'Tapioca', 'Tobacco', 'Turmeric', 'Urad', 'Wheat'
]

CROP_CLASSES = CROPS


# ════════════════════════════════════════════════════════════
#  HELPERS
# ════════════════════════════════════════════════════════════

def encode(value, options):
    """Integer-encode a categorical. Returns 0 if not found."""
    try:
        return options.index(str(value).strip())
    except ValueError:
        return 0


# ════════════════════════════════════════════════════════════
#  ROUTES
# ════════════════════════════════════════════════════════════

@app.route('/')
def home():
    return render_template('home.html')


# ── YIELD PREDICTION ────────────────────────────────────────

@app.route('/yield')
def yield_page():
    return render_template(
        'yield.html',
        crop_types=YIELD_CROP_TYPES,
        irrigation_types=IRRIGATION_TYPES,
        soil_types=SOIL_TYPES,
        seasons=YIELD_SEASONS
    )


@app.route('/predict_yield', methods=['POST'])
def predict_yield():
    ctx = dict(
        crop_types=YIELD_CROP_TYPES,
        irrigation_types=IRRIGATION_TYPES,
        soil_types=SOIL_TYPES,
        seasons=YIELD_SEASONS,
        form_data=request.form
    )
    try:
        farm_id         = request.form.get('farm_id', '').strip()
        crop_type       = request.form.get('crop_type', '').strip()
        farm_area       = float(request.form.get('farm_area', 0))
        irrigation_type = request.form.get('irrigation_type', '').strip()
        fertilizer_used = float(request.form.get('fertilizer_used', 0))
        pesticide_used  = float(request.form.get('pesticide_used', 0))
        soil_type       = request.form.get('soil_type', '').strip()
        season          = request.form.get('season', '').strip()
        water_usage     = float(request.form.get('water_usage', 0))

        crop_type_enc   = encode(crop_type,       YIELD_CROP_TYPES)
        irrigation_enc  = encode(irrigation_type, IRRIGATION_TYPES)
        soil_enc        = encode(soil_type,       SOIL_TYPES)
        season_enc      = encode(season,          YIELD_SEASONS)

        features = np.array([[
            crop_type_enc,
            farm_area,
            irrigation_enc,
            fertilizer_used,
            pesticide_used,
            soil_enc,
            season_enc,
            water_usage
        ]])

        result = round(float(yield_model.predict(features)[0]), 2)
        ctx['prediction'] = f"{result:,.2f} tons"

    except ValueError as e:
        ctx['error'] = f"Invalid input value — {e}"
    except Exception as e:
        ctx['error'] = f"Prediction failed — {e}"

    return render_template('yield.html', **ctx)


# ── CROP RECOMMENDATION ──────────────────────────────────────

@app.route('/crop')
def crop_page():
    return render_template(
        'crop.html',
        states=STATES,
        districts=DISTRICTS,
        seasons=CROP_SEASONS,
        crops=CROPS
    )


@app.route('/predict_crop', methods=['POST'])
def predict_crop():
    ctx = dict(
        states=STATES,
        districts=DISTRICTS,
        seasons=CROP_SEASONS,
        crops=CROPS,
        form_data=request.form
    )
    try:
        state_name    = request.form.get('state_name',    '').strip()
        district_name = request.form.get('district_name', '').strip()
        crop_year     = float(request.form.get('crop_year',     0))
        season        = request.form.get('season',         '').strip()
        crop          = request.form.get('crop',           '').strip()
        temperature   = float(request.form.get('temperature',   0))
        humidity      = float(request.form.get('humidity',      0))
        soil_moisture = float(request.form.get('soil_moisture', 0))
        area          = float(request.form.get('area',          0))
        production    = float(request.form.get('production',    0))

        state_enc    = encode(state_name,    STATES)
        district_enc = encode(district_name, DISTRICTS)
        season_enc   = encode(season,        CROP_SEASONS)
        crop_enc     = encode(crop,          CROPS)

        features = np.array([[
            state_enc,
            district_enc,
            crop_year,
            season_enc,
            crop_enc,
            temperature,
            humidity,
            soil_moisture,
            area,
            production
        ]])

        prediction_raw = crop_model.predict(features)[0]

        if isinstance(prediction_raw, (int, np.integer)):
            idx       = int(prediction_raw)
            crop_name = CROP_CLASSES[idx] if idx < len(CROP_CLASSES) else f"Crop #{idx}"
        else:
            crop_name = str(prediction_raw).capitalize()

        ctx['prediction'] = crop_name.capitalize()

        if hasattr(crop_model, 'predict_proba'):
            proba       = crop_model.predict_proba(features)[0]
            top_indices = np.argsort(proba)[::-1][:3]
            ctx['top_crops'] = [
                {
                    'name':       (CROP_CLASSES[i] if i < len(CROP_CLASSES) else f"Crop {i}").capitalize(),
                    'confidence': round(float(proba[i]) * 100, 1)
                }
                for i in top_indices
            ]

    except ValueError as e:
        ctx['error'] = f"Invalid input value — {e}"
    except Exception as e:
        ctx['error'] = f"Prediction failed — {e}"

    return render_template('crop.html', **ctx)


# ════════════════════════════════════════════════════════════
#  JSON API
# ════════════════════════════════════════════════════════════

@app.route('/api/predict_yield', methods=['POST'])
def api_predict_yield():
    try:
        d = request.get_json(force=True)
        features = np.array([[
            encode(d.get('crop_type',''),       YIELD_CROP_TYPES),
            float(d.get('farm_area', 0)),
            encode(d.get('irrigation_type',''), IRRIGATION_TYPES),
            float(d.get('fertilizer_used', 0)),
            float(d.get('pesticide_used', 0)),
            encode(d.get('soil_type',''),       SOIL_TYPES),
            encode(d.get('season',''),          YIELD_SEASONS),
            float(d.get('water_usage', 0)),
        ]])
        result = round(float(yield_model.predict(features)[0]), 2)
        return jsonify({'success': True, 'prediction': result, 'unit': 'tons'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400


@app.route('/api/predict_crop', methods=['POST'])
def api_predict_crop():
    try:
        d = request.get_json(force=True)
        features = np.array([[
            encode(d.get('state_name',''),    STATES),
            encode(d.get('district_name',''), DISTRICTS),
            float(d.get('crop_year', 0)),
            encode(d.get('season',''),        CROP_SEASONS),
            encode(d.get('crop',''),          CROPS),
            float(d.get('temperature', 0)),
            float(d.get('humidity', 0)),
            float(d.get('soil_moisture', 0)),
            float(d.get('area', 0)),
            float(d.get('production', 0)),
        ]])
        raw  = crop_model.predict(features)[0]
        name = CROP_CLASSES[int(raw)].capitalize() if isinstance(raw, (int, np.integer)) else str(raw).capitalize()
        return jsonify({'success': True, 'recommended_crop': name})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400


# ════════════════════════════════════════════════════════════
if __name__ == '__main__':
    app.run(debug=True)
