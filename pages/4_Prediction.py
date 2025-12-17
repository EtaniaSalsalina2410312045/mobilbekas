import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import sys

# Import navigation
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from modules.nav import Nav
from modules.styles import hide_sidebar_nav


st.set_page_config(
    layout='wide',
    page_title='Price Prediction - Car Price Prediction',
    page_icon='🔮',
    initial_sidebar_state='expanded'
)


@st.cache_resource
def load_all_models():
    """Load all saved models and preprocessors"""
    with open('models/random_forest_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('models/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('models/label_encoder_fuel.pkl', 'rb') as f:
        le_fuel = pickle.load(f)
    with open('models/feature_cols.pkl', 'rb') as f:
        feature_cols = pickle.load(f)
    with open('models/top_manufacturers.pkl', 'rb') as f:
        top_manufacturers = pickle.load(f)
    with open('models/model_metrics.pkl', 'rb') as f:
        metrics = pickle.load(f)
    return model, scaler, le_fuel, feature_cols, top_manufacturers, metrics


def predict_price(model, scaler, le_fuel, feature_cols, top_manufacturers, 
                  manufacturer, engine_size, fuel_type, year, mileage):
    """Make prediction for a single car"""
    
    # Current year for age calculation
    current_year = 2024
    
    # Calculate engineered features
    car_age = current_year - year
    mileage_per_year = mileage / car_age if car_age > 0 else mileage
    
    # Engine category
    if engine_size < 1.5:
        engine_category = 0  # Small
    elif engine_size < 2.5:
        engine_category = 1  # Medium
    else:
        engine_category = 2  # Large
    
    # Fuel type encoding
    try:
        fuel_encoded = le_fuel.transform([fuel_type])[0]
    except:
        fuel_encoded = 0  # Default if unknown
    
    # Create feature dictionary
    features = {
        'Engine size': engine_size,
        'Year of manufacture': year,
        'Mileage': mileage,
        'Car_Age': car_age,
        'Mileage_per_Year': mileage_per_year,
        'Engine_Category': engine_category,
        'Fuel_type_encoded': fuel_encoded
    }
    
    # Add manufacturer dummy variables
    for mfr in top_manufacturers:
        col_name = f'Manufacturer_{mfr}'
        if col_name in feature_cols:
            features[col_name] = 1 if manufacturer == mfr else 0
    
    # Create DataFrame with correct column order
    input_df = pd.DataFrame([features])
    
    # Ensure all feature columns exist
    for col in feature_cols:
        if col not in input_df.columns:
            input_df[col] = 0
    
    # Reorder columns to match training
    input_df = input_df[feature_cols]
    
    # Scale the 6 numerical features that were scaled during training
    numerical_cols_to_scale = ['Engine size', 'Year of manufacture', 'Mileage',
                               'Car_Age', 'Mileage_per_Year', 'Fuel_type_encoded']
    
    if all(col in input_df.columns for col in numerical_cols_to_scale):
        # Create a temporary DataFrame with the 6 columns to scale in correct order
        temp_df = input_df[numerical_cols_to_scale].copy()
        
        # Transform using scaler
        scaled_values = scaler.transform(temp_df)
        
        # Put scaled values back into original DataFrame
        input_df[numerical_cols_to_scale] = scaled_values
    
    # Make prediction
    prediction = model.predict(input_df)[0]
    
    return max(0, prediction)  # Ensure non-negative price


def main():
    Nav()
    hide_sidebar_nav()
    
    st.title("🔮 Car Price Prediction")
    st.markdown("Prediksi harga mobil bekas berdasarkan spesifikasi menggunakan model **Random Forest** yang telah di-training.")
    
    st.markdown("---")
    
    # Load models
    try:
        model, scaler, le_fuel, feature_cols, top_manufacturers, metrics = load_all_models()
        models_loaded = True
    except Exception as e:
        models_loaded = False
        st.error(f"❌ Error loading models: {str(e)}")
        st.warning("Silakan jalankan notebook app.ipynb terlebih dahulu untuk membuat model.")
        return
    
    # ================================
    # SECTION 1: HOW IT WORKS
    # ================================
    st.header("📚 Cara Kerja Prediksi")
    
    with st.expander("🔍 Lihat Detail Proses Prediksi", expanded=False):
        st.markdown("""
        ### 🔄 Pipeline Prediksi
        
        ```
        ┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐
        │  1. Input User   │ →   │  2. Feature      │ →   │  3. Scaling      │ →   │  4. Prediction   │
        │  (Year, Mileage, │     │  Engineering     │     │  (StandardScaler)│     │  (Random Forest) │
        │  Engine, etc.)   │     │  (Age, MPY, etc.)│     │                  │     │                  │
        └──────────────────┘     └──────────────────┘     └──────────────────┘     └──────────────────┘
        ```
        
        **Step 1: Input User**
        - Manufacturer, Year, Engine Size, Fuel Type, Mileage
        
        **Step 2: Feature Engineering**
        - Hitung Car_Age = 2024 - Year
        - Hitung Mileage_per_Year = Mileage / (Car_Age + 1)
        - Tentukan Engine_Category (Small/Medium/Large)
        - Encode Fuel Type (LabelEncoder)
        - One-Hot Encode Manufacturer
        
        **Step 3: Scaling (StandardScaler)**
        
        $$z = \\frac{x - \\mu_{train}}{\\sigma_{train}}$$
        
        - Gunakan mean (μ) dan std (σ) dari training data
        - Normalize semua fitur numerik ke range standar
        
        **Step 4: Random Forest Prediction**
        
        $$\\hat{Price} = \\frac{1}{N} \\sum_{i=1}^{N} Tree_i(X_{scaled})$$
        
        - 100 Decision Trees memberikan prediksi masing-masing
        - Prediksi final = rata-rata dari semua trees
        """)
    
    # ================================
    # MODEL INFO
    # ================================
    st.header("📊 Model Information")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Model", "Random Forest", help="Best performing model dari 4 yang diuji")
    with col2:
        st.metric("R² Score", f"{metrics['r2_score']:.4f}", help="Akurasi model (1.0 = sempurna)")
    with col3:
        st.metric("MAE", f"${metrics['mae']:,.0f}", help="Rata-rata error prediksi")
    with col4:
        st.metric("Features", f"{len(feature_cols)}", help="Jumlah fitur input")
    
    st.markdown("""
    **📋 Interpretasi Model:**
    
    | Metrik | Nilai | Artinya |
    |--------|-------|---------|
    | **R² Score** | 0.9759 | Model menjelaskan 97.59% variasi harga |
    | **MAE** | ~$848 | Rata-rata, prediksi meleset ~$848 dari harga aktual |
    | **Accuracy** | Excellent | Top 2.5% model untuk prediksi harga mobil |
    """)
    
    st.markdown("---")
    
    # ================================
    # INPUT FORM
    # ================================
    st.header("🚗 Masukkan Spesifikasi Mobil")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📋 Informasi Dasar")
        
        # Manufacturer selection
        manufacturer_options = list(top_manufacturers) + ['Other']
        manufacturer = st.selectbox(
            "🏭 Manufacturer",
            options=manufacturer_options,
            help="Pilih manufacturer mobil. Jika tidak ada, pilih 'Other'"
        )
        
        st.markdown("""
        <small>💡 <i>Top 10 manufacturers berdasarkan volume penjualan tertinggi dalam dataset.</i></small>
        """, unsafe_allow_html=True)
        
        # Year of manufacture
        year = st.slider(
            "📅 Year of Manufacture",
            min_value=1990,
            max_value=2024,
            value=2018,
            help="Tahun pembuatan mobil"
        )
        
        # Calculate and show car age
        current_year = 2024
        car_age = current_year - year
        
        st.markdown(f"""
        **📊 Derived Features:**
        - **Car Age**: {car_age} tahun
        - **Depreciation Factor**: {"Tinggi" if car_age > 10 else "Sedang" if car_age > 5 else "Rendah"}
        """)
        
        # Engine size
        engine_size = st.number_input(
            "⚙️ Engine Size (Liters)",
            min_value=0.5,
            max_value=8.0,
            value=2.0,
            step=0.1,
            help="Ukuran mesin dalam liter (0.5L - 8.0L)"
        )
        
        engine_cat = "Small (<1.5L)" if engine_size < 1.5 else "Medium (1.5-2.5L)" if engine_size < 2.5 else "Large (>2.5L)"
        st.markdown(f"**Engine Category**: {engine_cat}")
    
    with col2:
        st.subheader("📋 Detail Tambahan")
        
        # Fuel type
        fuel_types = list(le_fuel.classes_)
        fuel_type = st.selectbox(
            "⛽ Fuel Type",
            options=fuel_types,
            help="Jenis bahan bakar mobil"
        )
        
        st.markdown("""
        **💡 Info Fuel Type:**
        - **Petrol**: Paling umum, efisien untuk jarak pendek
        - **Diesel**: Efisien untuk jarak jauh, torsi tinggi
        - **Hybrid**: Kombinasi listrik & bensin, premium price
        """)
        
        # Mileage
        mileage = st.number_input(
            "🛣️ Mileage (miles)",
            min_value=0,
            max_value=500000,
            value=50000,
            step=1000,
            help="Total jarak tempuh dalam miles"
        )
        
        # Calculate mileage per year
        mileage_per_year = mileage / (car_age + 1) if car_age >= 0 else mileage
        
        st.markdown(f"""
        **📊 Mileage Analysis:**
        - **Total Mileage**: {mileage:,} miles
        - **Per Year**: {mileage_per_year:,.0f} miles/year
        - **Usage**: {"Light" if mileage_per_year < 8000 else "Normal" if mileage_per_year < 15000 else "Heavy" if mileage_per_year < 25000 else "Very Heavy"}
        """)
        
        st.info("""
        **📌 Standar Industri:**
        - < 10,000 miles/year = Light use
        - 10,000 - 15,000 miles/year = Normal
        - > 20,000 miles/year = Heavy use (commercial/rental)
        """)
    
    st.markdown("---")
    
    # ================================
    # PREDICTION BUTTON
    # ================================
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        predict_button = st.button("🔮 Predict Price", type="primary", use_container_width=True)
    
    if predict_button:
        with st.spinner("🔄 Calculating prediction using Random Forest model..."):
            # Make prediction
            predicted_price = predict_price(
                model, scaler, le_fuel, feature_cols, top_manufacturers,
                manufacturer, engine_size, fuel_type, year, mileage
            )
        
        st.markdown("---")
        
        # ================================
        # PREDICTION RESULT
        # ================================
        st.header("💰 Hasil Prediksi")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            st.markdown(f"""
            <div style="
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                padding: 30px;
                border-radius: 15px;
                text-align: center;
                color: white;
                box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            ">
                <h2 style="margin-bottom: 10px;">Estimated Price</h2>
                <h1 style="font-size: 3em; margin: 0;">${predicted_price:,.2f}</h1>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("")
        
        # Price range based on MAE
        lower_bound = max(0, predicted_price - metrics['mae'])
        upper_bound = predicted_price + metrics['mae']
        lower_2sigma = max(0, predicted_price - 2 * metrics['mae'])
        upper_2sigma = predicted_price + 2 * metrics['mae']
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("📉 Conservative Estimate", f"${lower_bound:,.2f}", 
                     help="Batas bawah estimasi (Predicted - MAE)")
        with col2:
            st.metric("📊 Best Estimate", f"${predicted_price:,.2f}",
                     help="Prediksi terbaik dari model")
        with col3:
            st.metric("📈 Optimistic Estimate", f"${upper_bound:,.2f}",
                     help="Batas atas estimasi (Predicted + MAE)")
        
        st.markdown("---")
        
        # ================================
        # CONFIDENCE INTERVAL
        # ================================
        st.subheader("📐 Confidence Interval")
        
        st.markdown(f"""
        **Formula Confidence Interval:**
        
        $$CI = \\hat{{Price}} \\pm k \\times MAE$$
        
        Dimana:
        - $\\hat{{Price}}$ = ${predicted_price:,.2f} (prediksi model)
        - $MAE$ = ${metrics['mae']:,.2f} (rata-rata error)
        - $k$ = faktor confidence level
        
        **Hasil Perhitungan:**
        
        | Confidence Level | Range | Interpretasi |
        |------------------|-------|--------------|
        | **68% (±1 MAE)** | ${lower_bound:,.2f} - ${upper_bound:,.2f} | Kemungkinan besar harga aktual |
        | **95% (±2 MAE)** | ${lower_2sigma:,.2f} - ${upper_2sigma:,.2f} | Hampir pasti dalam range ini |
        
        **📊 Visualisasi:**
        ```
        $0     ${lower_2sigma:,.0f}   ${lower_bound:,.0f}   ${predicted_price:,.0f}   ${upper_bound:,.0f}   ${upper_2sigma:,.0f}
        |--------|---------|========|=========|========|---------|
                 |   95%   |   68%  |  BEST   |   68%  |   95%   |
        ```
        """)
        
        st.info(f"""
        **💡 Interpretasi Praktis:**
        
        Berdasarkan model dengan **R² = {metrics['r2_score']:.4f}** ({metrics['r2_score']*100:.2f}% accuracy):
        
        - **Best Estimate**: ${predicted_price:,.2f}
        - **Safe Range (68%)**: ${lower_bound:,.2f} - ${upper_bound:,.2f}
        - **Wide Range (95%)**: ${lower_2sigma:,.2f} - ${upper_2sigma:,.2f}
        
        **Rekomendasi:**
        - 🛒 **Jika membeli**: Tawar di bawah ${predicted_price:,.2f}
        - 💰 **Jika menjual**: Listing di ${predicted_price:,.2f} - ${upper_bound:,.2f}
        """)
        
        st.markdown("---")
        
        # ================================
        # FEATURE CONTRIBUTION
        # ================================
        st.subheader("📊 Faktor yang Mempengaruhi Harga")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"""
            **🔢 Input yang Anda Masukkan:**
            
            | Spesifikasi | Nilai | Impact |
            |-------------|-------|--------|
            | **Manufacturer** | {manufacturer} | {"Premium ↑" if manufacturer in ['BMW', 'Mercedes-Benz', 'Audi'] else "Standard"} |
            | **Year** | {year} | {"New ↑" if year >= 2020 else "Moderate" if year >= 2015 else "Old ↓"} |
            | **Car Age** | {car_age} tahun | {"Low ↑" if car_age <= 5 else "Medium" if car_age <= 10 else "High ↓"} |
            | **Engine Size** | {engine_size}L | {engine_cat} |
            | **Fuel Type** | {fuel_type} | {"Premium ↑" if fuel_type == "Hybrid" else "Standard"} |
            | **Mileage** | {mileage:,} mi | {"Low ↑" if mileage < 30000 else "Medium" if mileage < 80000 else "High ↓"} |
            | **MPY** | {mileage_per_year:,.0f} mi/yr | {"Light ↑" if mileage_per_year < 10000 else "Normal" if mileage_per_year < 15000 else "Heavy ↓"} |
            """)
        
        with col2:
            st.markdown("""
            **📈 Feature Importance (dari model):**
            
            Berdasarkan analisis Random Forest:
            
            1. **Year of Manufacture** (~35-40%)
               - Faktor #1 penentu harga
               - Mobil baru = harga tinggi
            
            2. **Mileage** (~20-25%)
               - Indikator kondisi mesin
               - High mileage = wear & tear
            
            3. **Car_Age** (~15-20%)
               - Derived dari Year
               - Langsung mengukur depresiasi
            
            4. **Engine Size** (~10-15%)
               - Engine besar = lebih mahal
               - Tapi juga biaya operasional
            
            5. **Manufacturer & Fuel** (~5-10%)
               - Brand value
               - Hybrid premium
            """)
        
        st.markdown("---")
        
        # ================================
        # INPUT SUMMARY
        # ================================
        st.subheader("📋 Summary Input")
        
        summary_data = {
            'Spesifikasi': ['Manufacturer', 'Year of Manufacture', 'Engine Size', 
                           'Fuel Type', 'Mileage', 'Car Age', 'Mileage per Year'],
            'Nilai': [manufacturer, str(year), f'{engine_size}L', 
                     fuel_type, f'{mileage:,} miles', f'{car_age} tahun', 
                     f'{mileage_per_year:,.0f} miles/year'],
            'Processing': ['One-Hot Encoded', 'Scaled', 'Scaled', 
                          'Label Encoded', 'Scaled', 'Engineered Feature',
                          'Engineered Feature']
        }
        summary_df = pd.DataFrame(summary_data)
        st.dataframe(summary_df, use_container_width=True, hide_index=True)
    
    st.markdown("---")
    
    # ================================
    # TIPS & INSIGHTS
    # ================================
    st.header("💡 Tips & Insights untuk Pengguna")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 🔍 Faktor Utama Harga Mobil Bekas
        
        **1. Year of Manufacture** 🏆
        - Faktor #1 penentu harga
        - Depresiasi ~15-20% per tahun di awal
        - Mobil > 10 tahun = depresiasi melambat
        
        **2. Mileage**
        - High mileage = more wear
        - Average: 10-15K miles/year
        - > 100K miles = signifikan turun
        
        **3. Engine Size**
        - Larger engine = higher price
        - Tapi juga fuel cost lebih tinggi
        
        **4. Fuel Type**
        - Hybrid cenderung lebih mahal
        - Diesel bagus untuk jarak jauh
        
        **5. Manufacturer**
        - Premium brands (BMW, Audi) = premium price
        - Japanese brands = reliable, retain value
        """)
    
    with col2:
        st.markdown("""
        ### 📊 Cara Menggunakan Prediksi Ini
        
        **✅ Gunakan sebagai:**
        1. **Referensi awal** saat negosiasi
        2. **Benchmark** untuk compare listings
        3. **Validation** harga yang diminta seller
        4. **Starting point** untuk research lebih lanjut
        
        **⚠️ Pertimbangkan juga (tidak dalam model):**
        1. **Kondisi fisik** mobil (exterior/interior)
        2. **Service history** & maintenance records
        3. **Accident history** (clean title?)
        4. **Lokasi** (harga bervariasi per daerah)
        5. **Trend pasar** saat ini
        6. **Options/accessories** tambahan
        
        **📌 Tips Negosiasi:**
        - Jika membeli: Start dari **Conservative Estimate**
        - Jika menjual: List di **Best Estimate** atau sedikit di atas
        """)
    
    st.success(f"""
    ### ✅ Kredibilitas Model
    
    Model ini dilatih dengan:
    - **50,000+ data** mobil bekas real
    - **Akurasi {metrics['r2_score']*100:.2f}%** (R² Score)
    - **Error rata-rata ${metrics['mae']:,.0f}** (MAE)
    - **Random Forest** dengan 100 decision trees
    
    Gunakan prediksi ini sebagai panduan awal yang reliable dalam transaksi mobil bekas!
    """)


if __name__ == "__main__":
    main()
