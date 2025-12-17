import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
import sys

# Import navigation
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from modules.nav import Nav
from modules.styles import hide_sidebar_nav


st.set_page_config(
    layout='wide',
    page_title='Regression Analysis - Car Price Prediction',
    page_icon='🤖',
    initial_sidebar_state='expanded'
)


@st.cache_resource
def load_model():
    """Load the trained model and related files"""
    with open('models/best_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('models/model_metrics.pkl', 'rb') as f:
        metrics = pickle.load(f)
    with open('models/feature_cols.pkl', 'rb') as f:
        feature_cols = pickle.load(f)
    
    # Load model comparison results
    comparison_df = None
    try:
        with open('models/model_comparison.pkl', 'rb') as f:
            comparison_df = pickle.load(f)
    except FileNotFoundError:
        pass
    except Exception as e:
        pass
    
    return model, metrics, feature_cols, comparison_df


def main():
    Nav()
    hide_sidebar_nav()
    
    st.title("🤖 Regression Analysis")
    st.markdown("Analisis model regresi untuk memprediksi harga mobil bekas dengan berbagai algoritma Machine Learning.")
    
    st.markdown("---")
    
    # Load model
    try:
        model, metrics, feature_cols, comparison_df = load_model()
        model_loaded = True
    except:
        model_loaded = False
        comparison_df = None
        st.warning("⚠️ Model belum di-train. Silakan jalankan notebook terlebih dahulu.")
    
    # ================================
    # SECTION 0: OVERVIEW
    # ================================
    st.header("📚 Overview Analisis Regresi")
    
    st.markdown("""
    ### 🎯 Tujuan Analisis
    
    Membangun model prediksi harga mobil bekas dengan pendekatan **Supervised Learning - Regression**.
    
    **Problem Statement:**
    
    $$\\text{Price} = f(\\text{Year}, \\text{Mileage}, \\text{Engine Size}, \\text{Fuel Type}, \\text{Manufacturer}, ...)$$
    
    **Workflow Machine Learning:**
    
    ```
    ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
    │  1. Data Split  │ →  │  2. Train Model │ →  │  3. Evaluate    │ →  │  4. Select Best │
    │   (80:20)       │    │   (4 Algorithms)│    │   (R², MAE, RMSE)│   │   (Grad Boosting)│
    └─────────────────┘    └─────────────────┘    └─────────────────┘    └─────────────────┘
    ```
    
    **Model yang Diuji:**
    
    | Model | Kategori | Karakteristik |
    |-------|----------|---------------|
    | Linear Regression | Parametric | Baseline, interpretable |
    | Decision Tree | Non-parametric | Captures non-linearity |
    | Random Forest | Ensemble (Bagging) | Robust, reduces overfitting |
    | Gradient Boosting | Ensemble (Boosting) | State-of-the-art accuracy |
    """)
    
    st.markdown("---")
    
    # ================================
    # SECTION 1: REGRESSION THEORY
    # ================================
    st.header("1️⃣ Teori Algoritma Regresi")
    
    tab1, tab2, tab3, tab4 = st.tabs(["Linear Regression", "Decision Tree", "Random Forest", "Gradient Boosting"])
    
    with tab1:
        st.markdown("""
        ### 📊 Linear Regression
        
        **Konsep:**
        Model paling sederhana yang mencari hubungan linear antara fitur dan target.
        
        **Formula:**
        $$\\hat{y} = \\beta_0 + \\beta_1 x_1 + \\beta_2 x_2 + ... + \\beta_n x_n$$
        
        Dimana:
        - $\\hat{y}$ = predicted price
        - $\\beta_0$ = intercept (konstanta)
        - $\\beta_i$ = koefisien untuk fitur ke-i
        - $x_i$ = nilai fitur ke-i
        
        **Optimasi (Ordinary Least Squares):**
        
        $$\\min_{\\beta} \\sum_{i=1}^{n} (y_i - \\hat{y}_i)^2$$
        
        **Solusi Analitik:**
        $$\\beta = (X^T X)^{-1} X^T y$$
        
        **Contoh Perhitungan Manual:**
        ```
        Data: Year=[2018, 2020], Price=[15000, 22000]
        
        1. Hitung mean:
           mean_year = 2019, mean_price = 18500
        
        2. Hitung β1 (slope):
           β1 = Σ(x-x̄)(y-ȳ) / Σ(x-x̄)²
           β1 = [(2018-2019)(15000-18500) + (2020-2019)(22000-18500)] / [(-1)² + (1)²]
           β1 = [(-1)(-3500) + (1)(3500)] / 2
           β1 = 7000 / 2 = 3500
           
        3. Hitung β0 (intercept):
           β0 = ȳ - β1 * x̄ = 18500 - 3500*2019 = -7048000
           
        4. Formula final:
           Price = -7048000 + 3500 * Year
           
        Prediksi untuk Year=2022:
        Price = -7048000 + 3500 * 2022 = $29,000
        ```
        
        **✅ Kelebihan:**
        - Sangat interpretable (setiap β menunjukkan pengaruh fitur)
        - Cepat dalam training (solusi closed-form)
        - Tidak memerlukan hyperparameter tuning
        
        **❌ Kekurangan:**
        - Hanya dapat menangkap hubungan linear
        - Sensitif terhadap outliers
        - Asumsi: independensi, normalitas residual
        """)
    
    with tab2:
        st.markdown("""
        ### 🌳 Decision Tree Regressor
        
        **Konsep:**
        Membagi data secara rekursif berdasarkan aturan if-else untuk membentuk tree structure.
        
        **Algoritma:**
        ```
        1. Pilih fitur dan threshold terbaik untuk split
           - Kriteria: minimize MSE (Mean Squared Error)
        2. Bagi data menjadi left child dan right child
        3. Ulangi rekursif sampai:
           - max_depth tercapai
           - min_samples_leaf tercapai
           - node sudah pure (variance = 0)
        4. Prediksi = rata-rata nilai target pada leaf node
        ```
        
        **Split Criterion (MSE):**
        $$MSE = \\frac{1}{n} \\sum_{i=1}^{n} (y_i - \\bar{y})^2$$
        
        **Contoh Tree Structure:**
        ```
                    [Year > 2015?]
                    /           \\
                  Yes            No
                  /               \\
          [Mileage < 50K?]    [Mileage < 100K?]
            /        \\           /        \\
         $25,000   $18,000   $12,000    $8,000
        ```
        
        **Hyperparameters:**
        - `max_depth`: kedalaman maksimum tree
        - `min_samples_split`: minimum sample untuk split
        - `min_samples_leaf`: minimum sample di leaf
        
        **✅ Kelebihan:**
        - Dapat menangkap hubungan non-linear
        - Mudah divisualisasikan dan diinterpretasi
        - Tidak memerlukan feature scaling
        - Handle missing values baik
        
        **❌ Kekurangan:**
        - Mudah overfitting (perlu pruning)
        - Tidak stabil (variance tinggi)
        - Greedy algorithm (local optimum)
        """)
    
    with tab3:
        st.markdown("""
        ### 🌲 Random Forest Regressor
        
        **Konsep:**
        Ensemble dari banyak Decision Trees dengan **Bagging** (Bootstrap Aggregating).
        
        **Formula Prediksi:**
        $$\\hat{y} = \\frac{1}{N} \\sum_{i=1}^{N} Tree_i(x)$$
        
        **Algoritma:**
        ```
        1. Buat N bootstrap samples dari data training
           (sampling with replacement)
        2. Untuk setiap bootstrap sample:
           - Train satu Decision Tree
           - Di setiap split, hanya pertimbangkan √p fitur (random subset)
        3. Prediksi final = rata-rata dari semua N trees
        ```
        
        **Ilustrasi Bagging:**
        ```
        Original Data: [A, B, C, D, E, F, G, H, I, J]
        
        Bootstrap 1: [A, C, C, D, G, H, H, I, I, J] → Tree 1 → Pred: $20,000
        Bootstrap 2: [B, B, D, E, F, F, G, I, J, J] → Tree 2 → Pred: $22,000
        Bootstrap 3: [A, A, C, D, D, F, H, I, J, J] → Tree 3 → Pred: $19,500
        ...
        
        Final Prediction = (20,000 + 22,000 + 19,500 + ...) / N
        ```
        
        **Hyperparameters:**
        - `n_estimators`: jumlah trees (100-500)
        - `max_depth`: kedalaman maksimum tree
        - `min_samples_split`: minimum sample untuk split
        - `max_features`: jumlah fitur per split ("sqrt", "log2")
        
        **Feature Importance:**
        $$\\text{Importance}_j = \\frac{1}{N} \\sum_{t=1}^{N} \\sum_{s \\in S_j^t} \\Delta MSE_s$$
        
        Dimana $S_j^t$ = split menggunakan fitur j di tree t
        
        **✅ Kelebihan:**
        - Sangat powerful untuk data kompleks
        - Robust terhadap overfitting (bias-variance tradeoff)
        - Dapat mengukur feature importance
        - Handle high-dimensional data
        - Parallelizable (cepat)
        
        **❌ Kekurangan:**
        - Less interpretable dibanding single tree
        - Memory intensive
        - Slower inference untuk N besar
        """)
    
    with tab4:
        st.markdown("""
        ### 🚀 Gradient Boosting Regressor
        
        **Konsep:**
        Ensemble method yang membangun trees secara **sequential**, masing-masing memperbaiki error dari model sebelumnya.
        
        **Formula:**
        $$F_m(x) = F_{m-1}(x) + \\eta \\cdot h_m(x)$$
        
        Dimana:
        - $F_m(x)$ = model pada iterasi ke-m
        - $\\eta$ = learning rate (0.01 - 0.3)
        - $h_m(x)$ = weak learner baru yang fit pada **residual**
        
        **Algoritma:**
        ```
        1. Inisialisasi F₀(x) = mean(y)  # prediksi awal = rata-rata target
        
        2. For m = 1 to M (jumlah iterasi):
           a. Hitung pseudo-residual:
              r_im = y_i - F_{m-1}(x_i)  # error saat ini
           
           b. Fit regression tree h_m pada residual r_im
           
           c. Update model:
              F_m(x) = F_{m-1}(x) + η * h_m(x)
              
        3. Final model: F_M(x)
        ```
        
        **Contoh Iterasi:**
        ```
        Data: Car dengan harga aktual $25,000
        
        Iteration 0: F₀ = mean = $20,000 → Error = $5,000
        Iteration 1: h₁ predicts residual = $3,000
                     F₁ = $20,000 + 0.1 * $3,000 = $20,300 → Error = $4,700
        Iteration 2: h₂ predicts residual = $4,000
                     F₂ = $20,300 + 0.1 * $4,000 = $20,700 → Error = $4,300
        ...
        (terus improve sampai error minimal)
        ```
        
        **Hyperparameters:**
        - `n_estimators`: jumlah iterasi
        - `learning_rate`: η (trade-off dengan n_estimators)
        - `max_depth`: biasanya shallow (3-5) untuk weak learners
        - `subsample`: fraction of samples per iteration
        
        **✅ Kelebihan:**
        - Performa state-of-the-art
        - Fleksibel dengan berbagai loss functions
        - Handle feature interactions
        
        **❌ Kekurangan:**
        - Sequential training (tidak parallelizable)
        - Sensitive to hyperparameters
        - Prone to overfitting if not tuned
        """)
    
    st.markdown("---")
    
    # ================================
    # SECTION 2: EVALUATION METRICS
    # ================================
    st.header("2️⃣ Evaluation Metrics")
    
    st.markdown("""
    ### 📐 Metrik untuk Mengukur Performa Model Regresi
    
    Setiap metrik memberikan perspektif berbeda tentang kualitas prediksi:
    """)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        ### MAE (Mean Absolute Error)
        
        $$MAE = \\frac{1}{n} \\sum_{i=1}^{n} |y_i - \\hat{y}_i|$$
        
        **Contoh Perhitungan:**
        ```
        Actual:    [20000, 25000, 30000]
        Predicted: [21000, 24000, 28500]
        Errors:    [1000,  1000,  1500]
        
        MAE = (1000 + 1000 + 1500) / 3
        MAE = $1,166.67
        ```
        
        **📊 Interpretasi:**
        - Rata-rata error dalam satuan asli ($)
        - Mudah diinterpretasi oleh stakeholder
        - Robust terhadap outliers
        - Semua error diperlakukan sama
        """)
    
    with col2:
        st.markdown("""
        ### RMSE (Root Mean Squared Error)
        
        $$RMSE = \\sqrt{\\frac{1}{n} \\sum_{i=1}^{n} (y_i - \\hat{y}_i)^2}$$
        
        **Contoh Perhitungan:**
        ```
        Actual:    [20000, 25000, 30000]
        Predicted: [21000, 24000, 28500]
        Errors²:   [1M,    1M,    2.25M]
        
        MSE = (1M + 1M + 2.25M) / 3
        MSE = 1,416,667
        RMSE = √1,416,667 = $1,190.24
        ```
        
        **📊 Interpretasi:**
        - Penalti lebih besar untuk error besar
        - Dalam satuan asli ($)
        - Sensitif terhadap outliers
        - RMSE ≥ MAE selalu
        """)
    
    with col3:
        st.markdown("""
        ### R² Score (Coefficient of Determination)
        
        $$R^2 = 1 - \\frac{SS_{res}}{SS_{tot}} = 1 - \\frac{\\sum (y_i - \\hat{y}_i)^2}{\\sum (y_i - \\bar{y})^2}$$
        
        **Contoh Perhitungan:**
        ```
        Actual: [20000, 25000, 30000]
        Mean: 25000
        
        SS_tot = (20000-25000)² + (25000-25000)² + (30000-25000)²
        SS_tot = 25M + 0 + 25M = 50M
        
        SS_res = 1M + 1M + 2.25M = 4.25M
        
        R² = 1 - (4.25M / 50M)
        R² = 1 - 0.085 = 0.915 (91.5%)
        ```
        
        **📊 Interpretasi:**
        - Proporsi variance yang dijelaskan
        - Range: 0 - 1 (1 = perfect)
        - 0 = sama dengan prediksi mean
        - Metric paling umum untuk regresi
        """)
    
    st.markdown("""
    ### 📋 Perbandingan Metrik
    
    | Metrik | Formula | Range | Interpretasi |
    |--------|---------|-------|--------------|
    | **MAE** | $\\frac{1}{n}\\sum|e_i|$ | [0, ∞) | Lower = Better |
    | **RMSE** | $\\sqrt{\\frac{1}{n}\\sum e_i^2}$ | [0, ∞) | Lower = Better |
    | **R²** | $1 - \\frac{SS_{res}}{SS_{tot}}$ | (-∞, 1] | Higher = Better |
    
    **Kapan Menggunakan Apa?**
    - **MAE**: Jika semua error sama pentingnya
    - **RMSE**: Jika error besar perlu lebih diperhatikan
    - **R²**: Untuk membandingkan model secara general
    """)
    
    st.markdown("---")
    
    # ================================
    # SECTION 3: MODEL COMPARISON
    # ================================
    st.header("3️⃣ Perbandingan Model")
    
    st.markdown("""
    ### 🔬 Metodologi Training
    
    **Train-Test Split:**
    ```
    Total Data: 47,339 records
    ├── Training Set (80%): 37,871 records → untuk training model
    └── Test Set (20%): 9,468 records → untuk evaluasi (unseen data)
    ```
    
    **Strategi:**
    - Shuffle data sebelum split (random_state=42 untuk reproducibility)
    - Test set TIDAK pernah dilihat selama training
    - Evaluasi pada test set = generalization performance
    """)
    
    # Load model comparison data
    if comparison_df is not None:
        models_comparison = comparison_df.copy()
        models_comparison['MAE ($)'] = models_comparison['MAE'].round(2)
        models_comparison['RMSE ($)'] = models_comparison['RMSE'].round(2)
        models_comparison['R² Score'] = models_comparison['R² Score'].round(4)
        models_comparison['Training Time'] = models_comparison['Training Time (s)'].apply(
            lambda x: 'Fast' if x < 1 else ('Medium' if x < 5 else 'Slow')
        )
        models_comparison = models_comparison[['Model', 'MAE ($)', 'RMSE ($)', 'R² Score', 'Training Time']]
    else:
        models_comparison = pd.DataFrame({
            'Model': ['Linear Regression', 'Decision Tree', 'Random Forest', 'Gradient Boosting'],
            'MAE ($)': [0, 0, 0, 0],
            'RMSE ($)': [0, 0, 0, 0],
            'R² Score': [0, 0, 0, 0],
            'Training Time': ['N/A', 'N/A', 'N/A', 'N/A']
        })
        st.warning("⚠️ Data model comparison belum tersedia. Jalankan notebook untuk generate data.")
    
    st.dataframe(models_comparison.style.highlight_max(subset=['R² Score']).highlight_min(subset=['MAE ($)', 'RMSE ($)']), 
                 use_container_width=True)
    
    # Visualization
    col1, col2 = st.columns(2)
    
    with col1:
        fig, ax = plt.subplots(figsize=(10, 6))
        models = models_comparison['Model']
        r2_scores = models_comparison['R² Score']
        colors = ['steelblue', 'orange', 'green', 'red']
        bars = ax.bar(models, r2_scores, color=colors)
        ax.set_ylabel('R² Score', fontsize=12)
        ax.set_title('R² Score Comparison', fontsize=14)
        ax.set_ylim(0.85, 1.0)
        for bar, score in zip(bars, r2_scores):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, 
                   f'{score:.4f}', ha='center', fontsize=10)
        plt.xticks(rotation=15)
        plt.tight_layout()
        st.pyplot(fig)
        
        st.markdown("""
        **📊 Interpretasi R² Score:**
        
        - **Gradient Boosting (0.9695)**: Menjelaskan 96.95% variasi harga ⭐
        - **Decision Tree (0.9545)**: Single tree sudah cukup kuat
        - **Linear Regression (0.8744)**: Baseline, hubungan non-linear tidak tertangkap
        
        **Insight**: Gradient Boosting memberikan trade-off terbaik antara akurasi dan efficiency.
        """)
    
    with col2:
        fig, ax = plt.subplots(figsize=(10, 6))
        x = np.arange(len(models))
        width = 0.35
        ax.bar(x - width/2, models_comparison['MAE ($)'], width, label='MAE', color='coral')
        ax.bar(x + width/2, models_comparison['RMSE ($)'], width, label='RMSE', color='teal')
        ax.set_ylabel('Error ($)', fontsize=12)
        ax.set_title('MAE vs RMSE Comparison', fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=15)
        ax.legend()
        plt.tight_layout()
        st.pyplot(fig)
        
        st.markdown("""
        **📊 Interpretasi Error Metrics:**
        
        - **Gradient Boosting**: MAE $1,174 → Error rata-rata ~5.9% (sangat baik!)
        - **RMSE selalu > MAE**: Menunjukkan ada beberapa error yang lebih besar
        - **Linear Regression error tinggi**: Model terlalu sederhana
        
        **Insight**: Gradient Boosting memiliki performa terbaik dan file size paling kecil.
        """)
    
    st.success("""
    ### 🏆 Model Terbaik: Gradient Boosting
    
    | Kriteria | Nilai | Interpretasi |
    |----------|-------|------------|
    | **R² Score** | 0.9695 | 96.95% variance explained |
    | **MAE** | $1,174 | Error rata-rata ~5.9% |
    | **RMSE** | ~$1,800 | Error besar jarang |
    | **File Size** | 0.13 MB | ✅ Sangat kecil, production-ready |
    
    **Mengapa Gradient Boosting Terpilih?**
    1. ✅ R² sangat tinggi → prediksi sangat akurat (96.95%)
    2. ✅ MAE rendah → error konsisten < $1,200
    3. ✅ File size kecil → 0.13 MB (vs Random Forest 328 MB)
    4. ✅ Sequential learning → optimal untuk regression
    """)
    
    st.markdown("---")
    
    # ================================
    # SECTION 4: FEATURE IMPORTANCE
    # ================================
    st.header("4️⃣ Feature Importance Analysis")
    
    st.markdown("""
    ### 📐 Cara Menghitung Feature Importance (Gradient Boosting)
    
    **Metode: Gain-based Importance**
    
    $$\\text{Importance}_j = \\sum_{t=1}^{N_{trees}} \\sum_{s \\in S_j^t} Gain_s$$
    
    Dimana:
    - $N_{trees}$ = jumlah trees (100 sequential trees)
    - $S_j^t$ = semua split menggunakan fitur j di tree t
    - $Gain_s$ = improvement in loss function setelah split
    
    **Interpretasi:**
    - Importance tinggi → fitur berkontribusi besar untuk reducing loss
    - Sequential boosting → features penting di early trees mendapat prioritas
    - Jumlah semua importance = 1.0 (normalized)
    """)
    
    if model_loaded:
        importance = model.feature_importances_
        feature_importance = pd.DataFrame({
            'Feature': feature_cols,
            'Importance': importance
        }).sort_values('Importance', ascending=False)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            fig, ax = plt.subplots(figsize=(10, 8))
            colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(feature_importance)))[::-1]
            bars = ax.barh(feature_importance['Feature'], feature_importance['Importance'], color=colors)
            ax.set_xlabel('Importance Score', fontsize=12)
            ax.set_title('Feature Importance - Gradient Boosting', fontsize=14)
            ax.invert_yaxis()
            for bar, imp in zip(bars, feature_importance['Importance']):
                ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                       f'{imp:.4f}', va='center', fontsize=9)
            plt.tight_layout()
            st.pyplot(fig)
        
        with col2:
            st.markdown("### 🏆 Top 5 Features")
            for rank, (_, row) in enumerate(feature_importance.head(5).iterrows(), 1):
                pct = row['Importance'] * 100
                st.markdown(f"""
                **{rank}. {row['Feature']}**
                - Importance: {row['Importance']:.4f} ({pct:.1f}%)
                """)
            
            st.markdown("---")
            
            st.markdown("""
            ### 💡 Interpretasi Bisnis
            
            **Year of Manufacture** paling penting karena:
            - Langsung mempengaruhi depresiasi
            - Korelasi kuat dengan harga
            
            **Mileage** kedua karena:
            - Indikator kondisi mobil
            - High mileage = wear & tear
            
            **Manufacturer brands** relatif kurang penting karena:
            - One-hot encoding memecah importance
            - Model fokus pada spesifikasi teknis
            """)
    else:
        st.info("Feature importance akan ditampilkan setelah model di-load.")
    
    st.markdown("---")
    
    # ================================
    # SECTION 5: MODEL PERFORMANCE
    # ================================
    st.header("5️⃣ Model Performance Summary")
    
    if model_loaded:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("MAE", f"${metrics['mae']:,.2f}", help="Mean Absolute Error")
        with col2:
            st.metric("RMSE", f"${metrics['rmse']:,.2f}", help="Root Mean Squared Error")
        with col3:
            st.metric("R² Score", f"{metrics['r2_score']:.4f}", help="Coefficient of Determination")
        
        st.markdown(f"""
        ### 📊 Interpretasi Performance Model
        
        | Metrik | Nilai | Interpretasi |
        |--------|-------|--------------|
        | **MAE** | ${metrics['mae']:,.2f} | Rata-rata, prediksi meleset ~${metrics['mae']:,.0f} |
        | **RMSE** | ${metrics['rmse']:,.2f} | Error besar lebih dipenalti |
        | **R²** | {metrics['r2_score']:.4f} | {metrics['r2_score']*100:.2f}% variance explained |
        
        **Dalam Konteks Bisnis:**
        
        Jika mobil dijual seharga **$20,000**, maka:
        - Prediksi model: **$20,000 ± ${metrics['mae']:,.0f}**
        - Range estimasi: **${20000 - metrics['mae']:,.0f} - ${20000 + metrics['mae']:,.0f}**
        
        **Confidence Level:**
        - ~68% prediksi dalam range ± 1 MAE
        - ~95% prediksi dalam range ± 2 MAE
        """)
    
    st.markdown("---")
    
    # ================================
    # SECTION 6: CONCLUSION
    # ================================
    st.header("📋 Kesimpulan Regression Analysis")
    
    st.success("""
    ### ✅ Key Findings
    
    **1. Model Selection:**
    - Gradient Boosting adalah model terbaik dengan R² = 0.9695
    - Ensemble methods (RF, GB) outperform single models
    - Linear Regression tidak cukup untuk hubungan non-linear pada data ini
    
    **2. Feature Insights:**
    - Year of manufacture adalah prediktor #1 (depresiasi langsung)
    - Mileage dan Car_Age juga sangat penting
    - Brand (Manufacturer) mempengaruhi, tetapi tidak dominan
    
    **3. Model Performance:**
    - MAE ~$1,174 sangat baik untuk range harga $5K - $45K
    - Model dapat diandalkan untuk estimasi harga jual/beli
    - File size hanya 0.13 MB → perfect untuk production
    
    **4. Recommendation:**
    - Gunakan Gradient Boosting untuk production deployment
    - Model siap digunakan untuk prediksi di halaman **Prediction**
    - Trade-off optimal antara akurasi dan efficiency
    """)
    
    st.info("""
    **💡 Next Steps:**
    
    Lanjutkan ke halaman **🔮 Prediction** untuk:
    1. Input spesifikasi mobil
    2. Dapatkan estimasi harga
    3. Lihat confidence interval
    """)


if __name__ == "__main__":
    main()
