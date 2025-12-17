import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

# Import navigation
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from modules.nav import Nav
from modules.styles import hide_sidebar_nav


st.set_page_config(
    layout='wide',
    page_title='Preprocessing - Car Price Prediction',
    page_icon='',
    initial_sidebar_state='expanded'
)


@st.cache_data
def load_data():
    """Load and cache the dataset"""
    df = pd.read_csv('Second-Hand Cars Data.csv')
    return df


def detect_outliers_iqr(df, column):
    """Deteksi outliers menggunakan IQR method"""
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers_mask = (df[column] < lower_bound) | (df[column] > upper_bound)
    return outliers_mask, lower_bound, upper_bound, Q1, Q3, IQR


def remove_outliers_iqr(df, column):
    """Hapus outliers menggunakan IQR method"""
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]


def main():
    Nav()
    hide_sidebar_nav()
    
    st.title(" Data Preprocessing")
    st.markdown("Tahapan preprocessing data untuk mempersiapkan dataset sebelum proses modeling.")
    
    st.markdown("---")
    
    # Load data
    df = load_data()
    
    # ================================
    # SECTION 1: MISSING VALUES
    # ================================
    st.header("1⃣ Penanganan Missing Values")
    
    st.markdown("""
    ### Teori Missing Values
    
    **Missing values** adalah nilai yang hilang atau tidak tercatat dalam dataset. Penanganan missing values sangat penting karena:
    
    1. **Model ML tidak bisa memproses NaN** (akan error)
    2. **Bias analisis** jika missing tidak random
    3. **Kehilangan informasi** jika langsung dihapus
    
    **Strategi Penanganan:**
    
    | Metode | Kapan Digunakan | Formula/Pendekatan |
    |--------|-----------------|-------------------|
    | **Deletion** | Missing < 5%, MCAR | Hapus row/column |
    | **Mean Imputation** | Numerical, MCAR | $x_{missing} = \\bar{x}$ |
    | **Median Imputation** | Numerical, skewed | $x_{missing} = \\text{median}(x)$ |
    | **Mode Imputation** | Categorical | $x_{missing} = \\text{mode}(x)$ |
    | **KNN Imputation** | Complex patterns | Nearest neighbors |
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader(" Missing Values Analysis")
        missing_data = pd.DataFrame({
            'Column': df.columns,
            'Missing Count': df.isnull().sum().values,
            'Missing %': (df.isnull().sum().values / len(df) * 100).round(2)
        })
        st.dataframe(missing_data, use_container_width=True)
    
    with col2:
        st.subheader(" Visualisasi")
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(df.columns, df.isnull().sum(), color='coral')
        ax.set_ylabel('Missing Count')
        ax.set_title('Missing Values per Column')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        st.pyplot(fig)
    
    total_missing = df.isnull().sum().sum()
    if total_missing == 0:
        st.success("""
         **Tidak ada missing values** dalam dataset!
        
        **Interpretasi:**
        - Dataset memiliki kualitas tinggi
        - Proses pengumpulan data dilakukan dengan baik
        - Tidak diperlukan imputasi atau deletion
        - Semua 50,001 records dapat digunakan untuk analisis
        """)
    else:
        st.warning(f" Total missing values: {total_missing}")
    
    st.markdown("---")
    
    # ================================
    # SECTION 2: DUPLICATE CHECK
    # ================================
    st.header("2⃣ Penanganan Duplikat Data")
    
    st.markdown("""
    ### Mengapa Perlu Cek Duplikat?
    
    Data duplikat dapat menyebabkan:
    - **Overfitting**: Model terlalu "menghafal" data yang sama
    - **Bias evaluation**: Duplikat bisa masuk ke train DAN test set
    - **Inflated metrics**: Akurasi terlihat lebih tinggi dari seharusnya
    
    **Formula Duplicate Detection:**
    
    $$\\text{Duplicate} = \\{x_i : x_i = x_j, i \\neq j\\}$$
    """)
    
    duplicate_count = df.duplicated().sum()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Total Rows", f"{len(df):,}")
    with col2:
        st.metric("Duplicate Rows", f"{duplicate_count:,}")
    
    if duplicate_count == 0:
        st.success("""
         **Tidak ada data duplikat!**
        
        **Interpretasi:**
        - Setiap record adalah transaksi mobil unik
        - Tidak perlu penanganan duplikat
        - Data quality: EXCELLENT
        """)
    else:
        st.warning(f" Ditemukan {duplicate_count} rows duplikat")
    
    st.markdown("---")
    
    # ================================
    # SECTION 3: OUTLIER DETECTION
    # ================================
    st.header("3⃣ Deteksi & Penanganan Outlier")
    
    st.markdown("""
    ### Metode IQR (Interquartile Range)
    
    IQR adalah metode statistik robust untuk mendeteksi outliers berdasarkan distribusi data.
    
    **Formula:**
    
    $$IQR = Q_3 - Q_1$$
    
    $$\\text{Lower Bound} = Q_1 - 1.5 \\times IQR$$
    
    $$\\text{Upper Bound} = Q_3 + 1.5 \\times IQR$$
    
    **Contoh Perhitungan Manual (Price):**
    
    ```
    Data Price: [5000, 8000, 10000, 12000, 15000, 18000, 20000, 50000, 100000]
    
    Step 1: Urutkan data (sudah terurut)
    
    Step 2: Hitung Quartiles
    Q1 (25th percentile) = 8000 + (10000-8000) × 0.25 = $8,500
    Q3 (75th percentile) = 18000 + (20000-18000) × 0.75 = $19,500
    
    Step 3: Hitung IQR
    IQR = Q3 - Q1 = 19,500 - 8,500 = $11,000
    
    Step 4: Tentukan Bounds
    Lower Bound = 8,500 - (1.5 × 11,000) = 8,500 - 16,500 = -$8,000 ≈ $0
    Upper Bound = 19,500 + (1.5 × 11,000) = 19,500 + 16,500 = $36,000
    
    Step 5: Identifikasi Outliers
    Data > $36,000 → Outliers: [$50,000, $100,000] ← akan dihapus
    ```
    
    **Mengapa 1.5 × IQR?**
    - Empirically derived (works well for most distributions)
    - ~99.3% data normal berada dalam range ini
    - 3.0 × IQR = extreme outliers only
    """)
    
    numeric_cols = ['Engine size', 'Year of manufacture', 'Mileage', 'Price']
    
    # Display outlier info in tabs
    tabs = st.tabs(numeric_cols)
    
    for i, col in enumerate(numeric_cols):
        with tabs[i]:
            outliers_mask, lower, upper, Q1, Q3, IQR = detect_outliers_iqr(df, col)
            outlier_count = outliers_mask.sum()
            outlier_pct = (outlier_count / len(df) * 100)
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Q1 (25%)", f"{Q1:,.2f}")
            with col2:
                st.metric("Q3 (75%)", f"{Q3:,.2f}")
            with col3:
                st.metric("IQR", f"{IQR:,.2f}")
            with col4:
                st.metric("Outliers", f"{outlier_count:,} ({outlier_pct:.1f}%)")
            
            st.markdown(f"""
            ** Statistik {col}:**
            - Lower Bound: **{lower:,.2f}**
            - Upper Bound: **{upper:,.2f}**
            - Data yang dianggap normal: [{lower:,.2f}, {upper:,.2f}]
            """)
            
            # Boxplot
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
            
            axes[0].boxplot(df[col])
            axes[0].set_title(f'{col} - Before')
            axes[0].set_ylabel(col)
            axes[0].grid(True, alpha=0.3)
            
            df_clean_temp = remove_outliers_iqr(df, col)
            axes[1].boxplot(df_clean_temp[col])
            axes[1].set_title(f'{col} - After Removing Outliers')
            axes[1].set_ylabel(col)
            axes[1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Interpretation for each column
            if col == 'Price':
                st.markdown("""
                ** Interpretasi Outliers Price:**
                - Outliers tinggi (>Upper Bound) = Mobil luxury/premium (BMW, Porsche, dll)
                - Outliers rendah (<Lower Bound) = Mobil sangat murah (mungkin rusak/salvage)
                - **Keputusan**: Hapus outliers karena target variable → model lebih stabil
                """)
            elif col == 'Mileage':
                st.markdown("""
                ** Interpretasi Outliers Mileage:**
                - High outliers (>200K mil) = Mobil komersial/taksi/rental
                - Low outliers = Jarang (mileage 0 bukan outlier)
                - **Keputusan**: Hati-hati menghapus, high mileage masih valid
                """)
            elif col == 'Year of manufacture':
                st.markdown("""
                ** Interpretasi Outliers Year:**
                - Old cars (< ~1985) = Mobil klasik/antik
                - New cars (2024) = Mobil terbaru, bukan outlier
                - **Keputusan**: Mobil klasik mungkin memiliki pola pricing berbeda
                """)
            elif col == 'Engine size':
                st.markdown("""
                ** Interpretasi Outliers Engine Size:**
                - Large engines (>4.5L) = Mobil sport/SUV mewah
                - Small engines (<0.8L) = Mobil city car kecil
                - **Keputusan**: Tetap pertahankan, variasi engine penting untuk segmentasi
                """)
    
    st.markdown("---")
    
    # ================================
    # SECTION 4: APPLY OUTLIER REMOVAL
    # ================================
    st.header("4⃣ Hasil Penghapusan Outlier (Price Only)")
    
    st.markdown("""
    ### Strategi Penghapusan
    
    Berdasarkan analisis di atas, kita hanya menghapus outliers dari **Price** karena:
    
    1. **Price adalah target variable** → outliers dapat menyesatkan model
    2. **Outliers pada fitur lain valid** → merepresentasikan variasi pasar yang nyata
    3. **Mobil luxury/premium memerlukan model terpisah** → segmentasi berbeda
    
    **Formula yang diterapkan:**
    
    $$\\text{Data Valid} = \\{x : Q1 - 1.5 \\times IQR \\leq x \\leq Q3 + 1.5 \\times IQR\\}$$
    """)
    
    # Remove outliers from Price only
    df_clean = remove_outliers_iqr(df, 'Price')
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Data Awal", f"{len(df):,} rows")
    with col2:
        st.metric("Data Setelah Cleaning", f"{len(df_clean):,} rows")
    with col3:
        removed = len(df) - len(df_clean)
        st.metric("Data Dihapus", f"{removed:,} ({removed/len(df)*100:.1f}%)")
    
    st.success("""
    ** Hasil Penghapusan Outliers:**
    
    | Aspek | Sebelum | Sesudah |
    |-------|---------|---------|
    | **Total Records** | 50,001 | ~47,339 |
    | **Price Range** | $500 - $100K+ | $1K - $45K |
    | **Distribusi** | Heavily skewed | More normal |
    
    **Implikasi untuk Modeling:**
    - Model akan lebih fokus pada segmen pasar mainstream
    - Prediksi lebih akurat untuk mobil "normal"
    - Mobil luxury (>$50K) memerlukan model terpisah
    """)
    
    st.markdown("---")
    
    # ================================
    # SECTION 5: FEATURE ENGINEERING
    # ================================
    st.header("5⃣ Feature Engineering")
    
    st.markdown("""
    ### Membuat Fitur Baru
    
    Feature engineering adalah proses membuat fitur baru dari fitur existing untuk meningkatkan performa model.
    """)
    
    # Create features
    current_year = 2024
    df_clean['Car_Age'] = current_year - df_clean['Year of manufacture']
    df_clean['Mileage_per_Year'] = df_clean['Mileage'] / (df_clean['Car_Age'] + 1)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### Fitur #1: Car_Age (Umur Mobil)
        
        **Formula:**
        $$\\text{Car Age} = 2024 - \\text{Year of Manufacture}$$
        
        **Contoh Perhitungan:**
        ```
        Mobil tahun 2020:
        Car_Age = 2024 - 2020 = 4 tahun
        
        Mobil tahun 2010:
        Car_Age = 2024 - 2010 = 14 tahun
        ```
        
        **Mengapa penting?**
        - Lebih intuitif daripada "Year"
        - Langsung mengukur depresiasi
        - Korelasi negatif dengan Price
        """)
    
    with col2:
        st.markdown("""
        ### Fitur #2: Mileage_per_Year
        
        **Formula:**
        $$\\text{MPY} = \\frac{\\text{Total Mileage}}{\\text{Car Age} + 1}$$
        
        **Contoh Perhitungan:**
        ```
        Mobil 5 tahun dengan 60,000 mil:
        MPY = 60,000 / (5+1) = 10,000 mil/tahun
        → Penggunaan NORMAL
        
        Mobil 5 tahun dengan 150,000 mil:
        MPY = 150,000 / (5+1) = 25,000 mil/tahun
        → Penggunaan HEAVY (mungkin taksi/komersial)
        ```
        
        **Standar Industri:**
        - < 10K/year: Low usage (jarang dipakai)
        - 10-15K/year: Normal usage
        - > 20K/year: Heavy usage (wear tinggi)
        """)
    
    st.dataframe(df_clean[['Year of manufacture', 'Mileage', 'Car_Age', 'Mileage_per_Year']].head(10), 
                 use_container_width=True)
    
    st.markdown("---")
    
    # ================================
    # SECTION 6: FEATURE ENCODING
    # ================================
    st.header("6⃣ Feature Encoding")
    
    st.markdown("""
    ### Mengapa Perlu Encoding?
    
    Machine Learning model hanya bisa memproses **angka**, bukan teks. Encoding mengubah kategori menjadi numerik.
    
    **Dua Jenis Encoding:**
    
    | Metode | Kapan Digunakan | Contoh |
    |--------|-----------------|--------|
    | **Label Encoding** | Tree-based models, ordinal data | Fuel: Diesel=0, Hybrid=1, Petrol=2 |
    | **One-Hot Encoding** | Linear models, nominal data | Manufacturer → [0,1,0,0,...] |
    """)
    
    st.subheader(" Label Encoding - Fuel Type")
    
    from sklearn.preprocessing import LabelEncoder
    
    le = LabelEncoder()
    df_clean['Fuel_type_encoded'] = le.fit_transform(df_clean['Fuel type'])
    
    encoding_map = pd.DataFrame({
        'Fuel Type': le.classes_,
        'Encoded Value': range(len(le.classes_))
    })
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.dataframe(encoding_map, use_container_width=True)
    
    with col2:
        st.markdown("""
        ** Formula Label Encoding:**
        
        $$f: \\text{Category} \\rightarrow \\{0, 1, 2, ..., n-1\\}$$
        
        **Interpretasi:**
        - Diesel → 0
        - Hybrid → 1 
        - Petrol → 2
        
        **Catatan:**
        Urutan numerik **tidak berarti** Petrol > Hybrid > Diesel. 
        Untuk tree-based models, ini tidak masalah.
        """)
    
    st.markdown("---")
    
    st.subheader(" One-Hot Encoding - Top 10 Manufacturers")
    
    top_manufacturers = df_clean['Manufacturer'].value_counts().head(10).index.tolist()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Top 10 Manufacturers:**")
        for i, mfr in enumerate(top_manufacturers, 1):
            count = df_clean[df_clean['Manufacturer'] == mfr].shape[0]
            st.write(f"{i}. {mfr} ({count:,} units)")
    
    with col2:
        st.markdown("""
        ** Formula One-Hot Encoding:**
        
        Untuk k kategori, buat k kolom biner:
        
        $$\\text{OHE}(x) = [0, 0, ..., 1, ..., 0]$$
        
        **Contoh:**
        ```
        Manufacturer = "BMW"
        → [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
              ↑
            BMW column = 1
        
        Manufacturer = "VW"
        → [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
              ↑
            VW column = 1
        ```
        
        **Kenapa hanya Top 10?**
        - Menghindari curse of dimensionality
        - Manufacturer lain → "Other" category
        """)
    
    st.markdown("---")
    
    # ================================
    # SECTION 7: FEATURE SCALING
    # ================================
    st.header("7⃣ Feature Scaling (Standardization)")
    
    st.markdown("""
    ### Mengapa Perlu Scaling?
    
    Fitur memiliki range yang SANGAT berbeda:
    
    | Fitur | Min | Max | Range |
    |-------|-----|-----|-------|
    | Engine size | 0.6 | 6.6 | 6 |
    | Year | 1970 | 2024 | 54 |
    | Mileage | 0 | 300K | 300,000 |
    
    **Masalah tanpa scaling:**
    - Mileage akan "mendominasi" karena range besar
    - Gradient descent tidak optimal
    - Regularization tidak fair
    
    ** Formula Z-Score Standardization:**
    
    $$z = \\frac{x - \\mu}{\\sigma}$$
    
    **Contoh Perhitungan Manual:**
    
    ```
    Engine Size data: [1.6, 2.0, 2.5, 3.0, 3.5]
    Mean (μ) = (1.6+2.0+2.5+3.0+3.5)/5 = 2.52
    Std (σ) = sqrt(Σ(x-μ)²/n) = 0.753
    
    Untuk x = 1.6:
    z = (1.6 - 2.52) / 0.753
    z = -0.92 / 0.753
    z = -1.222 ← Nilai negatif = di bawah mean
    
    Untuk x = 3.5:
    z = (3.5 - 2.52) / 0.753
    z = 0.98 / 0.753
    z = 1.301 ← Nilai positif = di atas mean
    ```
    
    **Hasil setelah scaling:**
    - Mean baru = 0
    - Std baru = 1
    - Range ≈ [-3, +3] untuk 99.7% data
    """)
    
    from sklearn.preprocessing import StandardScaler
    
    scaler = StandardScaler()
    features_to_scale = ['Engine size', 'Year of manufacture', 'Mileage']
    
    # Show before/after
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Before Scaling")
        st.dataframe(df_clean[features_to_scale].describe().round(2), use_container_width=True)
    
    with col2:
        st.subheader("After Scaling")
        scaled_data = scaler.fit_transform(df_clean[features_to_scale])
        scaled_df = pd.DataFrame(scaled_data, columns=features_to_scale)
        st.dataframe(scaled_df.describe().round(2), use_container_width=True)
    
    st.success("""
    ** Hasil Scaling:**
    
    | Aspek | Sebelum | Sesudah |
    |-------|---------|---------|
    | **Mean** | Berbeda-beda | ~0 untuk semua |
    | **Std** | Berbeda-beda | ~1 untuk semua |
    | **Range** | Sangat berbeda | Seragam |
    
    **Catatan Penting:**
    - Train: **fit_transform** (hitung μ, σ, lalu transform)
    - Test: **transform only** (gunakan μ, σ dari train)
    - Production: Simpan scaler, reuse untuk prediksi baru
    """)
    
    st.markdown("---")
    
    # ================================
    # SUMMARY
    # ================================
    st.header(" Ringkasan Preprocessing Pipeline")
    
    st.success("""
    ### Pipeline Preprocessing yang Diterapkan
    
    | Step | Proses | Hasil |
    |------|--------|-------|
    | 1⃣ | Missing Values Check | 0 missing (clean data) |
    | 2⃣ | Duplicate Check | 0 duplicates |
    | 3⃣ | Outlier Detection (IQR) | Detected & analyzed |
    | 4⃣ | Outlier Removal (Price) | ~2,662 rows removed |
    | 5⃣ | Feature Engineering | 2 fitur baru (Car_Age, MPY) |
    | 6⃣ | Label Encoding | Fuel type → numeric |
    | 7⃣ | One-Hot Encoding | Top 10 manufacturers |
    | 8⃣ | Standardization | Z-score normalization |
    
    **Final Dataset:**
    - Records: ~47,339 (setelah outlier removal)
    - Features: 11+ (setelah encoding & engineering)
    - Target: Price (continuous)
    
    **Dataset siap untuk tahap Regression Analysis!**
    """)
    
    st.info("""
    ** Next Steps:**
    
    Lanjutkan ke halaman **Regression Analysis** untuk:
    1. Train-Test Split (80:20)
    2. Model Training (Linear Regression, Random Forest, etc.)
    3. Model Evaluation (R², MAE, RMSE)
    4. Model Comparison & Selection
    """)


if __name__ == "__main__":
    main()
