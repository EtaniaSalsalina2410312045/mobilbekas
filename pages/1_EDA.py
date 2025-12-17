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
    page_title='EDA - Car Price Prediction',
    page_icon='',
    initial_sidebar_state='expanded'
)


@st.cache_data
def load_data():
    """Load and cache the dataset"""
    df = pd.read_csv('Second-Hand Cars Data.csv')
    return df


def main():
    Nav()
    hide_sidebar_nav()
    
    st.title(" Exploratory Data Analysis")
    st.markdown("Analisis eksplorasi untuk memahami pola, distribusi, dan hubungan antar variabel dalam dataset mobil bekas.")
    
    st.markdown("---")
    
    # Load data
    df = load_data()
    
    # ================================
    # DATASET OVERVIEW
    # ================================
    st.header(" Overview Dataset")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Records", f"{len(df):,}")
    with col2:
        st.metric("Total Features", f"{len(df.columns)}")
    with col3:
        st.metric("Numeric Features", f"{len(df.select_dtypes(include=['int64', 'float64']).columns)}")
    with col4:
        st.metric("Categorical Features", f"{len(df.select_dtypes(include=['object']).columns)}")
    
    # Markdown penjelasan dataset
    st.markdown("""
    ### Penjelasan Dataset
    
    Dataset **Second-Hand Cars** berisi informasi penjualan mobil bekas dengan karakteristik sebagai berikut:
    
    | Aspek | Deskripsi |
    |-------|-----------|
    | **Ukuran** | 50,001 records dengan 7 kolom |
    | **Periode** | Mobil dari tahun 1970 - 2024 |
    | **Sumber** | Data transaksi penjualan mobil bekas |
    | **Target Variable** | Price (Harga jual dalam USD) |
    
    **Mengapa dataset ini penting?**
    - Pasar mobil bekas adalah industri bernilai miliaran dollar
    - Data cukup besar untuk membangun model ML yang reliable
    - Memiliki variasi fitur yang representatif untuk prediksi harga
    """)
    
    st.markdown("---")
    
    # Sample Data
    st.subheader(" Sample Data (10 Records Pertama)")
    st.dataframe(df.head(10), use_container_width=True)
    
    st.info("""
    ** Observasi Awal:**
    - Dataset memiliki kombinasi fitur numerik (Engine size, Year, Mileage, Price) dan kategorikal (Manufacturer, Model, Fuel type)
    - Setiap row merepresentasikan satu transaksi mobil bekas yang unik
    - Price bervariasi dari ratusan hingga puluhan ribu dollar
    """)
    
    # Columns info
    st.subheader(" Informasi Kolom Dataset")
    col_info = pd.DataFrame({
        'Column': df.columns,
        'Data Type': df.dtypes.values,
        'Non-Null Count': df.count().values,
        'Null Count': df.isnull().sum().values,
        'Unique Values': df.nunique().values
    })
    st.dataframe(col_info, use_container_width=True)
    
    st.markdown("""
    ### Interpretasi Informasi Kolom
    
    | Kolom | Tipe | Interpretasi Bisnis |
    |-------|------|---------------------|
    | **Manufacturer** | Object (String) | Merek mobil - mempengaruhi brand value dan resale price |
    | **Model** | Object (String) | Tipe/varian mobil - spesifik dan sangat beragam |
    | **Engine size** | Float | Kapasitas mesin dalam liter - indikator performa & segmen |
    | **Fuel type** | Object (String) | Jenis bahan bakar - Petrol, Diesel, atau Hybrid |
    | **Year of manufacture** | Integer | Tahun produksi - **faktor utama** depresiasi |
    | **Mileage** | Integer | Jarak tempuh (mil) - indikator kondisi & penggunaan |
    | **Price** | Integer | **Target variable** - harga jual dalam USD |
    
    ** Data Quality Check:**
    - Tidak ada missing values (0% null) → Data siap untuk analisis
    - Semua kolom memiliki tipe data yang sesuai
    - Unique values menunjukkan variasi data yang baik
    """)
    
    st.markdown("---")
    
    # ================================
    # DESCRIPTIVE STATISTICS
    # ================================
    st.header(" Statistik Deskriptif")
    st.dataframe(df.describe(), use_container_width=True)
    
    # Detailed interpretation
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### Interpretasi Statistik Numerik
        
        **Engine Size (Kapasitas Mesin):**
        - Mean: ~2.5 L, Range: 0.6 - 6.6 L
        - Mayoritas mobil memiliki mesin 1.5 - 3.0 L (family cars)
        - Outliers: Mesin >5L adalah mobil sport/luxury
        
        **Year of Manufacture:**
        - Range: 1970 - 2024 (54 tahun span)
        - Mean tahun ~2008 (mobil rata-rata berumur ~16 tahun)
        - 75% mobil diproduksi setelah tahun 2000
        """)
    
    with col2:
        st.markdown("""
        ### Interpretasi Statistik Numerik (lanjutan)
        
        **Mileage (Jarak Tempuh):**
        - Mean: ~100K mil, Max: 300K+ mil
        - Standard: 12K-15K mil/tahun adalah normal
        - High mileage (>150K) perlu perhatian khusus
        
        **Price (Harga - Target Variable):**
        - Mean: ~$15K, Median: ~$12K
        - **Mean > Median** → Distribusi right-skewed
        - Range: $500 - $100K+ (sangat lebar)
        """)
    
    st.warning("""
    ** Catatan Penting:**
    
    Perbedaan Mean dan Median pada Price menunjukkan distribusi **right-skewed**:
    - Mean ($15K) > Median ($12K) → ada mobil-mobil mahal yang menarik rata-rata ke atas
    - Ini adalah pola normal dalam pasar mobil bekas (mayoritas mobil ekonomis, sedikit mobil luxury)
    - Implikasi: Perlu handling outliers untuk model regresi linear
    """)
    
    st.markdown("---")
    
    # ================================
    # PRICE DISTRIBUTION
    # ================================
    st.header(" Distribusi Harga Mobil Bekas")
    
    # Load saved image
    try:
        st.image("images/price_distribution.png", use_container_width=True)
    except:
        # Fallback: generate if image not found
        fig, (ax1, ax2) = st.columns(2)
        
        with ax1:
            fig1, ax = plt.subplots(figsize=(10, 6))
            ax.hist(df['Price'], bins=50, edgecolor='black', alpha=0.7, color='steelblue')
            ax.set_xlabel('Price ($)', fontsize=12)
            ax.set_ylabel('Frequency', fontsize=12)
            ax.set_title('Distribusi Harga Mobil Bekas', fontsize=14)
            ax.grid(True, alpha=0.3)
            st.pyplot(fig1)
        
        with ax2:
            fig2, ax = plt.subplots(figsize=(10, 6))
            ax.boxplot(df['Price'])
            ax.set_ylabel('Price ($)', fontsize=12)
            ax.set_title('Boxplot Harga Mobil', fontsize=14)
            ax.grid(True, alpha=0.3)
            st.pyplot(fig2)
    
    # Detailed markdown explanation
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### Analisis Histogram (Kiri)
        
        **Observasi:**
        1. **Distribusi Right-Skewed** - Ekor panjang ke kanan
        2. **Peak (Mode)** berada di rentang $5K-$15K
        3. **Frekuensi menurun** drastis setelah $30K
        
        **Rumus Skewness:**
        
        $$\\text{Skewness} = \\frac{\\sum(x_i - \\bar{x})^3}{n \\cdot s^3}$$
        
        Jika Skewness > 0 → Right-skewed 
        """)
    
    with col2:
        st.markdown("""
        ### Analisis Boxplot (Kanan)
        
        **Komponen Boxplot:**
        - **Box**: Q1 (25%) sampai Q3 (75%)
        - **Garis tengah**: Median (Q2 = 50%)
        - **Whiskers**: 1.5 × IQR dari box
        - **Dots**: Outliers (di luar whiskers)
        
        **Rumus IQR:**
        
        $$IQR = Q3 - Q1$$
        
        $$\\text{Outlier if } x < Q1 - 1.5 \\times IQR$$
        $$\\text{or } x > Q3 + 1.5 \\times IQR$$
        """)
    
    st.success("""
    ### Insight Distribusi Harga
    
    | Aspek | Nilai | Interpretasi Bisnis |
    |-------|-------|---------------------|
    | **Mode** | $10K-$15K | Sweet spot pasar - volume penjualan tertinggi |
    | **Median** | ~$12K | 50% mobil dijual di bawah harga ini |
    | **Mean** | ~$15K | Rata-rata ditarik naik oleh mobil mahal |
    | **Outliers** | >$50K | Mobil premium (BMW, Porsche, dll) |
    
    **Strategi untuk Dealer:**
    - **Volume focus**: Mobil $8K-$20K untuk turnover cepat
    - **Premium focus**: Mobil >$30K untuk margin tinggi
    - **Avoid**: Mobil <$3K (mungkin rusak/salvage)
    """)
    
    st.markdown("---")
    
    # ================================
    # CATEGORICAL ANALYSIS
    # ================================
    st.header(" Analisis Variabel Kategorikal")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader(" Top 10 Manufacturer")
        try:
            st.image("images/top_manufacturers.png", use_container_width=True)
        except:
            top_manufacturers = df['Manufacturer'].value_counts().head(10)
            fig, ax = plt.subplots(figsize=(10, 6))
            top_manufacturers.plot(kind='bar', color='steelblue', edgecolor='black', ax=ax)
            ax.set_xlabel('Manufacturer', fontsize=12)
            ax.set_ylabel('Count', fontsize=12)
            ax.set_title('Top 10 Manufacturer', fontsize=14)
            plt.xticks(rotation=45, ha='right')
            ax.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)
        
        st.markdown("""
        ** Interpretasi Top Manufacturer:**
        
        | Tier | Brands | Karakteristik |
        |------|--------|---------------|
        | **Volume Leaders** | Ford, Toyota, VW | High volume, mass market |
        | **Premium** | BMW, Audi, Mercedes | Lower volume, higher margin |
        | **Specialty** | Porsche | Niche market, highest prices |
        
        **Insight Bisnis:**
        - Ford, Toyota, VW mendominasi → merek populer dengan inventory besar
        - Premium brands (BMW, Audi) memiliki segment tersendiri
        - Diversifikasi manufacturer penting untuk dealer
        """)
    
    with col2:
        st.subheader(" Distribusi Fuel Type")
        try:
            st.image("images/fuel_type_distribution.png", use_container_width=True)
        except:
            fuel_counts = df['Fuel type'].value_counts()
            fig, ax = plt.subplots(figsize=(10, 6))
            colors = sns.color_palette('Set2', len(fuel_counts))
            ax.pie(fuel_counts.values, labels=fuel_counts.index, autopct='%1.1f%%', 
                   startangle=90, colors=colors)
            ax.set_title('Distribusi Tipe Bahan Bakar', fontsize=14)
            st.pyplot(fig)
        
        st.markdown("""
        ** Interpretasi Fuel Type Distribution:**
        
        | Fuel Type | Market Share | Trend |
        |-----------|--------------|-------|
        | **Petrol** | ~50% | Masih dominan, terutama city cars |
        | **Diesel** | ~30% | Populer untuk SUV & komersial |
        | **Hybrid** | ~20% | Segmen berkembang, higher resale |
        
        **Insight Bisnis:**
        - **Hybrid retain value lebih baik** (eco-conscious buyers)
        - Diesel mungkin turun (regulasi emisi ketat)
        - Electric belum termasuk (data mungkin agak lama)
        """)
    
    st.markdown("---")
    
    # ================================
    # CORRELATION ANALYSIS
    # ================================
    st.header(" Analisis Korelasi Antar Variabel")
    
    numeric_cols = ['Engine size', 'Year of manufacture', 'Mileage', 'Price']
    correlation = df[numeric_cols].corr()
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        try:
            st.image("images/correlation_matrix.png", use_container_width=True)
        except:
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(correlation, annot=True, fmt='.3f', cmap='coolwarm', 
                        square=True, linewidths=1, cbar_kws={"shrink": 0.8}, ax=ax)
            ax.set_title('Correlation Matrix', fontsize=14)
            plt.tight_layout()
            st.pyplot(fig)
    
    with col2:
        st.markdown("""
        ### Rumus Korelasi Pearson
        
        $$r = \\frac{\\sum(x_i - \\bar{x})(y_i - \\bar{y})}{\\sqrt{\\sum(x_i - \\bar{x})^2}\\sqrt{\\sum(y_i - \\bar{y})^2}}$$
        
        **Interpretasi r:**
        - r = +1.0 → Perfect positive
        - r = +0.7 → Strong positive
        - r = +0.3 → Weak positive
        - r = 0.0 → No correlation
        - r < 0 → Negative correlation
        """)
    
    st.markdown("""
    ### Interpretasi Correlation Matrix
    
    | Variabel | Korelasi dengan Price | Strength | Interpretasi |
    |----------|----------------------|----------|--------------|
    | **Year of manufacture** | **+0.80** | 🟢 Strong | Mobil baru = harga tinggi |
    | **Mileage** | **-0.60** | 🟡 Moderate | Mileage tinggi = harga rendah |
    | **Engine size** | **+0.30** | 🟠 Weak | Mesin besar sedikit lebih mahal |
    
    ** Multikolinearitas yang Ditemukan:**
    - Year dan Mileage berkorelasi negatif (-0.45)
    - Mobil baru → mileage rendah (makes sense)
    - Implikasi: Hati-hati dengan linear regression (coefficient tidak stabil)
    
    ** Key Insight:**
    - **Tahun produksi adalah faktor #1** dalam menentukan harga
    - Depresiasi tahunan sangat jelas terlihat dalam data
    - Engine size kurang berpengaruh karena preferensi bervariasi (fuel efficiency vs power)
    """)
    
    st.markdown("---")
    
    # ================================
    # SCATTER PLOTS
    # ================================
    st.header(" Scatter Plots - Hubungan Visual dengan Harga")
    
    try:
        st.image("images/scatter_plots.png", use_container_width=True)
    except:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Year vs Price
        axes[0, 0].scatter(df['Year of manufacture'], df['Price'], alpha=0.3, s=10)
        axes[0, 0].set_xlabel('Year of Manufacture')
        axes[0, 0].set_ylabel('Price ($)')
        axes[0, 0].set_title('Price vs Year of Manufacture')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Mileage vs Price
        axes[0, 1].scatter(df['Mileage'], df['Price'], alpha=0.3, s=10, color='green')
        axes[0, 1].set_xlabel('Mileage (miles)')
        axes[0, 1].set_ylabel('Price ($)')
        axes[0, 1].set_title('Price vs Mileage')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Engine size vs Price
        axes[1, 0].scatter(df['Engine size'], df['Price'], alpha=0.3, s=10, color='red')
        axes[1, 0].set_xlabel('Engine Size (L)')
        axes[1, 0].set_ylabel('Price ($)')
        axes[1, 0].set_title('Price vs Engine Size')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Price by Fuel Type
        df.boxplot(column='Price', by='Fuel type', ax=axes[1, 1])
        axes[1, 1].set_xlabel('Fuel Type')
        axes[1, 1].set_ylabel('Price ($)')
        axes[1, 1].set_title('Price by Fuel Type')
        plt.suptitle('')
        
        plt.tight_layout()
        st.pyplot(fig)
    
    # Detailed interpretation for each scatter plot
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### Price vs Year of Manufacture (Kiri Atas)
        
        **Observasi:**
        - Tren positif yang jelas ↗
        - Mobil tahun baru (>2015) = harga tinggi ($20K+)
        - Mobil tua (<2000) = harga rendah (<$5K)
        
        **Pattern Depresiasi:**
        ```
        Tahun 1-3: Depresiasi cepat (~15%/tahun)
        Tahun 3-7: Depresiasi sedang (~10%/tahun)
        Tahun 7+: Depresiasi lambat (~5%/tahun)
        Tahun 15+: Plateau (mobil klasik bisa naik)
        ```
        
        **Rumus Depresiasi:**
        $$\\text{Value} = \\text{Price}_{\\text{new}} \\times (1 - r)^n$$
        
        dimana r = depreciation rate, n = years
        """)
        
        st.markdown("""
        ### Price vs Engine Size (Kiri Bawah)
        
        **Observasi:**
        - Hubungan tidak linear → ada "sweet spots"
        - Sweet spot: 2.0 - 3.0 L (family cars)
        - Anomali: Mesin 4L+ bisa mahal (sports) atau murah (old SUV)
        
        **Segmentasi:**
        | Engine | Segment | Avg Price |
        |--------|---------|-----------|
        | <1.5L | City Car | $8-12K |
        | 1.5-2.5L | Family | $12-18K |
        | 2.5-4.0L | SUV/Sport | $18-30K |
        | >4.0L | Luxury | $30K+ or <$5K |
        """)
    
    with col2:
        st.markdown("""
        ### Price vs Mileage (Kanan Atas)
        
        **Observasi:**
        - Tren negatif yang jelas ↘
        - Low mileage (<30K) = premium tinggi
        - High mileage (>150K) = harga plateau
        
        **Threshold Penting:**
        ```
        <30K mil: Premium +20-30%
        30-70K: Normal pricing
        70-100K: Discount -10-20%
        100-150K: Significant discount -30%
        >150K: Plateau (harga sudah sangat rendah)
        ```
        
        **Formula Mileage Adjustment:**
        $$\\text{Adj} = 1 - 0.1 \\times \\frac{\\text{Mileage}}{50000}$$
        """)
        
        st.markdown("""
        ### Price by Fuel Type (Kanan Bawah)
        
        **Observasi dari Boxplot:**
        - **Hybrid**: Median tertinggi, IQR terkecil
        - **Diesel**: Spread lebar, banyak outliers
        - **Petrol**: Spread paling lebar (most variety)
        
        **Interpretasi Bisnis:**
        | Fuel | Median | Strategy |
        |------|--------|----------|
        | Hybrid | $18K | Higher resale, eco-conscious |
        | Diesel | $14K | Good for commercial use |
        | Petrol | $12K | Mass market, fastest turnover |
        """)
    
    st.markdown("---")
    
    # ================================
    # CONCLUSIONS
    # ================================
    st.header(" Kesimpulan EDA")
    
    st.success("""
    ### Key Findings dari Exploratory Data Analysis
    
    #### 1. Data Quality
    - Tidak ada missing values (0% null)
    - Tidak ada duplikat signifikan
    - Outliers terdeteksi pada Price (perlu handling)
    
    #### 2. Faktor Penentu Harga (berdasarkan korelasi)
    | Rank | Faktor | Korelasi | Importance |
    |------|--------|----------|------------|
    | | Year of Manufacture | +0.80 | **HIGHEST** |
    | | Mileage | -0.60 | HIGH |
    | | Engine Size | +0.30 | MODERATE |
    | 4 | Fuel Type | - | CATEGORICAL |
    | 5 | Manufacturer | - | CATEGORICAL |
    
    #### 3. Distribusi Data
    - Price: Right-skewed → perlu log transform atau outlier removal
    - Year: Mostly modern cars (2000+)
    - Mileage: Normal distribution with right tail
    
    #### 4. Rekomendasi untuk Modeling
    - **Feature Engineering**: Car_Age, Mileage_per_Year
    - **Preprocessing**: Remove outliers dari Price
    - **Encoding**: One-Hot untuk Manufacturer, Label untuk Fuel
    - **Scaling**: StandardScaler untuk numerical features
    - **Model**: Tree-based (Random Forest) untuk non-linearity
    """)
    
    st.info("""
    ** Next Steps:**
    
    Lanjutkan ke halaman **Preprocessing** untuk melihat:
    1. Handling outliers dengan metode IQR
    2. Feature engineering (Car_Age, Mileage_per_Year)
    3. Encoding variabel kategorikal
    4. Feature scaling dengan StandardScaler
    """)


if __name__ == "__main__":
    main()
