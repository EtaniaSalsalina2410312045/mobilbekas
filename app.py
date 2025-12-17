import streamlit as st
from modules.nav import Nav
from modules.styles import hide_sidebar_nav


st.set_page_config(
    layout='wide',
    page_title='Second Hand Car Price Prediction',
    page_icon='',
    initial_sidebar_state='expanded'
)


def main():
    Nav()
    hide_sidebar_nav()
    
    # Custom CSS
    st.markdown("""
    <style>
    .main-title {
        text-align: center;
        font-size: 3.5rem;
        font-weight: bold;
        color: #1E88E5;
        margin-bottom: 0;
        padding-top: 1rem;
    }
    .subtitle {
        text-align: center;
        font-size: 1.3rem;
        color: var(--text-color);
        margin-top: 0.5rem;
        margin-bottom: 2rem;
        opacity: 0.8;
    }
    .feature-card {
        background: rgba(30, 136, 229, 0.1);
        border-radius: 10px;
        padding: 1.5rem;
        margin: 0.5rem 0;
        border-left: 4px solid #1E88E5;
        border: 1px solid rgba(30, 136, 229, 0.3);
    }
    .feature-card h4 {
        color: var(--text-color);
        margin-top: 0;
    }
    .feature-card p {
        color: var(--text-color);
        opacity: 0.9;
        margin-bottom: 0;
    }
    .metric-card {
        background: rgba(30, 136, 229, 0.15);
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        border: 1px solid rgba(30, 136, 229, 0.3);
    }
    
    /* Dark mode support */
    @media (prefers-color-scheme: dark) {
        .feature-card {
            background: rgba(30, 136, 229, 0.15);
            border: 1px solid rgba(30, 136, 229, 0.4);
        }
        .metric-card {
            background: rgba(30, 136, 229, 0.2);
            border: 1px solid rgba(30, 136, 229, 0.4);
        }
    }
    </style>
    """, unsafe_allow_html=True)
    
    # ================================
    # HERO SECTION
    # ================================
    st.markdown('<h1 class="main-title"> Second Hand Car Price Prediction</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Prediksi Harga Mobil Bekas Menggunakan Machine Learning</p>', unsafe_allow_html=True)
    
    # Hero Image
    left, center, right = st.columns(3)
    with center:
        st.image("car.png", caption="Prediksi Harga Mobil Bekas dengan Akurasi Tinggi")
    
    st.markdown("---")
    
    # ================================
    # PENGANTAR PENELITIAN
    # ================================
    st.header(" Tentang Aplikasi")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### Latar Belakang
        
        **Pasar Mobil Bekas** adalah industri yang berkembang pesat dengan tantangan 
        utama dalam menentukan harga yang tepat.
        
        **Fakta Penting:**
        - **Jutaan** mobil bekas diperjualbelikan setiap tahun
        - Harga dipengaruhi banyak faktor: umur, mileage, brand, dll
        - Depresiasi mobil tidak linear - sulit diprediksi manual
        - ⏰ Penetapan harga yang akurat sangat penting untuk profit
        
        Sistem prediksi otomatis berbasis Machine Learning dapat membantu 
        dealer dan pembeli menentukan harga yang fair.
        """)
    
    with col2:
        st.markdown("""
        ### Solusi yang Ditawarkan
        
        Sistem ini menggunakan **Gradient Boosting Regressor** untuk memprediksi 
        harga mobil bekas berdasarkan:
        
        | Fitur | Deskripsi |
        |-------|-----------|
        | **Tahun Produksi** | Umur mobil (faktor dominan ~38%) |
        | **Engine Size** | Kapasitas mesin dalam liter |
        | **Mileage** | Jarak tempuh dalam mil |
        | **Fuel Type** | Petrol, Diesel, atau Hybrid |
        | **Manufacturer** | Merek mobil |
        
        **Performa Model:**
        - R² Score: **96.9%**
        - MAE: **$1,174** (error rata-rata)
        """)
    
    st.markdown("---")
    
    # ================================
    # MODEL METRICS
    # ================================
    st.header(" Performa Model")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label=" R² Score",
            value="96.9%",
            delta="Best Model"
        )
    
    with col2:
        st.metric(
            label=" MAE",
            value="$1,174",
            delta="-64% vs Linear Reg",
            delta_color="inverse"
        )
    
    with col3:
        st.metric(
            label=" RMSE",
            value="$1,847",
            delta="-57% vs Linear Reg",
            delta_color="inverse"
        )
    
    with col4:
        st.metric(
            label=" Dataset",
            value="47,339",
            delta="Records"
        )
    
    st.markdown("---")
    
    # ================================
    # FEATURE IMPORTANCE
    # ================================
    st.header(" Faktor Penentu Harga")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        | Rank | Faktor | Pengaruh |
        |------|--------|----------|
        | | **Umur Mobil** | 38.4% |
        | | **Tahun Produksi** | 36.3% |
        | | **Ukuran Mesin** | 12.9% |
        | 4 | **Jarak Tempuh** | 7.2% |
        | 5 | **Tipe Bahan Bakar** | 2.1% |
        """)
    
    with col2:
        st.info("""
        ** Insight Utama:**
        
        - **Depresiasi** adalah faktor dominan dalam penentuan harga
        - Mobil dengan **mileage rendah** per tahun memiliki nilai lebih tinggi
        - **Hybrid** vehicles cenderung mempertahankan nilai lebih baik
        - **Brand** memiliki pengaruh kecil namun tetap signifikan
        """)
    
    st.markdown("---")
    
    # ================================
    # NAVIGATION GUIDE
    # ================================
    st.header(" Panduan Navigasi")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="feature-card">
        <h4> EDA</h4>
        <p>Eksplorasi data: distribusi harga, korelasi, dan visualisasi</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="feature-card">
        <h4> Preprocessing</h4>
        <p>Proses cleaning data: missing values, outliers, encoding</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-card">
        <h4> Model</h4>
        <p>Arsitektur Gradient Boosting dan perbandingan model</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="feature-card">
        <h4> Evaluation</h4>
        <p>Metrik evaluasi: R², MAE, RMSE, dan visualisasi</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="feature-card">
        <h4> Prediction</h4>
        <p>Prediksi harga mobil bekas berdasarkan input user</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="feature-card">
        <h4> About</h4>
        <p>Informasi tentang project dan tim pengembang</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Footer
    st.markdown("""
    <div style="text-align: center; color: #888; padding: 2rem;">
        <p> Second Hand Car Price Prediction System</p>
        <p>Built with using Streamlit & Scikit-learn</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
