import streamlit as st


def Home():
    st.sidebar.page_link('app.py', label='Home', icon='🏠')


def EDA():
    st.sidebar.page_link('pages/1_EDA.py', label='EDA', icon='📊')


def Preprocessing():
    st.sidebar.page_link('pages/2_Preprocessing.py', label='Preprocessing', icon='🔧')


def Regression():
    st.sidebar.page_link('pages/3_Regression.py', label='Regression Analysis', icon='🤖')


def Prediction():
    st.sidebar.page_link('pages/4_Prediction.py', label='Prediction', icon='🔮')


def Nav():
    # Display logo at the top of sidebar (centered)
    col1, col2, col3 = st.sidebar.columns(3)
    with col2:
        st.image("logo.png", width=150)
    st.sidebar.title("🚗 Car Price Prediction")
    st.sidebar.markdown("---")
    Home()
    EDA()
    Preprocessing()
    Regression()
    Prediction()
    st.sidebar.markdown("---")
    st.sidebar.success("Gunakan menu **Prediction** untuk memprediksi harga mobil bekas.")
