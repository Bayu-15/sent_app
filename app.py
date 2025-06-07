import streamlit as st
import pandas as pd
import joblib
from model.preprocessing import preprocess_text
from utils.visualisasi import show_pie_chart, generate_wordcloud
import chardet # Import chardet outside the if block for better practice

# Load model dan vectorizer
# Ensure the model and vectorizer files exist at these paths
try:
    model = joblib.load("model/naive_bayes_model.pkl") # Adjusted path to match saving location
    vectorizer = joblib.load("model/vectorizer.pkl") # Adjusted path to match saving location
except FileNotFoundError:
    st.error("Model or vectorizer files not found. Please ensure they are saved in the 'models' directory and the paths are correct.")
    st.stop() # Stop the script if models are not found

st.set_page_config(page_title="Klasifikasi Sentimen IKN", layout="wide")
st.title("📊 Aplikasi Klasifikasi Sentimen Relokasi Ibu Kota Negara (IKN)")

uploaded_file = st.file_uploader("Upload file CSV komentar", type=["csv"])

if uploaded_file:
    # Baca encoding dari file
    raw_data = uploaded_file.read()
    result = chardet.detect(raw_data)
    encoding = result['encoding']

    # Kembalikan pointer ke awal file
    uploaded_file.seek(0)

    # Baca CSV dengan encoding terdeteksi
    try:
        df = pd.read_csv(uploaded_file, encoding=encoding)
    except Exception as e:
        st.error(f"Error reading CSV file with encoding {encoding}: {e}")
        st.stop()

    if 'komentar' not in df.columns:
        st.error("File harus memiliki kolom 'komentar'")
    else:
        # Ensure preprocess_text function is accessible and works as expected
        try:
            df['cleaned'] = df['komentar'].astype(str).apply(preprocess_text) # Ensure column is string type
        except Exception as e:
            st.error(f"Error during text preprocessing: {e}")
            st.stop()

        # Ensure vectorizer is fitted before transforming
        # The vectorizer was fitted in the training notebook cells and saved.
        # If running this app directly, ensure the vectorizer is loaded correctly.
        try:
            X = vectorizer.transform(df['cleaned'])
            df['sentimen'] = model.predict(X)
        except Exception as e:
            st.error(f"Error during vectorization or prediction: {e}")
            st.stop()


        st.success("Prediksi berhasil!")
        st.dataframe(df[['komentar', 'sentimen']])

        # Ensure visualisasi functions are accessible and work as expected
        try:
            show_pie_chart(df)
            generate_wordcloud(df)
        except Exception as e:
             st.warning(f"Could not generate visualizations: {e}") # Use warning as it's not critical


        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("Download Hasil", csv, "hasil_klasifikasi.csv", "text/csv")