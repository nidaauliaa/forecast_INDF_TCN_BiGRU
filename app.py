from flask import Flask, render_template, jsonify
import pickle
import numpy as np
import pandas as pd  # Tambahkan pandas untuk baca CSV

from tensorflow.keras.models import load_model
from tcn import TCN
from tcn.tcn import ResidualBlock

app = Flask(__name__)

# Load Model & Scaler bawaan kamu
model = load_model(
    "model.keras",
    custom_objects={
        'TCN': TCN,
        'ResidualBlock': ResidualBlock
    }
)
scaler = pickle.load(open("scaler.pkl", "rb"))

@app.route("/")
def home():
    # Mengarah langsung ke templates/index.html
    return render_template("index.html")

# =======================================================
# ENDPOINT API BARU: Membaca CSV & Kirim ke Dashboard Website
# =======================================================
@app.route("/api/data")
def get_dashboard_data():
    try:
        # 1. BACA DATA AKTUAL INDF (Menggunakan delimiter ';')
        df_hist = pd.read_csv("Data Historis INDF.csv", sep=';')
        
        # Deteksi otomatis nama kolom tanggal dan harga
        col_tanggal = 'Tanggal' if 'Tanggal' in df_hist.columns else 'Date'
        col_harga = 'Terakhir' if 'Terakhir' in df_hist.columns else ('Close' if 'Close' in df_hist.columns else 'Price')
        
        # Ambil kolom yang diperlukan saja
        df_hist = df_hist[[col_tanggal, col_harga]].copy()
        
        # Konversi kolom tanggal menjadi objek Datetime (Format sesuai file: %d/%m/%y)
        df_hist[col_tanggal] = pd.to_datetime(df_hist[col_tanggal], format="%d/%m/%y", errors='coerce')
        
        # --- PERBAIKAN TOTAL: Logika Konversi Ribuan dari Skrip Kamu ---
        df_hist[col_harga] = pd.to_numeric(df_hist[col_harga], errors='coerce')
        df_hist[col_harga] = df_hist[col_harga].apply(lambda x: x * 1000 if x < 10 else x)
        
        # Hapus data kosong jika ada
        df_hist = df_hist.dropna(subset=[col_tanggal, col_harga])
        
        # Urutkan dari tanggal terlama ke terbaru (Sangat penting untuk runut waktu bursa)
        df_hist = df_hist.sort_values(col_tanggal).reset_index(drop=True)
        
        # Ambil 30 baris terakhir untuk visualisasi dashboard
        df_hist_last30 = df_hist.tail(30).copy()
        
        # --- PENYELARASAN FORMAT TANGGAL KE ISO (YYYY-MM-DD) ---
        # Kita seragamkan ke format ISO agar JavaScript Chart.js tidak bingung memetakan polanya
        df_hist_last30['Tanggal_Format'] = df_hist_last30[col_tanggal].dt.strftime('%Y-%m-%d')
        
        # Masukkan ke list dictionary untuk JSON aktual
        historical_json = []
        for _, row in df_hist_last30.iterrows():
            historical_json.append({
                "Tanggal": row['Tanggal_Format'],
                "Terakhir": int(row[col_harga])
            })

        # 2. BACA DATA FORECAST 30 HARI
        try:
            # 2. BACA DATA FORECAST GABUNGAN SEMUA MODEL
            df_fore = pd.read_csv("gabungan.csv")

            # Pastikan format tanggal ISO
            df_fore['Tanggal'] = pd.to_datetime(
                df_fore['Tanggal'],
                errors='coerce'
            ).dt.strftime('%Y-%m-%d')

            forecast_json = df_fore.to_dict(orient="records")
            if len(df_fore.columns) == 1:
                df_fore = pd.read_csv("gabungan.csv", sep=';')
        except Exception:
            df_fore = pd.read_csv("gabungan.csv", sep=';')
            
        # Penyelarasan format tanggal forecast ke ISO (YYYY-MM-DD) jika tipenya masih string acak/bahasa Indonesia
        # Kita asumsikan kolomnya bernama 'Tanggal'
        if 'Tanggal' in df_fore.columns:
            # Mengubah berbagai variasi string tanggal ke datetime, lalu diubah ke YYYY-MM-DD
            df_fore['Tanggal'] = pd.to_datetime(df_fore['Tanggal'], errors='coerce').dt.strftime('%Y-%m-%d')
            
        forecast_json = df_fore.to_dict(orient="records")

        return jsonify({
            "status": "success",
            "historical": historical_json,
            "forecast": forecast_json
        })
        
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route("/predict")
def predict():
    last_sequence = np.load("last_sequence.npy")
    pred = model.predict(last_sequence)
    pred_real = scaler.inverse_transform(pred)
    hasil = round(pred_real[0][0] * 1000)
    
    return render_template("index.html", prediction=hasil)

if __name__ == "__main__":
    app.run(debug=True)