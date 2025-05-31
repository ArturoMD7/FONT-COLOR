from flask import Flask, render_template, request
import joblib
import numpy as np
import pandas as pd

# Inicializar Flask
app = Flask(__name__)

# Cargar modelo, scaler y label encoder
modelo = joblib.load("modelo_font_color.pkl")
scaler = joblib.load("scaler.pkl")
le = joblib.load("label_encoder.pkl")

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    rgb = None
    debug_details = {}

    if request.method == "POST":
        try:
            # Obtener y validar valores
            r = request.form.get("red", type=int) or 0
            g = request.form.get("green", type=int) or 0
            b = request.form.get("blue", type=int) or 0

            rgb = [r, g, b]
            debug_details['input_rgb'] = f"Input RGB: {rgb}"
            
            valor_df = pd.DataFrame([rgb], columns=["RED", "GREEN", "BLUE"])
            debug_details['raw_values'] = f"DataFrame: {valor_df.values}"
            
            valor_escalado = scaler.transform(valor_df)
            debug_details['scaled_values'] = f"Scaled: {valor_escalado}"
            
            pred = modelo.predict(valor_escalado)
            debug_details['raw_prediction'] = f"Raw prediction: {pred}"
            
            prediction = le.inverse_transform(pred)[0]
            debug_details['final_prediction'] = f"Final prediction: {prediction}"
            
            print("\n".join(debug_details.values()))  # Print all debug info
            
        except Exception as e:
            print("Error:", e)
            prediction = f"Error en la predicción: {str(e)}"
            debug_details['error'] = str(e)

    return render_template(
        "index.html", 
        prediction=prediction, 
        rgb=rgb,
        debug_info=debug_details
    )

if __name__ == "__main__":
    app.run(debug=True)