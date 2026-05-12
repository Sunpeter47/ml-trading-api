from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import uvicorn
import joblib
import pandas as pd
import requests
import os

app = FastAPI(title="NQ ML Trading Radar API")

# Betöltjük a betanított Random Forest modellt.
try:
    rf_model = joblib.load('pro_nq_model_5m.joblib')
    print("✅ ML Modell sikeresen betöltve!")
except Exception as e:
    print(f"❌ Hiba a modell betöltésekor: {e}")

# Kinyerjük a titkos kulcsokat a Render Környezeti Változóiból (Biztonság!)
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN", "IDE_JON_A_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "IDE_JON_A_CHAT_ID")

def send_telegram_message(message: str):
    """Küld egy üzenetet a Telegramodra."""
    if TELEGRAM_TOKEN == "IDE_JON_A_TOKEN":
        print("A Telegram még nincs beállítva.")
        return
        
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
        "parse_mode": "HTML"
    }
    try:
        requests.post(url, json=payload)
    except Exception as e:
        print(f"Hiba a Telegram üzenet küldésekor: {e}")

@app.post("/webhook")
async def tradingview_webhook(request: Request):
    """Ezt hívja a TradingView, ha setupot lát a charton."""
    try:
        data = await request.json()
        
        features = ['Close', 'RSI_14', 'SMA_20', 'Dist_to_Lower', 'Dist_to_Upper', 'SMA_200', 'Is_Uptrend', 'Dist_to_SMA200']
        
        for feature in features:
            if feature not in data:
                return JSONResponse(status_code=400, content={"error": f"Hiányzó adat: {feature}"})
                
        df_live = pd.DataFrame([data], columns=features)
        
        # ML Számítás - Kérünk egy jóslatot a géptől
        prob_up = rf_model.predict_proba(df_live)[0][1]
        action = data.get('action', 'UNKNOWN')
        price = data['Close']
        
        print(f"📡 Piaci jelzés érkezett: {action} @ Ár: {price}")
        print(f"🧠 Modell számítása: {prob_up*100:.1f}% esély a Fel irányra.")
        
        # ÉLES LOGIKA: Csak akkor riasztunk, ha a gép szerint > 55% az esély
        if prob_up >= 0.55 and action == 'LONG':
            msg = f"🟢 <b>NQ LONG JELZÉS!</b>\nÁr: {price}\nML Esély (Fel): {prob_up*100:.1f}%\n<i>A TA setup és az ML modell egyetért.</i>"
            send_telegram_message(msg)
            
        elif prob_up <= 0.45 and action == 'SHORT':
            prob_down = 1 - prob_up
            msg = f"🔴 <b>NQ SHORT JELZÉS!</b>\nÁr: {price}\nML Esély (Le): {prob_down*100:.1f}%\n<i>A TA setup és az ML modell egyetért.</i>"
            send_telegram_message(msg)
        else:
            print("❌ Az ML modell megvétózta a jelzést. Nem küldünk Telegramot.")
            
        return {"status": "success", "ml_probability_up": float(prob_up)}

    except Exception as e:
        print(f"Hiba történt a webhooknál: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/")
def read_root():
    return {"status": "A szerver online, az ML modell várja a jeleket!"}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
