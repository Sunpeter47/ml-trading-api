from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import uvicorn
import joblib
import pandas as pd
import requests

app = FastAPI(title="NQ ML Trading Radar API")

# Betöltjük a betanított Random Forest modellt. 
# Ez a fájl ugyanabban a mappában kell legyen, mint ez a kód!
try:
    rf_model = joblib.load('pro_nq_model_5m.joblib')
    print("✅ ML Modell sikeresen betöltve!")
except Exception as e:
    print(f"❌ Hiba a modell betöltésekor: {e}")

# Ide fogjuk beírni a Te Telegram botod adatait a következő lépésben
TELEGRAM_TOKEN = "IDE_JON_A_TOKEN"
TELEGRAM_CHAT_ID = "IDE_JON_A_CHAT_ID"

def send_telegram_message(message: str):
    """Küld egy üzenetet a Telegramodra."""
    if TELEGRAM_TOKEN == "IDE_JON_A_TOKEN":
        print("A Telegram még nincs beállítva, de a jelzés a következő lenne:")
        print(message)
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
    """
    Ezt az útvonalat hívja meg a TradingView. 
    JSON formátumban várja a gyertya és az indikátorok adatait.
    """
    try:
        # 1. Adatcsomag fogadása a TradingView-tól
        data = await request.json()
        
        # 2. Átalakítjuk olyan formátummá (DataFrame), amit a modell megért
        # FONTOS: Az oszlopok nevének PONTOSAN egyeznie kell a tanításnál használtakkal!
        features = ['Close', 'RSI_14', 'SMA_20', 'Dist_to_Lower', 'Dist_to_Upper', 'SMA_200', 'Is_Uptrend', 'Dist_to_SMA200']
        
        # Ellenőrizzük, hogy minden adat megjött-e
        for feature in features:
            if feature not in data:
                return JSONResponse(status_code=400, content={"error": f"Hiányzó adat: {feature}"})
                
        # Létrehozzuk a táblázatot 1 sorral
        df_live = pd.DataFrame([data], columns=features)
        
        # 3. Kérünk egy jóslatot a géptől (Valószínűség)
        prob_up = rf_model.predict_proba(df_live)[0][1]
        
        # 4. Értékelés és Riasztás
        action = data.get('action', 'UNKNOWN') # Ezt is a TradingView küldi (LONG vagy SHORT jel)
        price = data['Close']
        
        # Csak akkor riasztunk, ha a gép szerint > 55% az esély
        if prob_up >= 0.55 and action == 'LONG':
            msg = f"🟢 <b>NQ LONG JELZÉS!</b>\nÁr: {price}\nML Esély (Fel): {prob_up*100:.1f}%"
            send_telegram_message(msg)
            
        elif prob_up <= 0.45 and action == 'SHORT':
            # Ha prob_up 45% alatt van, akkor a prob_down > 55% (Short esély)
            prob_down = 1 - prob_up
            msg = f"🔴 <b>NQ SHORT JELZÉS!</b>\nÁr: {price}\nML Esély (Le): {prob_down*100:.1f}%"
            send_telegram_message(msg)
            
        # Visszaszólunk a TradingView-nak, hogy megkaptuk
        return {"status": "success", "ml_probability_up": float(prob_up)}

    except Exception as e:
        print(f"Hiba történt: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/")
def read_root():
    """Ez csak arra jó, hogy a böngészőből lásd, él a szervered."""
    return {"status": "A szerver online, az ML modell várja a jeleket!"}

# Ez csak akkor fut, ha lokálisan teszteljük a gépeden
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)