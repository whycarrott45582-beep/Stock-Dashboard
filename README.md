# Stock Dashboard: วิเคราะห์หุ้น 3 กลุ่ม

Dashboard ที่ใช้ Streamlit สำหรับวิเคราะห์หุ้นจากกลุ่ม Biotech, Tech-semicon และ Mining ด้วยข้อมูลจาก SET สำหรับหุ้นไทย และ yfinance สำหรับหุ้นต่างประเทศ พร้อมตัวชี้วัดทางเทคนิคและ Sentiment Analysis

## 🚀 ฟีเจอร์หลัก

1. **Price Comparison** - เปรียบเทียบราคาหุ้นจากกลุ่มต่างๆ
2. **Technical Indicators** - EMA 20, EMA 50 และ RSI เพื่อหาสัญญาณซื้อขาย
3. **Sentiment Analysis** - วิเคราะห์ข่าวล่าสุด และประเมิน Sentiment เป็นบวก/ลบ

## 📋 สินค้า (Stock Tickers)

### Biotech Group
CRSP, TEM, JNJ, REGN, CMPS, ATAI, NTLA, PFE, BEAM, UTHR, MDGL, LLY

### Tech-Semiconductor Group
NVDA, TSMC, ASML, MSFT, META, AAPL, AMZN, GOOG, TSLA

### Mining Group
FCX, AG, RIO

## 🔧 วิธีการติดตั้ง

### 1. Clone หรือ Download โปรเจค
```bash
cd stock-dashboard
```

### 2. สร้าง Virtual Environment (ตัวเลือก)
```bash
python -m venv venv
venv\Scripts\activate  # Windows
```

### 3. ติดตั้ง Dependencies
```bash
pip install -r requirements.txt
```

### 4. ตั้งค่า Yahoo Finance API Key
- ใช้ API Key ของ Yahoo Finance ผ่าน RapidAPI
- เปิดไฟล์ `.streamlit/secrets.toml`
- ตั้งค่าตามตัวอย่าง

```toml
YAHOO_FINANCE_API_KEY = "Sh2JBbbtk2HCBGenvZt8JPpFJMoHB73u"
```

- หากไม่ตั้งค่าไว้ แอปจะใช้ค่าเริ่มต้นที่กำหนดไว้ในโค้ด

## ▶️ วิธีการใช้งาน

```bash
streamlit run app.py
```

Dashboard จะเปิดใน Browser ที่ `http://localhost:8501`

## 📊 ตัวบ่งชี้ทางเทคนิค (Technical Indicators)

- **EMA 20** - Moving Average 20 วัน
- **EMA 50** - Moving Average 50 วัน
- **RSI** - Relative Strength Index (14)

### สัญญาณซื้อขาย:
- 🟢 EMA 20 > EMA 50: Uptrend
- 🔴 EMA 20 < EMA 50: Downtrend
- 🟢 RSI < 30: Oversold (Opportunity to buy)
- 🔴 RSI > 70: Overbought (Opportunity to sell)

## 🤖 Sentiment Analysis

Dashboard ใช้ Hugging Face Transformers (distilbert-base-uncased-finetuned-sst-2-english) เพื่อวิเคราะห์ข่าวโดย:

1. ดึงข่าวล่าสุด (5 บทความ) จาก News API
2. วิเคราะห์ Sentiment ของแต่ละบทความ
3. รวบรวม Score และแสดงผลสรุป
4. แบ่งเป็นกลุ่ม: Positive 🟢, Negative 🔴, Neutral ⚪

## 📦 Dependencies

- **Streamlit** - Web Framework
- **yfinance** - ดึงข้อมูลหุ้น
- **pandas** - Data Manipulation
- **plotly** - วาดกราฟ
- **nltk** - วิเคราะห์ Sentiment และ Text
- **requests** - ดึงข่าวจาก News API

## 🇹🇭 รองรับหุ้นไทย

- หุ้นไทยจะดึงข้อมูลราคาจาก SET โดยตรง
- มีกลุ่ม `SET (ตลาดหุ้นไทย)` สำหรับหุ้นไทยในตลาดหลักทรัพย์ SET
- ใช้ ticker รูปแบบ `.BK` เช่น `PTT.BK`, `AOT.BK`, `SCB.BK`
- หากกรอก `PTT`, `AOT`, `SCB` ฯลฯ แอปจะช่วยเติม `.BK` ให้อัตโนมัติ

## 🚀 Deployment

1. **Streamlit Cloud**
   - สร้าง repository บน GitHub
   - push โปรเจคนี้ขึ้น GitHub
   - เปิด https://share.streamlit.io และเชื่อมต่อกับ repository
   - เลือกไฟล์ `app.py`

2. **Heroku**
   - สร้างแอปใหม่ใน Heroku
   - เชื่อม GitHub repository หรือ push ด้วย `git push heroku main`
   - Heroku จะใช้งาน `Procfile` ที่มีคำสั่ง:
     ```text
     web: streamlit run app.py --server.port $PORT --server.address 0.0.0.0
     ```

3. **Local**
   - ติดตั้ง deps แล้วรันด้วย:
     ```bash
     streamlit run app.py
     ```

> หากจะส่งให้คนอื่นใช้งาน ให้ส่ง link GitHub หรือ link Streamlit Cloud ที่ deploy แล้ว

## ⚠️ หมายเหตุ

- Technical Indicators จะแสดงเฉพาะเมื่อเลือกหุ้นเพียง 1 ตัว
- Sentiment Analysis ต้อง News API Key จึงจะทำงาน
- ข้อมูลราคาอาจใช้เวลารอบ้างเมื่อโหลดหลายตัวหุ้นพร้อมกัน
- ข้อมูล Sentiment ขึ้นอยู่กับคุณภาพของข่าวจาก News API

## 📝 License

Free to use for personal and educational purposes.

## 💡 ข้อแนะนำ

- ใช้ Time Range ที่เหมาะสม (ข้อมูล 1 ปี หรือมากกว่า ให้ผลดีกว่า)
- ดูตัวชี้วัดหลายตัวร่วมกัน ไม่ใช่เพียงตัวเดียว
- รวมข้อมูล Sentiment และราคาเพื่อการตัดสินใจที่ดีขึ้น
- อย่าลืมว่านี่เป็นเพียงเครื่องมือช่วย ไม่ใช่คำแนะนำการลงทุน

---

**Happy Analyzing! 📈**
