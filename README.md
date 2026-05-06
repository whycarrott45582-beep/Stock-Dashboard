# Stock Dashboard: วิเคราะห์หุ้น 3 กลุ่ม

Dashboard ที่ใช้ Streamlit สำหรับวิเคราะห์หุ้นจากกลุ่ม Biotech, Tech-semicon และ Mining ด้วยข้อมูลจาก yfinance พร้อมตัวชี้วัดทางเทคนิคและ Sentiment Analysis

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

### 4. ตั้งค่า News API Key
- สมัครสมาชิก News API ที่ https://newsapi.org
- เปิดไฟล์ `.streamlit/secrets.toml`
- แทนที่ `your_api_key_here` ด้วย API Key ของคุณ

```toml
NEWS_API_KEY = "your_actual_api_key"
```

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
- **pandas_ta** - Technical Analysis
- **requests** - HTTP Library สำหรับ News API
- **transformers** - AI Model สำหรับ Sentiment Analysis
- **torch** - Deep Learning Framework

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
