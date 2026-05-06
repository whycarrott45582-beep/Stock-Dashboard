import streamlit as st
import yfinance as yf
import pandas as pd

st.set_page_config(page_title="Biotech & Tech Dashboard", layout="wide")
st.title("📈 Stock Analysis: Biotech & Tech")

# สร้างตัวเลือกหุ้นจากกลุ่มที่คุณสนใจ
tickers = ["CRSP", "NVDA", "TSM", "FCX", "REGN", "NTLA"]
selected_tickers = st.multiselect("เลือกหุ้นที่ต้องการวิเคราะห์", tickers, default=["CRSP", "NVDA"])

# ดึงข้อมูลย้อนหลัง 1 ปี
if selected_tickers:
    # เพิ่ม group_by='column' เพื่อให้จัดการข้อมูลง่ายขึ้น
    raw = yf.download(selected_tickers, period="1y", group_by='column')
    
    if not raw.empty:
        # แก้ปัญหา KeyError โดยการเลือกดึงเฉพาะราคา Close ของทุกหุ้น
        # โครงสร้างใหม่ของ yfinance จะเก็บราคาไว้ที่ ['Close', 'Ticker']
        data = raw['Close']
        
        # คำนวณ Technical Indicators แบบใช้ Pandas ปกติ (เลี่ยงปัญหา Numba)
        st.subheader("กราฟราคาและเส้นค่าเฉลี่ย (EMA)")
        
        for ticker in selected_tickers:
            # ถ้าเลือกหุ้นหลายตัว data จะเป็น DataFrame ถ้าตัวเดียวจะเป็น Series
            col_data = data[ticker] if len(selected_tickers) > 1 else data
            
            # สร้างตารางสำหรับแสดงผลรายตัว
            df_display = pd.DataFrame(index=col_data.index)
            df_display['Price'] = col_data
            df_display['EMA20'] = col_data.ewm(span=20, adjust=False).mean()
            df_display['EMA50'] = col_data.ewm(span=50, adjust=False).mean()
            
            with st.expander(f"วิเคราะห์หุ้น {ticker}"):
                st.line_chart(df_display)
                last_price = col_data.iloc[-1]
                st.write(f"ราคาล่าสุด: ${last_price:.2f}")
    else:
        st.error("ไม่สามารถดึงข้อมูลได้ โปรดตรวจสอบการเชื่อมต่ออินเทอร์เน็ต")
else:
    st.info("โปรดเลือกหุ้นอย่างน้อย 1 ตัวเพื่อเริ่มการวิเคราะห์")
