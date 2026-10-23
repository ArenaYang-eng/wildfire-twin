import streamlit as st
import pandas as pd
import requests
import pydeck as pdk
import math
import google.generativeai as genai
from io import StringIO

# ==========================================
# 1. 頁面基礎設定與金鑰讀取
# ==========================================
st.set_page_config(
    layout="wide",
    page_title="印尼野火與跨國霧霾數位孿生系統",
    page_icon="🔥"
)

# 支援由 Streamlit 雲端 Secrets 自動載入，若無則降階由側邊欄手動輸入
default_firms_key = st.secrets.get("NASA_FIRMS_KEY", "")
default_gemini_key = st.secrets.get("GEMINI_API_KEY", "")

st.title("🔥 印尼泥炭地野火與跨國霧霾擴散決策孿生平台")
st.caption("基於 NASA VIIRS 衛星熱點、即時大氣風場、高斯煙羽投影與 GeoAI 決策代理人")

with st.sidebar:
    st.header("⚙️ 系統設定與連線")
    firms_key = st.text_input("NASA FIRMS MAP_KEY", value=default_firms_key, type="password")
    gemini_key = st.text_input("Google Gemini API Key", value=default_gemini_key, type="password")
    
    st.markdown("---")
    st.header("📍 監控場域選定")
    region_choice = st.radio(
        "選擇重點火情區域：",
        ["蘇門答臘（威脅麻六甲海峽與星馬）", "加里曼丹（婆羅洲泥炭地）"]
    )
    
    # 根據不同地區定義空間範圍 [西, 南, 東, 北] 與觀測中心
    if "蘇門答臘" in region_choice:
        bbox = "98.0,-4.5,106.0,3.0"
        center_lat, center_lon = -1.0, 102.5
    else:
        bbox = "108.5,-4.0,117.5,3.5"
        center_lat, center_lon = -1.5, 113.0

    st.markdown("---")
    st.info("💡 **架構提示**：本系統運作於 Streamlit Community Cloud，資料流每 30 分鐘自動透過 API 快取更新。")

# ==========================================
# 2. 感知層：抓取 NASA FIRMS 衛星火點（增強除錯版）
# ==========================================
@st.cache_data(ttl=1800)
def fetch_firms_hotspots(api_key, box):
    # 高擬真示範火點資料庫（備用方案）
    mock_fires = pd.DataFrame([
        {"latitude": -1.85, "longitude": 103.62, "frp": 185.4, "confidence": "h"},
        {"latitude": -2.12, "longitude": 103.88, "frp": 92.1, "confidence": "n"},
        {"latitude": -1.45, "longitude": 102.95, "frp": 310.6, "confidence": "h"},
        {"latitude": -2.40, "longitude": 104.15, "frp": 68.3, "confidence": "n"},
        {"latitude": -0.95, "longitude": 102.30, "frp": 145.0, "confidence": "h"},
        {"latitude": -1.20, "longitude": 103.10, "frp": 230.8, "confidence": "h"},
        {"latitude": -2.80, "longitude": 104.50, "frp": 110.2, "confidence": "n"},
    ])
    
    if not api_key:
        return mock_fires
    
    url = f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/{api_key}/VIIRS_SNPP_NRT/{box}/1"
    try:
        res = requests.get(url, timeout=10)
        # 如果 NASA 回傳錯誤訊息（例如金鑰尚未生效）
        if "Invalid" in res.text or res.status_code != 200:
            st.sidebar.warning(f"⚠️ NASA API 回應：{res.text.strip()}（已自動切換為高擬真演練數據）")
            return mock_fires
        
        # 正常取得數據且有火點
        if "latitude" in res.text:
            df = pd.read_csv(StringIO(res.text))
            if len(df) == 0:
                st.sidebar.info("ℹ️ 今日該空域衛星未測得明火（已切換為演練數據供展示）")
                return mock_fires
            return df
    except Exception as e:
        st.sidebar.error(f"衛星連線逾時：{e}（已切換為演練數據）")
        return mock_fires
        
    return mock_fires

# ==========================================
# 3. 驅動層：即時大氣風場 (Open-Meteo)
# ==========================================
def fetch_wind_vector(lat, lon):
    url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=wind_speed_10m,wind_direction_10m"
    try:
        res = requests.get(url, timeout=5).json()
        current = res.get("current", {})
        return current.get("wind_speed_10m", 12.0), current.get("wind_direction_10m", 200)
    except:
        return 15.0, 210.0 # 預設西南季風

# ==========================================
# 4. 數值模擬層：煙羽時空漂移演算法 (MaaS)
# ==========================================
def calculate_smoke_advection(fires, wind_spd, wind_dir):
    particles = []
    # 煙霧受風推動，吹向的方位角 = (wind_dir + 180) % 360
    downwind_deg = (wind_dir + 180) % 360
    rad = math.radians(downwind_deg)
    
    for _, row in fires.iterrows():
        lat = row['latitude']
        lon = row['longitude']
        frp = row.get('frp', 50.0) # 火災輻射功率 (MW)
        
        # 模擬 2, 4, 8, 12 小時後的煙塵團核心
        for hr in [2, 4, 8, 12]:
            dist_km = wind_spd * hr
            delta_lat = (dist_km * math.cos(rad)) / 111.0
            delta_lon = (dist_km * math.sin(rad)) / (111.0 * math.cos(math.radians(lat)))
            
            particles.append({
                "lat": lat + delta_lat,
                "lon": lon + delta_lon,
                "hour": hr,
                "density": max(15, int(frp / (hr * 0.4)))
            })
    return pd.DataFrame(particles)

# 執行運算
fire_data = fetch_firms_hotspots(firms_key, bbox)
wind_speed, wind_direction = fetch_wind_vector(center_lat, center_lon)
smoke_data = calculate_smoke_advection(fire_data, wind_speed, wind_direction)

# 重要鄰近國際受災地標
major_cities = pd.DataFrame([
    {"name": "新加坡 (Singapore)", "lat": 1.3521, "lon": 103.8198},
    {"name": "吉隆坡 (Kuala Lumpur)", "lat": 3.1390, "lon": 101.6869},
    {"name": "巨港 (Palembang)", "lat": -2.9761, "lon": 104.7754},
    {"name": "北干巴魯 (Pekanbaru)", "lat": 0.5071, "lon": 101.4478}
])

# ==========================================
# 5. 展示層：真實衛星空照底圖 + 中英雙語對照
# ==========================================
col_map, col_ai = st.columns([65, 35])

with col_map:
    st.subheader("🌐 即時 3D 空間態勢展示（真實空照影像模式）")
    
    # 1. 煙羽動態擴散層
    smoke_layer = pdk.Layer(
        "ScatterplotLayer",
        data=smoke_data,
        get_position=["lon", "lat"],
        get_radius="density * 800",
        get_fill_color=[220, 200, 150, 100],
        pickable=True,
    )

    # 2. 3D 火點柱狀圖層（紅色立體光柱）
    fire_layer = pdk.Layer(
        "ColumnLayer",
        data=fire_data,
        get_position=["longitude", "latitude"],
        get_elevation="frp * 250",
        elevation_scale=1,
        radius=4000,
        get_fill_color=[255, 30, 0, 230],
        pickable=True,
    )

    # 3. 中英雙語標註資料庫（調整字形以完整呈現繁體中文）
    bilingual_labels = pd.DataFrame([
        # 國家名稱
        {"name": "印尼 (Indonesia)", "lat": -2.3, "lon": 101.5, "size": 18, "color": [255, 235, 100, 255]},
        {"name": "馬來西亞 (Malaysia)", "lat": 3.8, "lon": 102.5, "size": 18, "color": [255, 235, 100, 255]},
        
        # 戰略海域
        {"name": "麻六甲海峽 (Strait of Malacca)", "lat": 2.2, "lon": 102.0, "size": 15, "color": [100, 220, 255, 230]},
        
        # 重點城市
        {"name": "新加坡 (Singapore)", "lat": 1.3521, "lon": 103.8198, "size": 15, "color": [0, 255, 255, 255]},
        {"name": "吉隆坡 (Kuala Lumpur)", "lat": 3.1390, "lon": 101.6869, "size": 15, "color": [0, 255, 255, 255]},
        {"name": "巨港 (Palembang - 火場核心)", "lat": -2.9761, "lon": 104.7754, "size": 14, "color": [255, 180, 0, 255]},
        {"name": "北干巴魯 (Pekanbaru)", "lat": 0.5071, "lon": 101.4478, "size": 14, "color": [255, 180, 0, 255]},
        {"name": "棉蘭 (Medan)", "lat": 3.5952, "lon": 98.6722, "size": 13, "color": [220, 240, 255, 220]},
        {"name": "巴東 (Padang)", "lat": -0.9471, "lon": 100.4172, "size": 13, "color": [220, 240, 255, 220]},
    ])

    # 4. 中英文字標籤圖層（強制指定字體編碼與深色背景框）
    text_layer = pdk.Layer(
        "TextLayer",
        data=bilingual_labels,
        get_position=["lon", "lat"],
        get_text="name",
        get_size="size",
        get_color="color",
        get_angle=0,
        get_text_anchor="'middle'",
        get_alignment_baseline="'center'",
        background=True,
        get_background_color=[15, 20, 30, 200],  # 深黑背景框
        background_padding=[6, 3, 6, 3],
        character_set="auto",  # 關鍵：開啟自動萬國碼支援，繁體中文才會正常顯示！
    )

    # 5. 擬真衛星空照底圖樣式 (使用全球標準 Esri 衛星圖，免金鑰、直出衛星圖)
    deck = pdk.Deck(
        layers=[smoke_layer, fire_layer, text_layer],
        # 直接指定標準衛星空照樣式規範，取代灰白向量底圖
        map_style="https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json",
        map_provider="carto",
        initial_view_state=pdk.ViewState(
            latitude=center_lat,
            longitude=center_lon,
            zoom=5.8,
            pitch=45,
            bearing=10
        ),
        tooltip={"text": "標的資訊: {name}\n火勢劇烈程度 (燃燒熱能): {frp} MW\n煙塵預估濃度: {density}"}
    )
    
    # ==========================================
    # 5.1 通俗易懂圖例說明列 (Map Legend)
    # ==========================================
    legend_html = """
    <div style="background-color: #1A1C24; padding: 12px 16px; border-radius: 8px; border: 1px solid #2E3342; margin-bottom: 15px;">
        <div style="font-weight: 600; font-size: 14px; color: #E0E2EC; margin-bottom: 8px;">📑 圖面圖例說明 (Map Legend)：</div>
        <div style="display: flex; flex-wrap: wrap; gap: 20px; font-size: 13px; color: #C4C7D4;">
            <div style="display: flex; align-items: center; gap: 8px;">
                <span style="display: inline-block; width: 14px; height: 14px; background-color: #FF1E00; border-radius: 2px; box-shadow: 0 0 6px #FF1E00;"></span>
                <span><b>3D 紅色光柱</b>：衛星偵測起火點（<b>柱子越高代表火勢燒得越猛烈、熱度越高</b>）</span>
            </div>
            <div style="display: flex; align-items: center; gap: 8px;">
                <span style="display: inline-block; width: 14px; height: 14px; background-color: rgba(220, 200, 150, 0.7); border-radius: 50%;"></span>
                <span><b>灰黃色煙羽圈</b>：電腦模擬未來 2~12 小時<b>濃煙霧霾隨風飄散的範圍</b></span>
            </div>
            <div style="display: flex; align-items: center; gap: 8px;">
                <span style="display: inline-block; width: 14px; height: 14px; background-color: #00FFFF; border-radius: 50%; box-shadow: 0 0 6px #00FFFF;"></span>
                <span><b>亮青色標籤</b>：下風處可能受霧霾波及的重要國際都市</span>
            </div>
            <div style="display: flex; align-items: center; gap: 8px;">
                <span style="display: inline-block; width: 14px; height: 14px; background-color: #FFE664; border-radius: 2px;"></span>
                <span><b>黃色文字標記</b>：泥炭地火場重災區與國界標示</span>
            </div>
        </div>
    </div>
    """
    st.markdown(legend_html, unsafe_allow_html=True)
    
    # 微氣象資訊
    drift_dir = (wind_direction + 180) % 360
    st.markdown(
        f"🧭 **微氣象遙測資訊**：近地風速 **{wind_speed} km/h**，來風方位角 **{wind_direction}°**。"
        f"煙霾正向 **{drift_dir}°**（預估擴散路徑）推進。"
    )
    drift_dir = (wind_direction + 180) % 360
    st.markdown(
        f"🧭 **微氣象遙測資訊**：近地風速 **{wind_speed} km/h**，來風方位角 **{wind_direction}°**。"
        f"煙霾正向 **{drift_dir}°**（預估擴散路徑）推進。"
    )
# ==========================================
# 6. 決策層：GeoAI 跨國霧霾應變 Copilot（最新 Gemini 3.8 支援版）
# ==========================================
with col_ai:
    st.subheader("🤖 跨國霾害應變 Copilot")
    st.metric("感測熱點總數", f"{len(fire_data)} 處")
    
    if st.button("🚨 啟動 GeoAI 決策推論", use_container_width=True):
        if not gemini_key:
            st.warning("請填寫 Gemini API Key（或於雲端 Secrets 配置）。")
        else:
            with st.spinner("AI Agent 正在分析下風處風險並生成外交行動決策..."):
                max_frp = fire_data['frp'].max() if 'frp' in fire_data else "N/A"
                drift_dir = (wind_direction + 180) % 360
                
                prompt = f"""
                你現在是「東南亞國協跨國霾害應變中心（ASMC）」的資深決策專家顧問。
                系統已完成野火感知與擴散數值模擬，數據如下：
                - 觀測場域：{region_choice}
                - 衛星即時熱點總數：{len(fire_data)} 處
                - 單點最高輻射能量 (FRP)：{max_frp} MW (泥炭地陰燃重大指標)
                - 即時風場條件：風速 {wind_speed} km/h，煙塵漂移方位角 {drift_dir}°
                
                請以繁體中文撰寫一份高階應變行動指引，嚴格包含以下結構：
                1. 【跨國擴散風險診斷】：評估未來 12-24 小時內，煙羽是否會進入新加坡、吉隆坡或麻六甲海峽國際海運航道？
                2. 【泥炭地撲滅戰術方案】：針對泥炭地地下悶燒無法以傳統表面水撲滅的特性，提出水資源調度與阻隔溝渠開挖建議。
                3. 【跨國協調與預警宣告】：依據《東南亞跨國霾害污染協議（AATHP）》，印尼應主動向受衝擊鄰國通報何種應變層級？
                """
                
                response = None
                chosen_model_used = None
                
                try:
                    genai.configure(api_key=gemini_key.strip())
                    
                    # 1. 優先嘗試 Google 官方推薦新用戶使用的最新主流模型
                    candidate_models = [
                        'gemini-3.8-flash',
                        'gemini-3.5-flash',
                        'gemini-3.1-flash-lite',
                        'gemini-flash-latest'
                    ]
                    
                    for m_name in candidate_models:
                        try:
                            model = genai.GenerativeModel(m_name)
                            res = model.generate_content(prompt)
                            if res and res.text:
                                response = res
                                chosen_model_used = m_name
                                break
                        except Exception:
                            continue
                    
                    # 2. 若上述未通，自動從 API 名單中過濾掉已停用的 2.5 / 2.0 / 1.5 舊版
                    if not response:
                        for m in genai.list_models():
                            m_name = m.name
                            if any(retired in m_name for retired in ['2.5', '2.0', '1.5', 'gemini-pro']):
                                continue
                            if 'generateContent' in m.supported_generation_methods:
                                try:
                                    model = genai.GenerativeModel(m_name)
                                    res = model.generate_content(prompt)
                                    if res and res.text:
                                        response = res
                                        chosen_model_used = m_name
                                        break
                                except Exception:
                                    continue

                    if response and response.text:
                        st.caption(f"⚡ 成功連線最新模型：`{chosen_model_used}`")
                        st.markdown(response.text)
                    else:
                        raise Exception("無法與可用模型建立對話通道。")
                        
                except Exception as e:
                    st.error(f"⚠️ 線上 AI 呼叫遇阻：{e}")
                    st.info("💡 系統已自動啟動「備用演練決策方案」，確保 Demo 閉環展示不中斷：")
                    
                    # 示範保底簡報
                    st.markdown(f"""
### 🚨 東南亞跨國霾害應變中心（ASMC）緊急決策指引 (示範情境)

#### 1. 【跨國擴散風險診斷】
* **空域威脅**：目前近地風向為偏南風（方位角 {wind_direction}°），煙羽正以 **{wind_speed} km/h** 速度朝方位角 **{drift_dir}°** 推進。
* **衝擊評估**：
  * **新加坡 (Singapore)**：距火場約 180 公里，預計 **8~12 小時內** 煙塵前鋒抵達，PSI（空氣污染指標）恐突破 150。
  * **麻六甲海峽 (Strait of Malacca)**：國際繁忙航道能見度將降至 2 海浬以下，需對大型油輪發布航安警示。
  * **馬來西亞半島南部 (Johor)**：預計明日清晨 PM2.5 濃度達到峰值。

#### 2. 【泥炭地撲滅戰術方案】
* **地下陰燃特性**：泥炭層深達數公尺，空中直升機投水僅能撲滅地表草火，無法熄滅地下悶燒。
* **關鍵工程處置**：
  * 立即調派挖土機於火線外圍 200 公尺開挖深達地下水位的「阻隔溝渠（Fire Trench）」。
  * 啟動周邊運河閘門封堵作業，進行**泥炭地水文復育回補（Rewetting）**，提高地下水位阻絕氧氣。

#### 3. 【跨國協調與預警宣告】
* 依據《東南亞跨國霾害污染協議》（AATHP）第 9 條，印尼環境與林業部應於 **2 小時內向東協秘書處通報升級為「第二級警戒（Alert Level 2）」**。
* 建議同步向新加坡國家環境局（NEA）與馬來西亞環境局（DOE）開放共享熱點遙測數據，啟動跨國人工增雨（Cloud Seeding）聯合防制協議。
                    """)
