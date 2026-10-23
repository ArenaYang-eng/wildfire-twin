import streamlit as st
import pandas as pd
import requests
import pydeck as pdk
import math
import google.generativeai as genai
import base64
from concurrent.futures import ThreadPoolExecutor
from PIL import Image
from io import BytesIO
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
        if "Invalid" in res.text or res.status_code != 200:
            st.sidebar.warning(f"⚠️ NASA API 回應：{res.text.strip()}（已自動切換為高擬真演練數據）")
            return mock_fires
        
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
    except Exception:
        return 15.0, 210.0 # 預設西南季風

# ==========================================
# 4. 數值模擬層：煙羽時空漂移演算法 (MaaS)
# ==========================================
def calculate_smoke_advection(fires, wind_spd, wind_dir):
    particles = []
    downwind_deg = (wind_dir + 180) % 360
    rad = math.radians(downwind_deg)
    
    for _, row in fires.iterrows():
        lat = row['latitude']
        lon = row['longitude']
        frp = row.get('frp', 50.0)
        
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


# ==========================================
# 4.5 衛星底圖：伺服器端拼接 Esri World Imagery 圖磚 -> 單張 BitmapLayer
#     (st.pydeck_chart 的前端不支援 TileLayer 的自訂圖磚渲染，故改用此法)
# ==========================================
ESRI_TILE = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"

def _lon2x(lon, z):
    return (lon + 180.0) / 360.0 * 2 ** z

def _lat2y(lat, z):
    return (1 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2 * 2 ** z

def _x2lon(x, z):
    return x / 2 ** z * 360.0 - 180.0

def _y2lat(y, z):
    return math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * y / 2 ** z))))

def _fetch_tile(z, x, y):
    for _ in range(2):  # 失敗重試一次
        try:
            r = requests.get(
                ESRI_TILE.format(z=z, x=x, y=y),
                timeout=8,
                headers={"User-Agent": "Mozilla/5.0 (streamlit-digital-twin)"},
            )
            if r.status_code == 200 and r.content:
                return Image.open(BytesIO(r.content)).convert("RGB")
        except Exception:
            pass
    return None

@st.cache_data(ttl=7 * 86400, show_spinner="🛰️ 正在載入衛星空照底圖（首次約需數秒）...")
def build_satellite_mosaic(west, south, east, north, z):
    """把範圍內的 XYZ 圖磚拼成一張圖，回傳 (data URI, [W,S,E,N] 精確邊界, 成功磚數, 總磚數)"""
    x0, x1 = int(_lon2x(west, z)), int(_lon2x(east, z))
    y0, y1 = int(_lat2y(north, z)), int(_lat2y(south, z))
    jobs = [(x, y) for y in range(y0, y1 + 1) for x in range(x0, x1 + 1)]
    canvas = Image.new("RGB", ((x1 - x0 + 1) * 256, (y1 - y0 + 1) * 256), (30, 34, 44))
    with ThreadPoolExecutor(8) as ex:
        tiles = list(ex.map(lambda xy: _fetch_tile(z, xy[0], xy[1]), jobs))
    ok = 0
    for (x, y), t in zip(jobs, tiles):
        if t is not None:
            canvas.paste(t, ((x - x0) * 256, (y - y0) * 256))
            ok += 1
    bounds = [_x2lon(x0, z), _y2lat(y1 + 1, z), _x2lon(x1 + 1, z), _y2lat(y0, z)]
    buf = BytesIO()
    canvas.save(buf, "JPEG", quality=82)
    uri = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
    return uri, bounds, ok, len(jobs)

def make_bitmap_layer(layer_id, mosaic):
    uri, bounds, ok, total = mosaic
    return pdk.Layer(
        "BitmapLayer",
        id=layer_id,
        data=None,
        # 外層單引號 = 告訴 pydeck「這是字串常數，不要當成運算式」
        image="'" + uri + "'",
        bounds=bounds,
        opacity=1.0,
        parameters={"depthTest": False},
    )

# 廣域低解析 (整個東南亞，縮小時不會露出灰黑) + 近景高解析 (依所選區域)
if "蘇門答臘" in region_choice:
    near_box = (93.0, -9.0, 113.0, 9.0)
else:
    near_box = (104.0, -9.0, 123.0, 8.0)

mosaic_wide = build_satellite_mosaic(88.0, -14.0, 128.0, 16.0, 5)
mosaic_near = build_satellite_mosaic(*near_box, 7)   # 想放大看更細：把 7 改成 8（圖磚量約 4 倍）

# ==========================================
# 5. 展示層：Esri 衛星空照底圖 + 中英雙語地名標籤
# ==========================================
col_map, col_ai = st.columns([65, 35])

with col_map:
    st.subheader("🌐 即時 3D 空間態勢展示（真實衛星空照影像）")

    base_layers = []
    for lid, m in (("sat-wide", mosaic_wide), ("sat-near", mosaic_near)):
        if m[2] > 0:
            base_layers.append(make_bitmap_layer(lid, m))
    if not base_layers:
        st.warning("⚠️ 無法連線 Esri 圖磚伺服器，暫時顯示預設底圖。")
    elif mosaic_near[2] < mosaic_near[3]:
        st.caption(f"ℹ️ 近景影像有 {mosaic_near[3] - mosaic_near[2]} 塊圖磚載入失敗（以深灰色補位），可重新整理重試。")

    # 煙羽動態擴散層
    smoke_layer = pdk.Layer(
        "ScatterplotLayer",
        id="smoke-plumes",
        data=smoke_data,
        get_position=["lon", "lat"],
        get_radius="density * 800",
        get_fill_color=[240, 220, 160, 120],
        pickable=True,
        parameters={"depthTest": False},  # 圓圈互相重疊時不再出現方塊狀破碎
    )

    # 3D 火點柱狀圖層
    fire_layer = pdk.Layer(
        "ColumnLayer",
        id="fire-columns",
        data=fire_data,
        get_position=["longitude", "latitude"],
        get_elevation="frp * 250",
        elevation_scale=1,
        radius=4000,
        get_fill_color=[255, 30, 0, 230],
        pickable=True,
    )

    # 中英雙語標註：(中文, 英文, 緯度, 經度, 類型)
    GOLD, CYAN, ORANGE, WHITE, ICE = [255, 235, 100], [0, 255, 255], [255, 180, 0], [255, 255, 255], [200, 240, 255]
    label_rows = [
        # 國家
        ("印尼", "Indonesia", -4.2, 104.0, 17, GOLD),
        ("印尼", "Indonesia", -0.8, 112.0, 17, GOLD),
        ("馬來西亞", "Malaysia", 5.2, 102.6, 17, GOLD),
        ("馬來西亞", "Malaysia", 2.6, 113.0, 17, GOLD),
        ("汶萊", "Brunei", 4.6, 114.7, 15, GOLD),
        ("泰國", "Thailand", 8.0, 99.8, 17, GOLD),
        ("新加坡", "Singapore", 1.3521, 103.8198, 15, CYAN),
        # 島嶼 / 區域
        ("蘇門答臘", "Sumatra", -1.6, 102.0, 15, WHITE),
        ("婆羅洲", "Borneo", 1.0, 114.0, 15, WHITE),
        ("爪哇", "Java", -7.4, 110.0, 15, WHITE),
        ("亞齊", "Aceh", 4.6, 96.9, 12, ICE),
        ("廖內", "Riau", 1.2, 101.9, 12, ICE),
        # 海域
        ("麻六甲海峽", "Strait of Malacca", 4.3, 99.3, 14, [100, 230, 255]),
        ("南海", "South China Sea", 5.5, 108.5, 14, [100, 230, 255]),
        ("爪哇海", "Java Sea", -5.0, 110.5, 14, [100, 230, 255]),
        ("印度洋", "Indian Ocean", -4.0, 96.0, 14, [100, 230, 255]),
        ("卡里馬塔海峽", "Karimata Strait", -2.0, 108.4, 12, [100, 230, 255]),
        # 國際都市（下風處）
        ("吉隆坡", "Kuala Lumpur", 3.1390, 101.6869, 14, CYAN),
        ("馬六甲", "Malacca", 2.1896, 102.2501, 13, CYAN),
        ("檳城", "Penang", 5.4141, 100.3288, 13, CYAN),
        ("雅加達", "Jakarta", -6.2088, 106.8456, 14, CYAN),
        # 火場周邊城市
        ("巨港", "Palembang", -2.9761, 104.7754, 14, ORANGE),
        ("北干巴魯", "Pekanbaru", 0.5071, 101.4478, 14, ORANGE),
        ("占碑", "Jambi", -1.6101, 103.6131, 14, ORANGE),
        ("班達楠榜", "Bandar Lampung", -5.4292, 105.2610, 13, ORANGE),
        ("坤甸", "Pontianak", -0.0263, 109.3425, 14, ORANGE),
        ("帕朗卡拉亞", "Palangka Raya", -2.2161, 113.9135, 14, ORANGE),
        ("馬辰", "Banjarmasin", -3.3194, 114.5908, 13, ORANGE),
        ("棉蘭", "Medan", 3.5952, 98.6722, 13, ICE),
        ("巴東", "Padang", -0.9471, 100.4172, 13, ICE),
        ("明古魯", "Bengkulu", -3.8004, 102.2655, 12, ICE),
    ]
    bilingual_labels = pd.DataFrame(
        [{"name": f"{zh} {en}", "lat": la, "lon": lo, "size": sz, "color": col}
         for zh, en, la, lo, sz, col in label_rows]
    )

    text_layer = pdk.Layer(
        "TextLayer",
        id="city-labels",
        data=bilingual_labels,
        get_position=["lon", "lat"],
        get_text="name",
        get_size="size",
        get_color="color",
        get_angle=0,
        get_text_anchor="'middle'",
        get_alignment_baseline="'center'",
        # 關鍵修正 1：字型清單必須整串用單引號包成「字串常數」，且清單內不可再有引號
        font_family="'Microsoft JhengHei, PingFang TC, Noto Sans CJK TC, Noto Sans TC, sans-serif'",
        # 關鍵修正 2：不要傳字元 list（會被當運算式而解析失敗），直接讓 deck.gl 依資料自動產生字集
        character_set="'auto'",
        background=True,
        get_background_color=[15, 20, 30, 190],
        background_padding=[4, 2, 4, 2],
        parameters={"depthTest": False},  # 文字永遠畫在最上層，不被煙羽圈蓋住
    )

    deck = pdk.Deck(
        layers=base_layers + [smoke_layer, fire_layer, text_layer],
        map_style=None,
        initial_view_state=pdk.ViewState(
            latitude=center_lat,
            longitude=center_lon,
            zoom=5.8,
            pitch=45,
            bearing=10
        ),
        tooltip={
            "html": "<b>火勢劇烈程度 FRP:</b> {frp} MW<br/><b>煙塵濃度指標:</b> {density}",
            "style": {"backgroundColor": "rgba(20, 25, 35, 0.9)", "color": "white"}
        }
    )

    st.pydeck_chart(deck)
    st.caption("底圖影像：Esri, Maxar, Earthstar Geographics, and the GIS User Community")

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
