import streamlit as st
import pandas as pd
import requests
import pydeck as pdk
import math
import google.generativeai as genai
import base64
import numpy as np
from datetime import datetime, timezone
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
# 從 Secrets 讀取金鑰，不放進前端輸入框
firms_key = st.secrets.get("NASA_FIRMS_KEY", "")
gemini_key = st.secrets.get("GEMINI_API_KEY", "")

st.title("🔥 印尼泥炭地野火與跨國霧霾擴散決策孿生平台")
st.caption(
    "基於 NASA VIIRS 衛星熱點、Open-Meteo 逐時風場預報（最長 7 天）、"
    "拉格朗日煙團軌跡模擬與 GeoAI 決策代理人"
)

with st.sidebar:
    st.header("⚙️ 系統設定與連線")

    # 只顯示設定狀態，不顯示金鑰內容
    st.caption(
        "NASA FIRMS API：已設定"
        if firms_key
        else "NASA FIRMS API：未設定"
    )
    st.caption(
        "Google Gemini API：已設定"
        if gemini_key
        else "Google Gemini API：未設定"
    )
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
# 3. 驅動層：Open-Meteo 逐時風場預報（未來 7 天，5x5 格點）
# ==========================================
GRID_LON = [94.0, 100.0, 106.0, 112.0, 118.0]   # 格距 6 度
GRID_LAT = [-8.0, -4.0, 0.0, 4.0, 8.0]          # 格距 4 度
N_HOURS = 169                                    # t = 0 ~ 168 小時
MAX_FIRES = 60                                   # 模擬採 FRP 最大的前 60 個火點
RELEASE_STEP = 3                                 # 每 3 小時釋放一批煙團（持續排放）

HOURS = list(range(6, 169, 6))                   # 時間軸：6 小時 ~ 7 天，每 6 小時一格
PLAY_INTERVAL_S = 1.5                            # 動態播放：每幾秒前進一格

def fmt_h(h):
    if h < 24:
        return f"{h} 小時"
    d, r = divmod(h, 24)
    return f"{d} 天" if r == 0 else f"{d} 天 {r} 小時"

def fmt_arrive(h):
    return "7 天內未受波及" if h is None else f"T+{h} 小時（第 {h // 24 + 1} 天）"

CITY_TARGETS = [
    ("新加坡 Singapore", 1.3521, 103.8198),
    ("吉隆坡 Kuala Lumpur", 3.1390, 101.6869),
    ("馬六甲 Malacca", 2.1896, 102.2501),
    ("檳城 Penang", 5.4141, 100.3288),
    ("古晉 Kuching", 1.5535, 110.3593),
    ("斯里巴加灣市 Bandar Seri Begawan", 4.9031, 114.9398),
    ("雅加達 Jakarta", -6.2088, 106.8456),
]

@st.cache_data(ttl=3600, show_spinner="🌬️ 正在下載 Open-Meteo 風場預報...")
def fetch_wind_forecast():
    """回傳 (風速 ws[km/h], 來風方位 wd[度], 是否成功, 預報起點UTC)；陣列形狀 (緯度格, 經度格, 小時)"""
    shape = (len(GRID_LAT), len(GRID_LON), N_HOURS)
    fallback = (np.full(shape, 15.0), np.full(shape, 210.0), False, "N/A")   # 預設西南季風
    lats = [la for la in GRID_LAT for _ in GRID_LON]
    lons = [lo for _ in GRID_LAT for lo in GRID_LON]
    params = {
        "latitude": ",".join(str(v) for v in lats),
        "longitude": ",".join(str(v) for v in lons),
        "hourly": "wind_speed_10m,wind_direction_10m",
        "forecast_days": 8,
        "timezone": "UTC",
        "wind_speed_unit": "kmh",
    }
    try:
        r = requests.get("https://api.open-meteo.com/v1/forecast", params=params, timeout=25)
        r.raise_for_status()
        js = r.json()
        if isinstance(js, dict):
            js = [js]
        if len(js) != len(lats):
            return fallback
        times = js[0]["hourly"]["time"]
        now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:00")
        start = times.index(now) if now in times else 0

        def grab(key):
            arr = np.array([j["hourly"][key][start:start + N_HOURS] for j in js], dtype=float)
            if arr.shape[1] < N_HOURS:
                arr = np.pad(arr, ((0, 0), (0, N_HOURS - arr.shape[1])), mode="edge")
            return arr

        ws, wd = grab("wind_speed_10m"), grab("wind_direction_10m")
        ws = np.where(np.isnan(ws), np.nanmean(ws) if np.isfinite(np.nanmean(ws)) else 15.0, ws)
        wd = np.where(np.isnan(wd), 210.0, wd)
        return ws.reshape(shape), wd.reshape(shape), True, times[start]
    except Exception:
        return fallback

def uv_from(ws, wd):
    """氣象風向(風從哪來) -> 煙團移動向量：u 向東、v 向北 (km/h)"""
    r = np.radians(wd)
    return -ws * np.sin(r), -ws * np.cos(r)

def sample_wind(U, V, lat, lon, t):
    """在格點風場上做雙線性空間內插，取第 t 小時的風"""
    fy = np.clip((lat - GRID_LAT[0]) / (GRID_LAT[1] - GRID_LAT[0]), 0, len(GRID_LAT) - 1 - 1e-9)
    fx = np.clip((lon - GRID_LON[0]) / (GRID_LON[1] - GRID_LON[0]), 0, len(GRID_LON) - 1 - 1e-9)
    iy, ix = fy.astype(int), fx.astype(int)
    dy, dx = fy - iy, fx - ix
    def g(A):
        return (A[iy, ix, t] * (1 - dx) * (1 - dy) + A[iy, ix + 1, t] * dx * (1 - dy)
                + A[iy + 1, ix, t] * (1 - dx) * dy + A[iy + 1, ix + 1, t] * dx * dy)
    return g(U), g(V)

def compass_zh(deg):
    names = ["北", "北北東", "東北", "東北東", "東", "東南東", "東南", "南南東",
             "南", "南南西", "西南", "西南西", "西", "西北西", "西北", "北北西"]
    return names[int((deg % 360) / 22.5 + 0.5) % 16] + "風"

# ==========================================
# 4. 數值模擬層：拉格朗日煙團軌跡 (逐時風場、持續排放、隨煙齡擴張)
# ==========================================
def spread_radius_km(age_h):
    """煙團半徑隨煙齡增長（示意參數，未經實測校正）"""
    return 8.0 + 4.0 * np.power(np.maximum(age_h, 0), 0.7)

@st.cache_data(show_spinner=False)
def simulate_all(fires, U, V):
    """一次模擬到 168 小時，保存每小時所有煙團位置；之後任何預測時間、動畫畫格都直接取用"""
    T = N_HOURS - 1
    f = fires.copy()
    f["frp"] = pd.to_numeric(f["frp"], errors="coerce").fillna(50.0) if "frp" in f else 50.0
    f = f.nlargest(MAX_FIRES, "frp")
    releases = np.arange(0, T, RELEASE_STEP)
    nF, nR = len(f), len(releases)
    lat = np.repeat(f["latitude"].values[:, None].astype(float), nR, axis=1)
    lon = np.repeat(f["longitude"].values[:, None].astype(float), nR, axis=1)
    frp = np.repeat(f["frp"].values[:, None].astype(float), nR, axis=1)
    rel = np.repeat(releases[None, :], nF, axis=0)
    hist_lat = np.empty((T + 1, nF, nR), dtype=np.float32)
    hist_lon = np.empty((T + 1, nF, nR), dtype=np.float32)
    first_hit = {c[0]: None for c in CITY_TARGETS}

    for t in range(T + 1):
        hist_lat[t], hist_lon[t] = lat, lon
        age = t - rel
        if t > 0:                                   # 各城市首次被煙團覆蓋的時間
            m = age > 0
            la, lo, rad = lat[m], lon[m], spread_radius_km(age[m])
            for name, clat, clon in CITY_TARGETS:
                if first_hit[name] is None:
                    d = np.hypot((la - clat) * 111.0, (lo - clon) * 111.0 * np.cos(np.radians(clat)))
                    if np.any(d <= rad):
                        first_hit[name] = t
        if t == T:
            break
        active = age >= 0                           # 以第 t 小時風場推進 1 小時
        la, lo = lat[active], lon[active]
        u, v = sample_wind(U, V, la, lo, t)
        lat[active] = la + v / 111.0
        lon[active] = lo + u / (111.0 * np.cos(np.radians(la)))
    return {"lat": hist_lat, "lon": hist_lon, "rel": rel, "frp": frp, "nF": nF, "first_hit": first_hit}

def frame_df(sim, H):
    """取出預測時間 H 小時當下的煙團，回傳圖層用 DataFrame"""
    age = H - sim["rel"]
    m = age > 0
    la, lo = sim["lat"][H][m].astype(float), sim["lon"][H][m].astype(float)
    ag, fr = age[m].astype(float), sim["frp"][m]
    dens = fr / (1.0 + ag / 12.0)                   # 相對濃度指標（非實際 µg/m3）
    young = np.clip(ag / 96.0, 0, 1)                # 煙齡越長，顏色越淡
    alpha = np.clip(18 + dens * 0.18, 18, 70)
    df = pd.DataFrame({
        "lat": la, "lon": lo,
        "age_h": ag.round().astype(int), "frp": fr.round().astype(int),
        "density": dens.round().astype(int),
        "radius_m": spread_radius_km(ag) * 1000.0,
        "fill": [[255, int(190 + 45 * y), int(60 + 110 * y), int(a)] for y, a in zip(young, alpha)],
    })
    return df

def city_status(df, name, clat, clon, first_h, H):
    d = np.hypot((df["lat"] - clat) * 111.0, (df["lon"] - clon) * 111.0 * np.cos(np.radians(clat)))
    if bool((d <= df["radius_m"] / 1000.0).any()):
        return "🟠 煙羽覆蓋中"
    if first_h is not None and first_h <= H:
        return "🟢 曾受影響，已移出"
    return "⚪ 尚未抵達"

# 執行：火點 + 風場預報（煙羽模擬在使用者設定情境後於地圖區執行）
fire_data = fetch_firms_hotspots(firms_key, bbox)
ws_fc, wd_fc, wind_ok, wind_start = fetch_wind_forecast()


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

    # ---------- 情境設定：預測時間 / 動態播放 / 風場來源 ----------
    horizon_h = st.select_slider("⏱️ 預測時間（自現在起算）", options=HOURS, value=24, format_func=fmt_h)
    horizon_label = fmt_h(horizon_h)
    play_on = False
    if hasattr(st, "fragment"):
        play_on = st.toggle(f"▶️ 動態播放（由 6 小時自動播放到 7 天，每 {PLAY_INTERVAL_S:g} 秒前進 6 小時）", value=False)
    else:
        st.caption("ℹ️ 目前的 Streamlit 版本不支援動態播放，請升級至 1.37 以上。")

    wind_mode = st.radio("🌬️ 風場來源", ["Open-Meteo 逐時預報", "手動設定風場"], horizontal=True)
    if wind_mode == "Open-Meteo 逐時預報":
        with st.expander("敏感度調整：在預報上疊加偏差，評估預報誤差的影響"):
            dir_off = st.slider("風向偏轉（度，正值＝順時針）", -90, 90, 0, 5)
            spd_mult = st.slider("風速倍率（也可用來模擬近地面以上較強的傳輸風）", 0.5, 2.0, 1.0, 0.1)
        ws_use = ws_fc * spd_mult
        wd_use = (wd_fc + dir_off) % 360
        if wind_ok:
            st.caption(f"風場資料：Open-Meteo 10 m 逐時預報，起點 {wind_start} UTC，5×5 格點雙線性內插，每小時快取更新。")
        else:
            st.warning("⚠️ 無法取得 Open-Meteo 預報，暫以固定西南風 15 km/h 作為替代。")
        scenario_txt = f"Open-Meteo 預報（風向偏轉 {dir_off}°，風速倍率 {spd_mult}）"
    else:
        man_dir = st.slider("來風方位角（風從哪個方向吹來；0=北、90=東、180=南、270=西）", 0, 359, 225, 5)
        man_spd = st.slider("風速 (km/h)", 0, 60, 15, 1)
        ws_use = np.full(ws_fc.shape, float(man_spd))
        wd_use = np.full(wd_fc.shape, float(man_dir))
        st.caption(f"手動情境：{compass_zh(man_dir)}（{man_dir}°），{man_spd} km/h，全時段、全區域固定。")
        scenario_txt = f"手動固定風場（來風 {man_dir}°，{man_spd} km/h）"

    U, V = uv_from(ws_use, wd_use)
    sim = simulate_all(fire_data, U, V)      # 只在風場情境改變時重算，動畫畫格直接取用
    n_fires_used = sim["nF"]
    first_hit = sim["first_hit"]
    arrival_summary = "；".join(f"{name}：{fmt_arrive(first_hit[name])}" for name, _, _ in CITY_TARGETS)

    # 區域中心的風時序（供微氣象文字、AI 與風場時序圖使用）
    c_lat, c_lon = np.array([center_lat]), np.array([center_lon])
    cu = np.array([sample_wind(U, V, c_lat, c_lon, t)[0][0] for t in range(N_HOURS)])
    cv = np.array([sample_wind(U, V, c_lat, c_lon, t)[1][0] for t in range(N_HOURS)])
    wind_speed = round(float(np.hypot(cu[0], cv[0])), 1)
    wind_direction = round(float(np.degrees(np.arctan2(-cu[0], -cv[0])) % 360), 0)
    mean_speed = round(float(np.hypot(cu[:horizon_h + 1].mean(), cv[:horizon_h + 1].mean())), 1)
    mean_direction = round(float(np.degrees(np.arctan2(-cu[:horizon_h + 1].mean(), -cv[:horizon_h + 1].mean())) % 360), 0)

    fire_plot = fire_data.copy()
    fire_plot["frp"] = pd.to_numeric(fire_plot["frp"], errors="coerce").fillna(50.0)
    fire_plot["age_h"] = 0
    fire_plot["density"] = fire_plot["frp"].round().astype(int)

    base_layers = []
    for lid, m in (("sat-wide", mosaic_wide), ("sat-near", mosaic_near)):
        if m[2] > 0:
            base_layers.append(make_bitmap_layer(lid, m))
    if not base_layers:
        st.warning("⚠️ 無法連線 Esri 圖磚伺服器，暫時顯示預設底圖。")
    elif mosaic_near[2] < mosaic_near[3]:
        st.caption(f"ℹ️ 近景影像有 {mosaic_near[3] - mosaic_near[2]} 塊圖磚載入失敗（以深灰色補位），可重新整理重試。")

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

    def draw_map():
        """地圖 + 城市表。播放時由 st.fragment 定時重跑，只更新這一塊，不會重跑整頁。"""
        if play_on:
            cur = st.session_state.get("anim_h", 0) + 6
            if cur > 168:
                cur = 6
            st.session_state["anim_h"] = cur
            H = cur
        else:
            H = horizon_h
        smoke_df = frame_df(sim, H)

        smoke_layer = pdk.Layer(
            "ScatterplotLayer",
            id="smoke-plumes",
            data=smoke_df,
            get_position=["lon", "lat"],
            get_radius="radius_m",
            get_fill_color="fill",
            pickable=True,
            parameters={"depthTest": False},
        )
        fire_layer = pdk.Layer(
            "ColumnLayer",
            id="fire-columns",
            data=fire_plot,
            get_position=["longitude", "latitude"],
            get_elevation="frp * 250",
            elevation_scale=1,
            radius=4000,
            get_fill_color=[255, 30, 0, 230],
            pickable=True,
        )
        deck = pdk.Deck(
            layers=base_layers + [smoke_layer, fire_layer, text_layer],
            map_style=None,
            initial_view_state=pdk.ViewState(
                latitude=center_lat, longitude=center_lon, zoom=5.8, pitch=45, bearing=10
            ),
            tooltip={
                "html": "<b>火勢 FRP:</b> {frp} MW<br/><b>煙齡:</b> {age_h} 小時<br/><b>相對濃度指標:</b> {density}",
                "style": {"backgroundColor": "rgba(20, 25, 35, 0.9)", "color": "white"}
            }
        )

        when = ""
        if wind_ok:
            t0 = datetime.strptime(wind_start, "%Y-%m-%dT%H:%M")
            when = f"｜約 {(t0 + pd.Timedelta(hours=H)).strftime('%m/%d %H:%M')} UTC"
        st.markdown(f"#### ⏱️ 目前顯示：預測 **T+{H} 小時**（{fmt_h(H)}）{when}")
        st.progress(H / 168)
        st.pydeck_chart(deck)
        st.caption("底圖影像：Esri, Maxar, Earthstar Geographics, and the GIS User Community"
                   + ("　｜　播放中縮放／平移會在下一格重置，想自由操作請先關閉播放。" if play_on else ""))

        st.markdown(f"##### 🏙️ 下風處城市影響評估（{scenario_txt}）")
        rows = [{"城市": name,
                 "目前狀態": city_status(smoke_df, name, clat, clon, first_hit[name], H),
                 "最早受煙羽覆蓋": fmt_arrive(first_hit[name])}
                for name, clat, clon in CITY_TARGETS]
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    if play_on:
        st.fragment(run_every=PLAY_INTERVAL_S)(draw_map)()
    else:
        draw_map()

    note = f"模擬採 FRP 最大的前 {n_fires_used} 個火點，" if len(fire_data) > n_fires_used else ""
    st.caption(f"{note}假設火點持續排放、每 {RELEASE_STEP} 小時釋放一批煙團；圈越大代表煙齡越長、擴散越廣，顏色越淡代表越老。"
               "此為簡化軌跡模型：未含邊界層高度、垂直混合、乾濕沉降與降雨清除，風場為 10 m 風，"
               "數值僅供情境比較。正式決策請搭配 CAMS、HYSPLIT 或 ASMC 官方預報。")

    with st.expander("📈 查看風場時序（區域中心，未來 7 天）"):
        st.line_chart(pd.DataFrame({"風速 (km/h)": [float(np.hypot(cu[h], cv[h])) for h in range(N_HOURS)]},
                                   index=list(range(N_HOURS))))
        pts = [0, 6, 12, 18, 24, 48, 72, 96, 120, 144, 168]
        st.dataframe(pd.DataFrame({
            "時間": [f"T+{h} 小時" for h in pts],
            "風速 (km/h)": [round(float(np.hypot(cu[h], cv[h])), 1) for h in pts],
            "來風方位角 (度)": [int(np.degrees(np.arctan2(-cu[h], -cv[h])) % 360) for h in pts],
            "風向": [compass_zh(np.degrees(np.arctan2(-cu[h], -cv[h])) % 360) for h in pts],
        }), hide_index=True, use_container_width=True)

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
                <span><b>灰黃色煙羽圈</b>：所選預測時間點<b>濃煙霧霾隨風飄散的範圍</b>（圈越大＝煙齡越長、擴散越廣）</span>
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
        f"🧭 **微氣象遙測資訊**：目前近地風速 **{wind_speed} km/h**，來風方位角 **{int(wind_direction)}°**（{compass_zh(wind_direction)}）。"
        f"未來 **{horizon_label}** 平均風為 **{mean_speed} km/h**、來風 **{int(mean_direction)}°**，"
        f"煙霾整體向 **{int((mean_direction + 180) % 360)}°** 方向推進。"
    )

def fmt_range(vals):
    """多個情境的最早抵達時間 -> 範圍文字"""
    got = [v for v in vals if v is not None]
    if not got:
        return "各情境 7 天內皆未受波及"
    lo, hi = min(got), max(got)
    txt = f"T+{lo} 小時" if lo == hi else f"T+{lo} ~ T+{hi} 小時"
    if len(got) < len(vals):
        txt += f"（{len(vals) - len(got)}/{len(vals)} 個情境未受波及）"
    return txt

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
                focus_h = st.session_state.get("anim_h", horizon_h) if play_on else horizon_h

                # --- 未來風場時序（區域中心）---
                tl_pts = [6, 12, 24, 48, 72, 168]
                def _wd(h):
                    return float(np.degrees(np.arctan2(-cu[h], -cv[h])) % 360)
                wind_lines = "；".join(
                    f"T+{h}h {round(float(np.hypot(cu[h], cv[h])), 1)} km/h、來風 {int(_wd(h))}°（{compass_zh(_wd(h))}）"
                    for h in tl_pts)

                # --- 各時間點城市狀態 ---
                status_lines = []
                for h in tl_pts:
                    df_h = frame_df(sim, h)
                    items = "、".join(
                        f"{name.split()[0]}{city_status(df_h, name, clat, clon, first_hit[name], h).split(' ', 1)[1]}"
                        for name, clat, clon in CITY_TARGETS)
                    status_lines.append(f"T+{h}h：{items}")
                status_text = "\n                  ".join(status_lines)

                # --- 敏感度：風向 ±15°、風速 x0.7 / x1.3，各城市最早抵達時間的範圍 ---
                variants = [(ws_use, (wd_use - 15) % 360), (ws_use, (wd_use + 15) % 360),
                            (ws_use * 0.7, wd_use), (ws_use * 1.3, wd_use)]
                hits = {name: [first_hit[name]] for name, _, _ in CITY_TARGETS}
                for w_s, w_d in variants:
                    fh = simulate_all(fire_data, *uv_from(w_s, w_d))["first_hit"]
                    for name in hits:
                        hits[name].append(fh[name])
                range_map = {name: fmt_range(v) for name, v in hits.items()}
                range_text = "；".join(f"{name.split()[0]} {r}" for name, r in range_map.items())

                prompt = f"""
                你現在是「東南亞國協跨國霾害應變中心（ASMC）」的資深決策專家顧問。
                系統已完成野火感知與未來 7 天的煙羽軌跡模擬（簡化拉格朗日模型，使用 10 m 風，未含邊界層與沉降），數據如下：
                - 觀測場域：{region_choice}
                - 衛星即時熱點總數：{len(fire_data)} 處；單點最高輻射能量 (FRP)：{max_frp} MW (泥炭地陰燃重大指標)
                - 目前風場：風速 {wind_speed} km/h，煙塵漂移方位角 {drift_dir}°
                - 模擬情境：{scenario_txt}；使用者目前檢視的預測時間為 T+{focus_h} 小時
                - 區域中心未來風場時序：{wind_lines}
                - 各城市最早受煙羽覆蓋時間（基準情境）：{arrival_summary}
                - 敏感度範圍（風向 ±15°、風速 ×0.7 與 ×1.3 共 4 個擾動情境加基準）：{range_text}
                - 各時間點城市狀態：
                  {status_text}

                請只使用上述數據，不要編造未提供的數字；數據不足之處請明確說明不確定性。
                請以繁體中文撰寫一份高階應變行動指引，嚴格包含以下結構：
                1. 【跨國擴散風險診斷】：依風場時序與各時間點城市狀態，說明未來 6 小時、24 小時、3 天、7 天各階段的風險變化（含風向轉變造成的進入、退出），並以敏感度範圍說明抵達時間的不確定性，指出最需優先預警的城市與麻六甲海峽航道。
                2. 【泥炭地撲滅戰術方案】：針對泥炭地地下悶燒無法以傳統表面水撲滅的特性，提出水資源調度與阻隔溝渠開挖建議。
                3. 【跨國協調與預警宣告】：依據《東南亞跨國霾害污染協議（AATHP）》，印尼應在何時、向哪些鄰國、以何種層級通報；請標示哪些決定需等風場預報更新後再確認。
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
                    
                    city_bullets = "\n".join(
                        f"  * **{name}**：基準 {fmt_arrive(first_hit[name])}；敏感度範圍 {range_map[name]}"
                        for name, _, _ in CITY_TARGETS)
                    notify = "、".join(n.split()[0] for n, _, _ in CITY_TARGETS if first_hit[n] is not None) or "（模擬期內無城市受波及）"
                    st.markdown(f"""
### 🚨 東南亞跨國霾害應變中心（ASMC）決策指引（示範輸出：由模擬結果自動彙整，非 AI 推論）

#### 1. 【跨國擴散風險診斷】
* **風場時序**：{wind_lines}
* **情境**：{scenario_txt}
* **各城市最早受煙羽覆蓋時間**：
{city_bullets}

#### 2. 【泥炭地撲滅戰術方案】
* 泥炭層可深達數公尺，地下陰燃無法以空中灑水撲滅，直升機投水僅能處理地表火。
* 於火線外圍開挖阻隔溝渠至地下水位以下，並關閉周邊運河閘門進行水文回補（Rewetting），提高地下水位以隔絕氧氣。

#### 3. 【跨國協調與預警宣告】
* 模擬顯示可能受波及的城市：{notify}。建議依 AATHP 與 ASMC 警戒機制向相關國家通報並共享熱點遙測數據；實際警戒等級與通報時限以官方機制為準。
* 風向轉變會造成煙羽進入後再退出，請於風場預報更新後（每小時）重新評估。
                    """)
