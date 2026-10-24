"""
坡地複合災害決策孿生 — 土石流與大規模崩塌警戒推演
資料來源：農村發展及水土保持署（土石流及大規模崩塌防災資訊網開放資料 API）、
          中央氣象署開放資料平臺（O-A0002-001 雨量觀測）
執行：streamlit run slope_twin.py
"""
import base64
import html
import math
import os
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO

import numpy as np
import pandas as pd
import pydeck as pdk
import requests
import streamlit as st
from PIL import Image

try:
    import google.generativeai as genai
except Exception:  # 沒裝 Gemini 套件時，AI 區塊改用備援文字
    genai = None

st.set_page_config(page_title="坡地複合災害決策孿生", page_icon="⛰️", layout="wide")

# ==========================================
# 0. API 端點（取自水保署資料平台「API 說明」頁與氣象署資料集說明）
# ==========================================
URL_ALERT = "https://ls.ardswc.gov.tw/api/LandslideAlertOpenData"
URL_ALERT_HIST = "https://ls.ardswc.gov.tw/api/LandslideAlertHistoryOpenData/{id}"
URL_EVENTS = "https://246.ardswc.gov.tw/webService/GetIDisasterInfo.ashx"
URL_DEBRIS_RAIN = "https://246.ardswc.gov.tw/webService/GetDebrisRainData.ashx"
URL_LS_VALUES = "https://246.ardswc.gov.tw/WebService/GetLSCountyTownAlertValueList.ashx"
URL_CWA_RAIN = "https://opendata.cwa.gov.tw/api/v1/rest/datastore/O-A0002-001"
UA = {"User-Agent": "Mozilla/5.0 (slope-digital-twin-demo)"}
PLAY_INTERVAL_S = 1.5


def secret(name):
    """先讀 Streamlit Secrets，沒有再讀環境變數（方便搬到 Docker / Cloud Run）"""
    try:
        v = st.secrets.get(name, "")
        if v:
            return v
    except Exception:
        pass
    return os.environ.get(name, "")


def get_json(url, params=None, timeout=25):
    r = requests.get(url, params=params, headers=UA, timeout=timeout)
    r.raise_for_status()
    return r.json()


def esc(x):
    return html.escape("" if x is None or (isinstance(x, float) and np.isnan(x)) else str(x))


# ==========================================
# 1. 資料層
# ==========================================
ALERT_COLS = ["AlertType", "DebrisNo", "LandslideID", "LandslideName", "County", "Town", "Vill",
              "AlertLevel", "LastUpdateDate", "ReportID"]


def _alert_df(js):
    if isinstance(js, dict):                       # 例如 {"errorMessage":"目前無警戒"}
        js = js.get("data") or js.get("records") or []
    df = pd.DataFrame(js)
    for c in ALERT_COLS:
        if c not in df:
            df[c] = None
    return df[ALERT_COLS]


@st.cache_data(ttl=120, show_spinner=False)
def fetch_alerts():
    try:
        return _alert_df(get_json(URL_ALERT)), ""
    except Exception as e:
        return None, str(e)


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_history(event_id):
    try:
        df = _alert_df(get_json(URL_ALERT_HIST.format(id=event_id)))
        df["t"] = pd.to_datetime(df["LastUpdateDate"], errors="coerce")
        return df.dropna(subset=["t"]), ""
    except Exception as e:
        return None, str(e)


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_events():
    try:
        df = pd.DataFrame(get_json(URL_EVENTS))
        if "OpenTime" in df:
            df = df.sort_values("OpenTime", ascending=False)
        return df, ""
    except Exception as e:
        return None, str(e)


@st.cache_data(ttl=300, show_spinner=False)
def fetch_debris_rain():
    """每條土石流潛勢溪流：警戒基準值 AlertValue、兩個參考雨量站與其 STRT（有效累積雨量，依範例數值推定）"""
    try:
        js = get_json(URL_DEBRIS_RAIN)
        if isinstance(js, dict):
            js = js.get("data") or []
        df = pd.DataFrame(js).rename(columns={"DebrisNO": "DebrisNo"})
        for c in ["AlertValue", "STRT1", "STRT2"]:
            df[c] = pd.to_numeric(df.get(c), errors="coerce")
        df["RT"] = np.fmax(df["STRT1"], df["STRT2"])      # 保守取兩站較大值
        return df, ""
    except Exception as e:
        return None, str(e)


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_ls_values():
    """大規模崩塌潛勢區：警戒值與座標"""
    try:
        df = pd.DataFrame(get_json(URL_LS_VALUES))
        for c in ["AlertValue", "Lng", "Lat"]:
            df[c] = pd.to_numeric(df.get(c), errors="coerce")
        return df, ""
    except Exception as e:
        return None, str(e)


def _rain(x):
    """氣象署特殊值：T=雨跡、-98=連續6小時無雨、-99/X=缺值"""
    if isinstance(x, str):
        return 0.0 if x.strip().upper() == "T" else np.nan
    try:
        v = float(x)
    except Exception:
        return np.nan
    if v == -98:
        return 0.0
    return np.nan if v <= -90 else v


@st.cache_data(ttl=600, show_spinner=False)
def fetch_cwa_rain(key):
    """氣象署雨量站：座標與累積雨量（用來定位參考雨量站）"""
    try:
        js = get_json(URL_CWA_RAIN, {"Authorization": key, "format": "JSON"})
        stations = js["records"]["Station"]
        rows = []
        for s in stations:
            geo = s.get("GeoInfo", {})
            coords = geo.get("Coordinates", [])
            if isinstance(coords, dict):
                coords = [coords]
            c = next((x for x in coords if x.get("CoordinateName") == "WGS84"), coords[0] if coords else {})
            rf = s.get("RainfallElement", {})
            g = lambda k: _rain(rf.get(k, {}).get("Precipitation"))
            rows.append({
                "StationId": s.get("StationId"), "StationName": s.get("StationName"),
                "lat": pd.to_numeric(c.get("StationLatitude"), errors="coerce"),
                "lon": pd.to_numeric(c.get("StationLongitude"), errors="coerce"),
                "past1h": g("Past1hr"), "past24h": g("Past24hr"), "past3d": g("Past3days"),
                "obs": (s.get("ObsTime") or {}).get("DateTime"),
            })
        return pd.DataFrame(rows), ""
    except Exception as e:
        return None, str(e)


def make_demo():
    """示範資料（非真實）：僅用於 API 無法連線時維持展示不中斷"""
    rng = np.random.default_rng(7)
    centers = [("嘉義縣", "阿里山鄉", 23.51, 120.80), ("高雄市", "那瑪夏區", 23.29, 120.69),
               ("屏東縣", "來義鄉", 22.50, 120.64), ("臺東縣", "延平鄉", 22.90, 121.05),
               ("花蓮縣", "秀林鄉", 24.15, 121.60), ("南投縣", "信義鄉", 23.70, 120.87)]
    deb, sta = [], []
    for ci, (cty, twn, la, lo) in enumerate(centers):
        for k in range(3):
            sid = f"DEMO{ci}{k}"
            sta.append({"StationId": sid, "StationName": f"示範站{ci}{k}", "lat": la + rng.normal(0, .08),
                        "lon": lo + rng.normal(0, .08), "past1h": 5.0, "past24h": float(rng.integers(50, 400)),
                        "past3d": 0.0, "obs": ""})
            for j in range(6):
                av = float(rng.choice([300, 350, 400, 450, 500]))
                rt = float(rng.uniform(0.2, 1.15) * av)
                deb.append({"County": cty, "Town": twn, "Vill": f"示範村{j}", "DebrisNo": f"DEMO-DF{ci}{k}{j}",
                            "AlertValue": av, "STID1": sid, "STName1": f"示範站{ci}{k}", "STRT1": rt,
                            "STID2": sid, "STName2": f"示範站{ci}{k}", "STRT2": rt * 0.9, "RT": rt})
    deb = pd.DataFrame(deb)
    alerts = pd.DataFrame([
        {"AlertType": "D", "DebrisNo": r.DebrisNo, "LandslideID": None, "LandslideName": None, "County": r.County,
         "Town": r.Town, "Vill": r.Vill, "AlertLevel": "r" if r.RT >= r.AlertValue * 1.05 else "y",
         "LastUpdateDate": "示範", "ReportID": "DEMO"}
        for r in deb.itertuples() if r.RT >= r.AlertValue])
    ls = pd.DataFrame([{"County": "高雄市", "Town": "六龜區", "LSNo": "DEMO-LL1", "Name": "示範崩塌區A",
                        "AlertValue": 800.0, "Lng": 120.64, "Lat": 23.03},
                       {"County": "屏東縣", "Town": "霧台鄉", "LSNo": "DEMO-LL2", "Name": "示範崩塌區B",
                        "AlertValue": 700.0, "Lng": 120.80, "Lat": 22.74}])
    return alerts, deb, ls, pd.DataFrame(sta)


# ==========================================
# 2. 衛星底圖（伺服器端拼接 Esri World Imagery 圖磚 -> BitmapLayer）
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
    for _ in range(2):
        try:
            r = requests.get(ESRI_TILE.format(z=z, x=x, y=y), timeout=8, headers=UA)
            if r.status_code == 200 and r.content:
                return Image.open(BytesIO(r.content)).convert("RGB")
        except Exception:
            pass
    return None


@st.cache_data(ttl=7 * 86400, show_spinner="🛰️ 正在載入衛星空照底圖（首次約需數秒）...")
def satellite_mosaic(west, south, east, north, z):
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
    canvas.save(buf, "JPEG", quality=80)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode(), bounds, ok, len(jobs)


def bitmap_layer(mosaic):
    uri, bounds, ok, total = mosaic
    return pdk.Layer("BitmapLayer", id="satellite", data=None,
                     image="'" + uri + "'",            # 外層單引號＝字串常數，避免被當成運算式
                     bounds=bounds, opacity=1.0, parameters={"depthTest": False})


# ==========================================
# 3. 分析層：推估達標、測站彙總
# ==========================================
def band_color(r):
    if not np.isfinite(r):
        return [120, 120, 120, 120]
    if r >= 1.0:
        return [230, 30, 30, 235]
    if r >= 0.8:
        return [255, 140, 0, 220]
    if r >= 0.5:
        return [240, 210, 60, 200]
    return [70, 180, 120, 170]


def attach_station(deb, cwa):
    """以參考雨量站 1（缺座標則站 2）的座標代表該溪流的位置（近似，非溪流實際座標）"""
    deb = deb.copy()
    if cwa is None or cwa.empty:
        for c in ["lat", "lon", "past24h", "st_id", "st_name"]:
            deb[c] = np.nan if c in ("lat", "lon", "past24h") else None
        return deb
    m = cwa.dropna(subset=["lat", "lon"]).drop_duplicates("StationId").set_index("StationId")
    use2 = ~deb["STID1"].isin(m.index) & deb["STID2"].isin(m.index)
    deb["st_id"] = deb["STID1"].where(~use2, deb["STID2"])
    deb["st_name"] = deb["STName1"].where(~use2, deb["STName2"])
    for c in ["lat", "lon", "past24h"]:
        deb[c] = deb["st_id"].map(m[c])
    return deb


def official_level_map(alerts):
    """DebrisNo -> 'r' 或 'y'（紅色優先）"""
    out = {}
    if alerts is None or alerts.empty:
        return out
    a = alerts[alerts["AlertType"] == "D"]
    for r in a.itertuples():
        if r.DebrisNo and (out.get(r.DebrisNo) != "r"):
            out[r.DebrisNo] = r.AlertLevel
    return out


def aggregate_stations(deb):
    d = deb.dropna(subset=["lat", "lon", "ratio"])
    if d.empty:
        return pd.DataFrame()
    rows = []
    for sid, g in d.groupby("st_id"):
        mr = float(g["ratio"].max())
        n_r = int((g["off"] == "r").sum())
        n_y = int((g["off"] == "y").sum())
        f = g.iloc[0]
        rows.append({
            "lat": float(f["lat"]), "lon": float(f["lon"]),
            "elev": min(mr, 1.5) * 6000.0, "fill": band_color(mr),
            "halo": [230, 30, 30, 80] if n_r else ([255, 215, 0, 80] if n_y else [0, 0, 0, 0]),
            "title": esc(f"{f['st_name']}（{f['County']}{f['Town']}）"),
            "l1": esc(f"溪流 {len(g)} 條｜推估達標 {int((g['ratio'] >= 1).sum())} 條｜接近(80–100%) {int(((g['ratio'] >= .8) & (g['ratio'] < 1)).sum())} 條"),
            "l2": esc(f"最高 {mr * 100:.0f}% ｜ 官方警戒 紅{n_r}／黃{n_y}"),
        })
    return pd.DataFrame(rows)


def ls_layer_df(ls, alerts):
    if ls is None or ls.empty:
        return pd.DataFrame()
    lv = {}
    if alerts is not None and not alerts.empty:
        for r in alerts[alerts["AlertType"] == "L"].itertuples():
            nm = r.LandslideName or r.LandslideID
            if nm and lv.get(nm) != "r":
                lv[nm] = r.AlertLevel
    d = ls.dropna(subset=["Lat", "Lng"]).copy()
    d["lv"] = d["Name"].map(lv).fillna("")
    d["lat"], d["lon"] = d["Lat"], d["Lng"]
    d["fill"] = d["lv"].map({"r": [200, 0, 160, 240], "y": [255, 215, 0, 230]}).apply(
        lambda v: v if isinstance(v, list) else [160, 160, 255, 150])
    d["title"] = d.apply(lambda r: esc(f"大規模崩塌：{r['Name']}"), axis=1)
    d["l1"] = d.apply(lambda r: esc(f"{r['County']}{r['Town']}｜警戒值 {r['AlertValue']:.0f} mm"), axis=1)
    d["l2"] = d["lv"].map({"r": "官方：紅色警戒", "y": "官方：黃色警戒"}).fillna("官方：目前無警戒")
    return d[["lat", "lon", "fill", "title", "l1", "l2"]]


def deck_for(layers):
    return pdk.Deck(
        layers=layers, map_style=None,
        initial_view_state=pdk.ViewState(latitude=23.7, longitude=120.95, zoom=7.3, pitch=50, bearing=0),
        tooltip={"html": "<b>{title}</b><br/>{l1}<br/>{l2}",
                 "style": {"backgroundColor": "rgba(20,25,35,0.92)", "color": "white"}})


# ==========================================
# 4. 側邊欄與資料載入
# ==========================================
st.sidebar.header("🔑 設定")
cwa_key = st.sidebar.text_input("氣象署授權碼 (CWA_API_KEY)", value=secret("CWA_API_KEY"), type="password",
                                help="用來取得雨量站座標；至 opendata.cwa.gov.tw 免費註冊取得")
gemini_key = st.sidebar.text_input("Gemini API Key", value=secret("GEMINI_API_KEY"), type="password")
show_sat = st.sidebar.checkbox("顯示衛星空照底圖", value=True)
force_demo = st.sidebar.checkbox("強制使用示範資料（非真實）", value=False)

alerts, err_a = fetch_alerts()
deb, err_d = fetch_debris_rain()
ls, err_l = fetch_ls_values()
cwa, err_c = fetch_cwa_rain(cwa_key) if cwa_key else (None, "未提供氣象署授權碼")

is_demo = force_demo or deb is None
if is_demo:
    alerts, deb, ls, cwa = make_demo()
if alerts is None:
    alerts = _alert_df([])
if ls is None:
    ls = pd.DataFrame(columns=["County", "Town", "LSNo", "Name", "AlertValue", "Lng", "Lat"])

st.title("⛰️ 坡地複合災害決策孿生")
st.caption("即時雨量 × 警戒基準值 × 官方警戒 × 情境推演 × AI 決策建議　｜　資料：農村水保署、中央氣象署（政府資料開放授權）")
if is_demo:
    st.warning("⚠️ 目前顯示的是**示範資料（非真實）**。" + ("" if force_demo else f"原因：水保署 API 連線失敗（{err_d}）。"))

with st.expander("📡 資料來源狀態"):
    def _s(name, err, n):
        return f"- {'✅' if not err else '❌'} {name}：{n if not err else err}"
    if not is_demo:
        st.markdown("\n".join([
            _s("官方警戒（ls.ardswc.gov.tw）", err_a, f"{0 if alerts is None else len(alerts)} 筆"),
            _s("溪流警戒基準值與參考雨量站雨量", err_d, f"{len(deb)} 條溪流"),
            _s("大規模崩塌潛勢區警戒值", err_l, f"{len(ls)} 處"),
            _s("氣象署雨量站座標", err_c, f"{0 if cwa is None else len(cwa)} 站"),
        ]))
        if err_c:
            st.info("沒有氣象署授權碼或連線失敗時，無法標出溪流位置（仍可看表格與大規模崩塌點位）。")
    else:
        st.write("示範模式。")

deb = attach_station(deb, cwa)
deb["off"] = deb["DebrisNo"].map(official_level_map(alerts)).fillna("")
has_xy = deb["lat"].notna().any()

tab1, tab2, tab3 = st.tabs(["① 即時態勢與情境推演", "② 歷史事件回放", "③ AI 決策建議"])

# ==========================================
# 5. 分頁①：即時態勢與情境推演
# ==========================================
with tab1:
    c1, c2 = st.columns(2)
    with c1:
        delta = st.slider("警戒基準值調降（mm）　↳ 模擬地震後坡地鬆動；2025 年嘉義大埔地震後，水保署曾在部分地區調降 50～100 mm",
                          0, 150, 0, 10)
    with c2:
        add_rain = st.slider("未來 24 小時預期再降雨（mm）　↳ 保守推估、未計雨量衰減，可填入氣象署定量降水預報數值",
                             0, 600, 0, 25)
    counties = sorted(deb["County"].dropna().unique())
    sel = st.multiselect("縣市篩選（空白＝全部）", counties)

    work = deb if not sel else deb[deb["County"].isin(sel)]
    work = work.copy()
    work["ratio_now"] = (work["RT"] / work["AlertValue"]).replace([np.inf, -np.inf], np.nan)
    work["adj_value"] = (work["AlertValue"] - delta).clip(lower=1)
    work["ratio"] = ((work["RT"] + add_rain) / work["adj_value"]).replace([np.inf, -np.inf], np.nan)

    a = alerts if not sel else alerts[alerts["County"].isin(sel)]
    nd_r = int(((a["AlertType"] == "D") & (a["AlertLevel"] == "r")).sum())
    nd_y = int(((a["AlertType"] == "D") & (a["AlertLevel"] == "y")).sum())
    nl_r = int(((a["AlertType"] == "L") & (a["AlertLevel"] == "r")).sum())
    nl_y = int(((a["AlertType"] == "L") & (a["AlertLevel"] == "y")).sum())
    base_reach = int((work["ratio_now"] >= 1).sum())
    scen_reach = int((work["ratio"] >= 1).sum())

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("官方土石流警戒 紅／黃", f"{nd_r} ／ {nd_y}")
    k2.metric("官方大規模崩塌警戒 紅／黃", f"{nl_r} ／ {nl_y}")
    k3.metric("目前雨量已達基準值（推估）", f"{base_reach} 條")
    k4.metric("情境推演後達基準值（推估）", f"{scen_reach} 條", delta=f"{scen_reach - base_reach:+d}")

    layers = []
    if show_sat:
        m = satellite_mosaic(119.8, 21.7, 122.3, 25.5, 9)
        if m[2] > 0:
            layers.append(bitmap_layer(m))
        else:
            st.caption("⚠️ 無法連線 Esri 圖磚伺服器，暫以預設底圖顯示。")
    stn = aggregate_stations(work) if has_xy else pd.DataFrame()
    if not stn.empty:
        halo = stn[stn["halo"].apply(lambda v: v[3] > 0)]
        if not halo.empty:
            layers.append(pdk.Layer("ScatterplotLayer", id="halo", data=halo, get_position=["lon", "lat"],
                                    get_radius=5000, radius_min_pixels=8, get_fill_color="halo",
                                    parameters={"depthTest": False}))
        layers.append(pdk.Layer("ColumnLayer", id="stations", data=stn, get_position=["lon", "lat"],
                                get_elevation="elev", elevation_scale=1, radius=1800,
                                get_fill_color="fill", pickable=True, auto_highlight=True))
    lsd = ls_layer_df(ls, alerts)
    if not lsd.empty:
        layers.append(pdk.Layer("ScatterplotLayer", id="landslides", data=lsd, get_position=["lon", "lat"],
                                get_radius=3000, radius_min_pixels=7, get_fill_color="fill", pickable=True,
                                parameters={"depthTest": False}))
    st.pydeck_chart(deck_for(layers))
    st.caption("柱高＝該雨量站所屬溪流中最高的「有效累積雨量 ÷ 警戒基準值」（100% 約 6 公里高）。顏色：綠 <50%、黃 50–80%、橙 80–100%、紅 ≥100%。"
               "外圈：官方已發布黃／紅色警戒。紫點／黃點／淡藍點：大規模崩塌潛勢區（紅警戒／黃警戒／無警戒）。"
               "溪流位置以其參考雨量站座標近似，非溪流實際座標。")
    if not has_xy:
        st.info("沒有雨量站座標，地圖僅顯示大規模崩塌潛勢區。請在側邊欄填入氣象署授權碼。")

    st.markdown("##### 🔎 最接近或已達警戒基準值的溪流（前 20）")
    top = work.sort_values("ratio", ascending=False).head(20)
    st.dataframe(pd.DataFrame({
        "溪流編號": top["DebrisNo"], "縣市": top["County"], "鄉鎮": top["Town"], "村里": top["Vill"],
        "參考雨量站": top["STName1"], "有效累積雨量 RT (mm)": top["RT"].round(1),
        "警戒基準值(調降後) (mm)": top["adj_value"].round(0),
        "推估百分比": (top["ratio"] * 100).round(0).astype("Int64").astype(str) + "%",
        "官方警戒": top["off"].map({"r": "🔴 紅色", "y": "🟡 黃色"}).fillna("—"),
    }), hide_index=True, use_container_width=True)

    st.markdown("##### 🏘️ 各縣市推估達標的溪流數")
    by_cty = work[work["ratio"] >= 1].groupby("County").size().sort_values(ascending=False)
    if by_cty.empty:
        st.caption("目前情境下沒有溪流達到基準值。")
    else:
        st.bar_chart(by_cty)
    st.caption("說明：RT 取參考雨量站 1、2 較大者（保守）；STRT 欄位意義依範例數值推定為有效累積雨量，請對照防災資訊網確認。"
               "「推估」僅為情境比較，不等同官方警戒——官方警戒另有其發布判準與人工判斷。")

# ==========================================
# 6. 分頁②：歷史事件回放
# ==========================================
with tab2:
    if is_demo:
        st.info("示範模式不提供歷史回放，請連線至水保署 API 後使用。")
    else:
        ev, err_e = fetch_events()
        opts = {}
        if ev is not None and not ev.empty:
            for r in ev.head(60).itertuples():
                opts[f"{r.DisasterID}｜{getattr(r, 'DisasterName', '')}（{str(getattr(r, 'OpenTime', ''))[:10]}）"] = r.DisasterID
        c1, c2 = st.columns([2, 1])
        with c1:
            pick = st.selectbox("選擇災害事件（取自應變小組開設資料）", list(opts.keys()) or ["（無法取得事件清單）"])
        with c2:
            manual = st.text_input("或直接輸入事件編號", value="", placeholder="例如 110D")
        event_id = manual.strip() or opts.get(pick, "")
        if err_e:
            st.caption(f"事件清單取得失敗：{err_e}")
        hist, err_h = fetch_history(event_id) if event_id else (None, "")
        if not event_id:
            st.info("請選擇事件或輸入編號。")
        elif hist is None or hist.empty:
            st.warning(f"事件 {event_id} 沒有警戒發布紀錄" + (f"（{err_h}）" if err_h else "") + "。")
        else:
            times = sorted(hist["t"].unique())
            lab = [pd.Timestamp(t).strftime("%m/%d %H:%M") for t in times]
            hist = hist.assign(rank=hist["AlertLevel"].map({"r": 2, "y": 1}).fillna(0),
                               key=np.where(hist["AlertType"] == "D", hist["DebrisNo"],
                                            hist["LandslideName"].fillna(hist["LandslideID"])))

            def counts_upto(t):
                h = hist[hist["t"] <= t].sort_values("rank").drop_duplicates(["AlertType", "key"], keep="last")
                return h

            series = []
            for t, l in zip(times, lab):
                h = counts_upto(t)
                series.append({"時間": l,
                               "土石流 紅": int(((h["AlertType"] == "D") & (h["rank"] == 2)).sum()),
                               "土石流 黃": int(((h["AlertType"] == "D") & (h["rank"] == 1)).sum()),
                               "大規模崩塌 紅": int(((h["AlertType"] == "L") & (h["rank"] == 2)).sum()),
                               "大規模崩塌 黃": int(((h["AlertType"] == "L") & (h["rank"] == 1)).sum())})
            sdf = pd.DataFrame(series).set_index("時間")

            play = False
            if hasattr(st, "fragment"):
                play = st.toggle(f"▶️ 自動播放（每 {PLAY_INTERVAL_S:g} 秒前進一個發布時間點）", value=False)
            idx_manual = st.select_slider("發布時間點", options=list(range(len(times))), value=len(times) - 1,
                                          format_func=lambda i: lab[i])

            def draw_replay():
                if play:
                    cur = st.session_state.get("rp_idx", -1) + 1
                    if cur >= len(times):
                        cur = 0
                    st.session_state["rp_idx"] = cur
                    i = cur
                else:
                    i = idx_manual
                h = counts_upto(times[i])
                nr = int(((h["AlertType"] == "D") & (h["rank"] == 2)).sum())
                ny = int(((h["AlertType"] == "D") & (h["rank"] == 1)).sum())
                lr = int(((h["AlertType"] == "L") & (h["rank"] == 2)).sum())
                ly = int(((h["AlertType"] == "L") & (h["rank"] == 1)).sum())
                st.markdown(f"#### ⏱️ {lab[i]}（第 {i + 1}/{len(times)} 個發布時間點）— 累計至此時曾發布的警戒")
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("土石流 紅色", nr)
                m2.metric("土石流 黃色", ny)
                m3.metric("大規模崩塌 紅色", lr)
                m4.metric("大規模崩塌 黃色", ly)
                lay = []
                if show_sat:
                    mm = satellite_mosaic(119.8, 21.7, 122.3, 25.5, 9)
                    if mm[2] > 0:
                        lay.append(bitmap_layer(mm))
                d = h[h["AlertType"] == "D"].merge(deb[["DebrisNo", "st_id", "st_name", "lat", "lon"]],
                                                   on="DebrisNo", how="left").dropna(subset=["lat", "lon"])
                if not d.empty:
                    rows = []
                    for sid, g in d.groupby("st_id"):
                        n_r, n_y = int((g["rank"] == 2).sum()), int((g["rank"] == 1).sum())
                        f = g.iloc[0]
                        rows.append({"lat": float(f["lat"]), "lon": float(f["lon"]),
                                     "elev": min(1500.0 * (n_r + 0.5 * n_y), 25000.0),
                                     "fill": [230, 30, 30, 235] if n_r else [255, 200, 0, 220],
                                     "title": esc(f"{f['st_name']}（{f['County']}{f['Town']}）"),
                                     "l1": esc(f"紅色 {n_r} 條／黃色 {n_y} 條"), "l2": ""})
                    lay.append(pdk.Layer("ColumnLayer", id="rp-stations", data=pd.DataFrame(rows),
                                         get_position=["lon", "lat"], get_elevation="elev", elevation_scale=1,
                                         radius=1800, get_fill_color="fill", pickable=True))
                lsd2 = ls_layer_df(ls, h.assign(AlertLevel=h["AlertLevel"]))
                if not lsd2.empty:
                    lay.append(pdk.Layer("ScatterplotLayer", id="rp-ls", data=lsd2, get_position=["lon", "lat"],
                                         get_radius=3000, radius_min_pixels=7, get_fill_color="fill",
                                         pickable=True, parameters={"depthTest": False}))
                st.pydeck_chart(deck_for(lay))
                unmatched = int(len(h[h["AlertType"] == "D"]) - len(d))
                if unmatched > 0:
                    st.caption(f"有 {unmatched} 條溪流因編號與現行資料不一致（歷年編號會調整），未標在地圖上。")

            if play:
                st.fragment(run_every=PLAY_INTERVAL_S)(draw_replay)()
            else:
                draw_replay()
            st.markdown("##### 📈 警戒條數隨時間累積")
            st.line_chart(sdf)

# ==========================================
# 7. 分頁③：AI 決策建議
# ==========================================
def build_context():
    d = deb.copy()
    d["ratio_now"] = (d["RT"] / d["AlertValue"]).replace([np.inf, -np.inf], np.nan)
    d["adj_value"] = (d["AlertValue"] - delta).clip(lower=1)
    d["ratio"] = ((d["RT"] + add_rain) / d["adj_value"]).replace([np.inf, -np.inf], np.nan)
    reach = d[d["ratio"] >= 1]
    cty = reach.groupby("County").size().sort_values(ascending=False).head(6)
    cty_txt = "、".join(f"{k} {v} 條" for k, v in cty.items()) or "無"
    top = d.sort_values("ratio", ascending=False).head(10)
    top_txt = "；".join(f"{r.DebrisNo}（{r.County}{r.Town}{r.Vill}）{r.ratio * 100:.0f}%"
                        f"{'，官方' + ('紅' if r.off == 'r' else '黃') + '色警戒' if r.off else ''}"
                        for r in top.itertuples())
    ls_alert = alerts[alerts["AlertType"] == "L"]
    ls_txt = "；".join(f"{(r.LandslideName or r.LandslideID)}（{'紅' if r.AlertLevel == 'r' else '黃'}）"
                       for r in ls_alert.itertuples()) or "無"
    return {
        "d_r": int(((alerts["AlertType"] == "D") & (alerts["AlertLevel"] == "r")).sum()),
        "d_y": int(((alerts["AlertType"] == "D") & (alerts["AlertLevel"] == "y")).sum()),
        "base": int((d["ratio_now"] >= 1).sum()), "scen": int((d["ratio"] >= 1).sum()),
        "cty": cty_txt, "top": top_txt, "ls": ls_txt, "n": len(d),
    }


def call_gemini(prompt, key):
    if genai is None:
        return None, None
    genai.configure(api_key=key)
    names = []
    try:
        for m in genai.list_models():
            n = m.name
            if "generateContent" in m.supported_generation_methods and \
                    not any(x in n for x in ["2.5", "2.0", "1.5", "gemini-pro"]):
                names.append(n)
    except Exception:
        pass
    names.sort(key=lambda n: (0 if "flash" in n else 1, n))
    for n in names:
        try:
            res = genai.GenerativeModel(n).generate_content(prompt)
            if res and res.text:
                return res.text, n
        except Exception:
            continue
    return None, None


with tab3:
    st.markdown("以目前的**官方警戒、雨量與你在分頁①設定的情境**，產生避難勸告與資源預置建議。"
                "AI 可能出錯，**最終的勸告與撤離決定仍由權責機關與人員負責**。")
    st.caption(f"目前情境：警戒基準值調降 {delta} mm、未來 24 小時再降雨 {add_rain} mm"
               + ("（示範資料）" if is_demo else ""))
    if st.button("🚨 啟動 GeoAI 決策推論", use_container_width=True):
        ctx = build_context()
        prompt = f"""
你是臺灣坡地災害防救的資深顧問，協助縣市政府與農村水保署判讀土石流與大規模崩塌風險。
以下是系統彙整的數據（資料{'為示範資料，非真實' if is_demo else '來自農村水保署與中央氣象署開放資料'}）：
- 官方目前已發布土石流警戒：紅色 {ctx['d_r']} 條、黃色 {ctx['d_y']} 條
- 官方大規模崩塌警戒：{ctx['ls']}
- 全國監測溪流 {ctx['n']} 條；目前雨量已達警戒基準值（推估）{ctx['base']} 條
- 情境推演：警戒基準值調降 {delta} mm、未來 24 小時再降雨 {add_rain} mm（保守推估，未計雨量衰減）→ 推估達基準值 {ctx['scen']} 條
- 推估達基準值最多的縣市：{ctx['cty']}
- 最接近或已達基準值的溪流：{ctx['top']}
法規背景：依災害防救法第 24 條，黃色警戒應進行疏散避難勸告；紅色警戒應勸告或強制撤離，並作適當安置。

請只使用上述數據，不要編造未提供的數字或地名；數據不足之處請明確說明不確定性。
請以繁體中文撰寫，嚴格包含以下結構：
1.【風險判讀】：官方警戒與推估之間的差異代表什麼（推估不等於官方警戒），並說明哪些縣市最需要提早準備。
2.【疏散避難勸告建議】：依警戒等級，說明優先勸告的縣市／鄉鎮（只能引用上述資料），以及黃、紅色各應採取的行動。
3.【資源預置】：搶修機具、避難收容、道路與通訊備援的預置重點。
4.【後續監控與重新評估】：哪些數據更新後應重新評估（雨量、警戒發布、地震），以及哪些決定需待更新後再確認。
"""
        with st.spinner("AI Agent 正在綜整警戒、雨量與情境..."):
            text, model_name = (None, None)
            if gemini_key:
                text, model_name = call_gemini(prompt, gemini_key)
        if text:
            st.caption(f"⚡ 使用模型：`{model_name}`")
            st.markdown(text)
        else:
            st.info("💡 " + ("尚未填寫 Gemini API Key" if not gemini_key else "線上 AI 呼叫失敗")
                    + "，以下為依資料自動彙整的備援摘要（非 AI 推論）：")
            st.markdown(f"""
### 坡地災害決策摘要（自動彙整{'・示範資料' if is_demo else ''}）
**1. 風險判讀**
- 官方已發布土石流警戒：紅色 {ctx['d_r']} 條、黃色 {ctx['d_y']} 條；大規模崩塌警戒：{ctx['ls']}。
- 目前雨量已達警戒基準值（推估）{ctx['base']} 條；在「基準值調降 {delta} mm、再降雨 {add_rain} mm」情境下為 {ctx['scen']} 條。
- 推估達標最多的縣市：{ctx['cty']}。
- 最接近或已達基準值的溪流：{ctx['top']}。

**2. 疏散避難勸告**
- 依災害防救法第 24 條：黃色警戒應疏散避難勸告；紅色警戒應勸告或強制撤離並安置。
- 建議優先對上述官方紅、黃色警戒的村里執行；推估達標但尚未發布警戒者，建議提前預置、持續觀察。

**3. 後續重新評估**
- 雨量與警戒資料每數分鐘更新；發生地震或雨勢預報改變時，請調整分頁①的情境重新推演。
""")

st.divider()
st.caption("資料來源：農業部農村發展及水土保持署（土石流及大規模崩塌防災資訊網開放資料）、交通部中央氣象署開放資料平臺；"
           "底圖影像：Esri, Maxar, Earthstar Geographics, and the GIS User Community。本系統為展示用途，非官方預警系統。")
