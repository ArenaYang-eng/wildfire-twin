"""
坡地複合災害決策孿生 — 土石流與大規模崩塌警戒推演
資料來源：農村發展及水土保持署（土石流及大規模崩塌防災資訊網開放資料 API）、
          中央氣象署開放資料平臺（O-A0002-001 雨量觀測）、
          經濟部水利署（水庫水情、水庫每日營運狀況、水庫集水區降雨）
執行：streamlit run slope_twin.py
"""
import base64
import html
import json
import math
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO

import numpy as np
import pandas as pd
import pydeck as pdk
import requests
import shapely
import streamlit as st
from shapely.geometry import shape
from shapely.ops import unary_union
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
# 1b. 水庫資料層（水利署：水庫水情、每日營運狀況、集水區降雨）
# ==========================================
# ★ 下列兩個水利署網址尚未驗證，請對照 https://opendata.wra.gov.tw/openapi/swagger/index.html；
#   可在 Secrets／環境變數以 WRA_REALTIME_URL、WRA_DAILY_URL 覆寫。
# ★ 集水區降雨網址無預設值，請設定 CATCHMENT_URL（未設定時略過，不影響其他功能）。
URL_WRA_REALTIME = "https://opendata.wra.gov.tw/Service/OpenData.aspx?format=json&id=1602CA19-B224-4CC3-AA31-11B86EA07F9C"
URL_WRA_DAILY = "https://opendata.wra.gov.tw/Service/OpenData.aspx?format=json&id=50C8256D-30C5-4B8D-9B84-2E14D5C6DF71"

# 水庫壩址約略座標（估計值，僅供地圖標示，不作精確量測）
RES_GEO = {
    "寶山水庫": (24.74, 121.05), "寶山第二水庫": (24.72, 121.03), "永和山水庫": (24.60, 120.93),
    "鯉魚潭水庫": (24.43, 120.77), "湖山水庫": (23.68, 120.58), "蘭潭水庫": (23.47, 120.48),
    "仁義潭水庫": (23.50, 120.52), "烏山頭水庫": (23.20, 120.40), "澄清湖水庫": (22.65, 120.35),
    "新山水庫": (25.08, 121.72), "翡翠水庫": (24.90, 121.58), "石門水庫": (24.81, 121.25),
    "德基水庫": (24.25, 121.20), "霧社水庫": (24.02, 121.14), "日月潭水庫": (23.85, 120.91),
    "曾文水庫": (23.27, 120.56), "南化水庫": (23.08, 120.50), "牡丹水庫": (22.18, 120.80),
}
# 示範資料（非真實）：取自經濟部簡報 115/10/24 05:00 蓄水率，僅在水利署 API 無法連線時使用
RES_DEMO_PCT = {
    "寶山水庫": 83, "寶山第二水庫": 93, "永和山水庫": 91, "鯉魚潭水庫": 97, "湖山水庫": 99, "蘭潭水庫": 86,
    "仁義潭水庫": 88, "烏山頭水庫": 84, "澄清湖水庫": 87, "新山水庫": 82, "翡翠水庫": 76, "石門水庫": 95,
    "德基水庫": 91, "霧社水庫": 15, "日月潭水庫": 98, "曾文水庫": 93, "南化水庫": 96, "牡丹水庫": 91,
}
RES_COLS = ["id", "name", "pct", "vol", "cap", "obs", "age_h", "lat", "lon", "alat", "alon",
            "rain24", "rain3", "rain_time", "inflow", "totalout", "spill", "desilt", "waterdraw", "statustype",
            "desilt_h", "spill_h", "draw_day", "draw_src", "days", "intake_rain"]


@st.cache_data(ttl=600, show_spinner=False)
def fetch_url_json(url):
    try:
        return get_json(url, timeout=40), ""
    except Exception as e:
        return None, str(e)


def _lower(js):
    df = pd.DataFrame(js)
    df.columns = [str(c).lower() for c in df.columns]
    return df


def parse_catchment(js):
    """水利署集水區降雨（FEWS-PI 逐時格式）：locationId = 'R'+水庫編號；缺值為 missVal(-999)。
    stationName 在實測檔案有亂碼，故不使用；取最後 24 筆有效值加總為近 24 小時雨量。"""
    out = {}
    for ts in (js or {}).get("timeSeries", []):
        h = ts.get("header", {})
        rid = str(h.get("locationId", "")).lstrip("R")
        ev = pd.DataFrame(ts.get("events", []))
        if not rid or ev.empty:
            continue
        ev["v"] = pd.to_numeric(ev["value"], errors="coerce")
        miss = pd.to_numeric(h.get("missVal"), errors="coerce")
        ev = ev[ev["v"].notna() & (ev["v"] >= 0)]
        if pd.notna(miss):
            ev = ev[ev["v"] != miss]
        if ev.empty:
            continue
        end = ev.iloc[-1]
        out[rid] = {
            "rain24": round(float(ev.tail(24)["v"].sum()), 1), "rain3": round(float(ev.tail(3)["v"].sum()), 1),
            "clat": pd.to_numeric(h.get("lat"), errors="coerce"), "clon": pd.to_numeric(h.get("lon"), errors="coerce"),
            "rain_time": f"{end['date']} {str(end['time'])[:5]}",
        }
    return out


def build_reservoirs(rt_js, dy_js, catch):
    """蓄水率 = 即時有效蓄水量 ÷ 有效容量（每日營運資料的 capacity）；上限 100%。
    另保留進出流量、溢洪、排砂、取水量，並統計近 24 小時排砂／溢洪的小時數。
    可支撐天數 = 有效蓄水量 ÷ 日取水量（逐時取水量 cms × 8.64 萬m³/日；無則用每日營運的取用水量），
    僅是「照目前取水速度、不計入流」的粗估，不是供水保證。"""
    rt, dy = _lower(rt_js), _lower(dy_js)
    rt["reservoiridentifier"] = rt["reservoiridentifier"].astype(str)
    dy["reservoiridentifier"] = dy["reservoiridentifier"].astype(str)
    dy["cap"] = pd.to_numeric(dy["capacity"], errors="coerce")
    dy["dayout"] = pd.to_numeric(dy.get("outflow"), errors="coerce")
    dy = dy.sort_values("datetime").drop_duplicates("reservoiridentifier", keep="last")
    idname = dict(zip(dy["reservoiridentifier"], dy["reservoirname"]))
    name2id = {v: k for k, v in idname.items()}
    rt["vol"] = pd.to_numeric(rt["effectivewaterstoragecapacity"], errors="coerce")
    ops = {"inflowdischarge": "inflow", "totaloutflow": "totalout", "spillwayoutflow": "spill",
           "desiltingtunneloutflow": "desilt", "waterdraw": "waterdraw"}
    for src, dst in ops.items():
        rt[dst] = pd.to_numeric(rt[src], errors="coerce") if src in rt else np.nan
    rt["statustype"] = rt["statustype"].astype(str).replace({"nan": "", "None": ""}) if "statustype" in rt else ""
    rt = rt.sort_values("observationtime")
    agg = (rt.groupby("reservoiridentifier").tail(24).groupby("reservoiridentifier")
             .agg(desilt_h=("desilt", lambda x: int((x > 0).sum())), spill_h=("spill", lambda x: int((x > 0).sum()))))
    rt = rt.dropna(subset=["vol"]).drop_duplicates("reservoiridentifier", keep="last")
    df = rt.merge(dy[["reservoiridentifier", "reservoirname", "cap", "dayout"]], on="reservoiridentifier")
    df = df.merge(agg, left_on="reservoiridentifier", right_index=True, how="left")
    df = df[df["reservoirname"].isin(RES_GEO) & (df["cap"] > 0)].copy()
    df["id"], df["name"] = df["reservoiridentifier"], df["reservoirname"]
    df["pct"] = (df["vol"] / df["cap"] * 100).clip(upper=100).round(0)
    df["obs"] = pd.to_datetime(df["observationtime"], errors="coerce")
    now = pd.Timestamp.now(tz="Asia/Taipei").tz_localize(None)
    df["age_h"] = ((now - df["obs"]).dt.total_seconds() / 3600).round(1)
    df["lat"] = df["name"].map(lambda n: RES_GEO[n][0])
    df["lon"] = df["name"].map(lambda n: RES_GEO[n][1])
    df[["desilt_h", "spill_h"]] = df[["desilt_h", "spill_h"]].fillna(0).astype(int)
    hourly = df["waterdraw"].where(df["waterdraw"] > 0) * 8.64
    daily = df["dayout"].where(df["dayout"] > 0)
    df["draw_day"] = hourly.fillna(daily)
    df["draw_src"] = np.where(hourly.notna(), "逐時取水量", np.where(daily.notna(), "每日取用水量", ""))
    df["days"] = (df["vol"] / df["draw_day"]).round(0)
    c = catch or {}
    for k in ["rain24", "rain3", "clat", "clon", "rain_time"]:
        df[k] = df["id"].map(lambda i, k=k: c.get(i, {}).get(k, np.nan))
    def intake_text(n):
        out = []
        for part in CATCH_INTAKES.get(n, []):
            short = part.split("(")[-1].rstrip(")")
            for cand in (part, short):
                i = name2id.get(cand)
                if i and i in c:
                    out.append(f"{short} 近24h {c[i]['rain24']:.0f} mm")
                    break
        return "；".join(out)
    df["intake_rain"] = df["name"].map(intake_text)
    # 關聯用錨點：有集水區中心點就用它（較接近上游），否則退回壩址約略座標
    df["alat"], df["alon"] = df["clat"].fillna(df["lat"]), df["clon"].fillna(df["lon"])
    return df[RES_COLS].reset_index(drop=True)


def res_demo_df():
    rows = [{"id": "DEMO", "name": n, "pct": float(p), "vol": np.nan, "cap": np.nan, "obs": pd.NaT, "age_h": np.nan,
             "lat": RES_GEO[n][0], "lon": RES_GEO[n][1], "alat": RES_GEO[n][0], "alon": RES_GEO[n][1],
             "rain24": np.nan, "rain3": np.nan, "rain_time": np.nan} for n, p in RES_DEMO_PCT.items()]
    d = pd.DataFrame(rows, columns=RES_COLS)
    d[["desilt_h", "spill_h"]] = 0
    d[["draw_src", "intake_rain", "statustype"]] = ""
    return d


def load_reservoirs(force_demo):
    """回傳 (資料表, {來源: 錯誤訊息}, 是否為示範資料)"""
    errs = {}
    if force_demo:
        return res_demo_df(), errs, True
    rt, errs["水庫水情"] = fetch_url_json(secret("WRA_REALTIME_URL") or URL_WRA_REALTIME)
    dy, errs["水庫每日營運"] = fetch_url_json(secret("WRA_DAILY_URL") or URL_WRA_DAILY)
    catch, errs["集水區降雨"] = None, ""
    cu = secret("CATCHMENT_URL")
    if cu:
        cj, errs["集水區降雨"] = fetch_url_json(cu)
        catch = parse_catchment(cj) if cj else None
    else:
        errs["集水區降雨"] = "未設定 CATCHMENT_URL"
    if rt is None or dy is None:
        return res_demo_df(), errs, True
    try:
        df = build_reservoirs(rt, dy, catch)
    except Exception as e:
        errs["水庫水情"] = f"解析失敗：{e}"
        return res_demo_df(), errs, True
    if df.empty:
        errs["水庫水情"] = "資料中找不到對應的水庫"
        return res_demo_df(), errs, True
    return df, errs, False


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
    return d[["lat", "lon", "fill", "title", "l1", "l2", "lv"]]


def deck_for(layers):
    return pdk.Deck(
        layers=layers, map_style=None,
        initial_view_state=pdk.ViewState(latitude=23.7, longitude=120.95, zoom=7.3, pitch=50, bearing=0),
        tooltip={"html": "<b>{title}</b><br/>{l1}<br/>{l2}",
                 "style": {"backgroundColor": "rgba(20,25,35,0.92)", "color": "white"}})


# ==========================================
# 3b. 水庫 × 坡地關聯分析（以半徑近似，非實際集水區）
# ==========================================
# ---- 集水區範圍（水利署「水庫集水區」，建置 2023-05；由 build_catchments.py 自 KML 轉出）----
# 離槽水庫的水主要由攔河堰越域引入，影響水量與濁度的上游包含取水口的集水區。
# 括號寫法的名稱取自水利署圖層本身（例：寶山第二水庫(上坪堰)）；鯉魚潭←士林攔河堰依水利署資料集說明。
# 其他引水關係（例如武界壩→日月潭）尚未納入，可自行在此補充。
CATCH_INTAKES = {
    "鯉魚潭水庫": ["士林攔河堰"],
    "寶山第二水庫": ["寶山第二水庫(上坪堰)"],
    "永和山水庫": ["永和山水庫(田美攔河堰)"],
    "湖山水庫": ["湖山水庫(桶頭堰)"],
    "仁義潭水庫": ["仁義潭水庫(竹山堰)"],
    "日月潭水庫": ["日月潭水庫(栗栖壩)"],
}


@st.cache_resource(show_spinner=False)
def load_catchments():
    """回傳 ({水庫名: {"geom": 準備好的幾何, "parts": 涵蓋範圍文字}}, 供地圖用的 DataFrame)"""
    p = Path(__file__).parent / "data" / "reservoir_catchments.geojson"
    if not p.exists():
        return {}, pd.DataFrame()
    feats = json.load(open(p, encoding="utf-8"))["features"]
    by = {f["properties"]["name"]: shape(f["geometry"]) for f in feats}
    out, rows = {}, []
    for name in RES_GEO:
        parts = [n for n in [name] + CATCH_INTAKES.get(name, []) if n in by]
        if not parts:
            continue
        g = unary_union([by[n] for n in parts])
        shapely.prepare(g)
        out[name] = {"geom": g, "parts": "、".join(parts)}
        for n in parts:
            for poly in (by[n].geoms if hasattr(by[n], "geoms") else [by[n]]):
                rows.append({"polygon": [list(c) for c in poly.exterior.coords], "title": esc(n),
                             "l1": esc("集水區（取水口）" if n != name else "集水區"), "l2": esc(f"屬於：{name}"),
                             "intake": n != name})
    return out, pd.DataFrame(rows)


def haversine_km(lat1, lon1, lat2, lon2):
    p = np.pi / 180
    a = np.sin((lat2 - lat1) * p / 2) ** 2 + np.cos(lat1 * p) * np.cos(lat2 * p) * np.sin((lon2 - lon1) * p / 2) ** 2
    return 12742 * np.arcsin(np.sqrt(a))


def in_upstream(r_name, alat, alon, df, radius, catch):
    """回傳 df 中落在該水庫上游的列。有集水區多邊形（含離槽水庫取水口）就用多邊形，否則用半徑近似。"""
    if df is None or df.empty:
        return df
    c = catch.get(r_name)
    if c is not None:
        return df[shapely.contains_xy(c["geom"], df["lon"].values, df["lat"].values)]
    return df[haversine_km(alat, alon, df["lat"].values, df["lon"].values) <= radius]


def link_slopes_to_reservoirs(res, deb_s, lsd, radius, low, catch=None):
    """對每座水庫統計上游的溪流（以參考雨量站座標近似）與大規模崩塌潛勢區。
    有水利署集水區多邊形的水庫用多邊形判斷（離槽水庫含取水口集水區），其餘以半徑近似。
    關注等級為透明的規則式分類（非官方指標）：上游有官方紅色警戒 ＝ 2；黃色警戒或推估達標 ＝ 1；
    上述任一成立且蓄水率低於門檻再 +1。"""
    catch = catch or {}
    dd = deb_s.dropna(subset=["lat", "lon", "ratio"]) if deb_s is not None and "lat" in deb_s else pd.DataFrame()
    rows, pairs = [], []
    for r in res.itertuples():
        c = catch.get(r.name)
        x = dict(n_deb=0, n_reach=0, deb_r=0, deb_y=0, max_ratio=np.nan, ls_n=0, ls_r=0, ls_y=0, county="",
                 method="集水區多邊形" if c else "半徑近似", scope=c["parts"] if c else f"中心點半徑 {radius} km")
        if not dd.empty:
            near = in_upstream(r.name, r.alat, r.alon, dd, radius, catch)
            x.update(n_deb=len(near), n_reach=int((near["ratio"] >= 1).sum()),
                     deb_r=int((near["off"] == "r").sum()), deb_y=int((near["off"] == "y").sum()),
                     max_ratio=float(near["ratio"].max()) if len(near) else np.nan,
                     county="、".join(sorted(near["County"].dropna().unique())[:3]))
            for _, g in near.groupby("st_id"):
                mr = float(g["ratio"].max())
                off = g["off"].isin(["r", "y"]).any()
                if mr >= 0.8 or off:
                    f = g.iloc[0]
                    pairs.append({"slon": r.lon, "slat": r.lat, "tlon": float(f["lon"]), "tlat": float(f["lat"]),
                                  "c": [230, 30, 30, 170] if (g["off"] == "r").any() else [255, 140, 0, 130]})
        if lsd is not None and not lsd.empty:
            m = in_upstream(r.name, r.alat, r.alon, lsd, radius, catch)
            x.update(ls_n=len(m), ls_r=int((m["lv"] == "r").sum()), ls_y=int((m["lv"] == "y").sum()))
        rows.append(x)
    link = pd.concat([res.reset_index(drop=True), pd.DataFrame(rows)], axis=1)
    red = link["deb_r"] + link["ls_r"]
    yel = link["deb_y"] + link["ls_y"]
    base = np.where(red > 0, 2, np.where((yel > 0) | (link["n_reach"] > 0), 1, 0))
    lowp = link["pct"] < low
    link["rank"] = base + np.where((base > 0) & lowp, 1, 0)
    def flags(i):
        f = []
        if red[i] > 0: f.append("上游紅色警戒")
        elif yel[i] > 0: f.append("上游黃色警戒")
        if link.at[i, "n_reach"] > 0: f.append("推估達標溪流")
        if lowp[i]: f.append("蓄水偏低")
        if pd.notna(link.at[i, "age_h"]) and link.at[i, "age_h"] > 24: f.append("資料過期")
        return "、".join(f) or "—"
    link["flags"] = [flags(i) for i in range(len(link))]
    def opstate(r):
        f = []
        if pd.notna(r["desilt"]) and r["desilt"] > 0: f.append("排砂中")
        elif r["desilt_h"] > 0: f.append(f"近24h排砂{r['desilt_h']}h")
        if pd.notna(r["spill"]) and r["spill"] > 0: f.append("溢洪中")
        elif r["spill_h"] > 0: f.append(f"近24h溢洪{r['spill_h']}h")
        return "、".join(f) or "—"
    link["ops"] = link.apply(opstate, axis=1)
    return link, pd.DataFrame(pairs).head(300)


def res_layer_df(link):
    lo, hi = np.array([190, 215, 245]), np.array([10, 80, 170])   # 蓄水率以藍色系表示，避免與警戒燈號混淆
    d = link.copy()
    d["fill"] = d["pct"].apply(lambda p: [int(v) for v in (lo + (hi - lo) * float(np.clip(p, 0, 100)) / 100)] + [240])
    d["title"] = d["name"].apply(lambda n: esc(f"水庫：{n}"))
    d["l1"] = d.apply(lambda r: esc(f"蓄水率 {r['pct']:.0f}%" + (f"｜觀測 {r['obs']:%m/%d %H:%M}" if pd.notna(r["obs"]) else "｜示範")
                                    + (f"｜集水區近24h {r['rain24']:.0f} mm" if pd.notna(r["rain24"]) else "")
                                    + (f"｜排砂{r['desilt']:.0f} cms" if pd.notna(r["desilt"]) and r["desilt"] > 0 else "")
                                    + (f"｜溢洪{r['spill']:.1f} cms" if pd.notna(r["spill"]) and r["spill"] > 0 else "")), axis=1)
    d["l2"] = d.apply(lambda r: esc(f"上游溪流 {r['n_deb']}（官方紅{r['deb_r']}／黃{r['deb_y']}、推估達標{r['n_reach']}）｜崩塌區 紅{r['ls_r']}／黃{r['ls_y']}"), axis=1)
    return d


# ==========================================
# 4. 側邊欄與資料載入
# ==========================================
# 直接讀取金鑰，不放進網頁輸入框
cwa_key = secret("CWA_API_KEY")
gemini_key = secret("GEMINI_API_KEY")

# 展示頁面只保留非敏感設定
st.sidebar.header("⚙️ 展示設定")

show_sat = st.sidebar.checkbox(
    "顯示衛星空照底圖",
    value=True,
)

force_demo = st.sidebar.checkbox(
    "強制使用示範資料（非真實）",
    value=False,
)

st.sidebar.markdown("**水庫 × 坡地關聯**")
radius_km = st.sidebar.slider("備援關聯半徑（km）", 5, 40, 15, 5,
                              help="只用於沒有水利署集水區多邊形的水庫；有多邊形的水庫直接用多邊形判斷。")
low_pct = st.sidebar.slider("蓄水偏低門檻（%）", 10, 60, 30, 5, help="僅為展示用的篩選門檻，非官方標準。")

# 只顯示設定狀態，不顯示金鑰內容
st.sidebar.caption(
    "氣象署 API：已設定" if cwa_key else "氣象署 API：未設定"
)
st.sidebar.caption(
    "Gemini API：已設定" if gemini_key else "Gemini API：未設定"
)

# 以下維持原本資料取得邏輯
alerts, err_a = fetch_alerts()
deb, err_d = fetch_debris_rain()
ls, err_l = fetch_ls_values()

cwa, err_c = (
    fetch_cwa_rain(cwa_key)
    if cwa_key
    else (None, "未提供氣象署授權碼")
)

is_demo = force_demo or deb is None

if is_demo:
    alerts, deb, ls, cwa = make_demo()

if alerts is None:
    alerts = _alert_df([])

if ls is None:
    ls = pd.DataFrame(
        columns=[
            "County",
            "Town",
            "LSNo",
            "Name",
            "AlertValue",
            "Lng",
            "Lat",
        ]
    )

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

res, res_errs, res_is_demo = load_reservoirs(force_demo)
catch, catch_df = load_catchments()
with st.expander("🚰 水庫資料來源狀態"):
    if res_is_demo:
        st.write("水庫為**示範資料（非真實，取自簡報蓄水率）**。" + ("" if force_demo else "原因：" + "；".join(f"{k}：{v}" for k, v in res_errs.items() if v)))
    else:
        st.markdown("\n".join(f"- {'✅' if not v else '⚠️'} {k}：{(str(len(res)) + ' 座') if (k == '水庫水情' and not v) else (v or '正常')}"
                              for k, v in res_errs.items()))

tab1, tab2, tab3, tab4 = st.tabs(["① 即時態勢與情境推演", "② 歷史事件回放", "③ AI 決策建議", "④ 水庫與供水連動"])

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


_LINK_CACHE = {}


def scenario_deb():
    d = deb.copy()
    d["ratio"] = ((d["RT"] + add_rain) / (d["AlertValue"] - delta).clip(lower=1)).replace([np.inf, -np.inf], np.nan)
    return d


def get_link():
    if "v" not in _LINK_CACHE:
        _LINK_CACHE["v"] = link_slopes_to_reservoirs(res, scenario_deb(), ls_layer_df(ls, alerts), radius_km, low_pct, catch)
    return _LINK_CACHE["v"]


def build_res_context():
    link, _ = get_link()
    cand = link[link["rank"] > 0].sort_values(["rank", "pct"], ascending=[False, True]).head(6)
    parts = []
    for r in cand.itertuples():
        rain = f"，集水區近24小時 {r.rain24:.0f} mm" if pd.notna(r.rain24) else ""
        parts.append(f"{r.name}（蓄水率 {r.pct:.0f}%{rain}；上游溪流 {r.n_deb} 條、推估達標 {r.n_reach} 條、"
                     f"官方紅{r.deb_r}／黃{r.deb_y}，崩塌區紅{r.ls_r}／黃{r.ls_y}；{r.flags}）")
    low = link[link["pct"] < low_pct]["name"].tolist()
    opsl = [f"{r.name}（{r.ops}）" for r in link.itertuples() if r.ops != "—"]
    short = [f"{r.name}約 {r.days:.0f} 天" for r in link.dropna(subset=["days"]).sort_values("days").head(3).itertuples()]
    return {"text": "；".join(parts) or "目前沒有水庫上游出現警戒或推估達標",
            "low": "、".join(low) or "無", "n": len(link),
            "ops": "、".join(opsl) or "無", "short": "、".join(short) or "無資料"}


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
        rctx = build_res_context()
        prompt = f"""
你是臺灣坡地災害防救的資深顧問，協助縣市政府與農村水保署判讀土石流與大規模崩塌風險。
以下是系統彙整的數據（資料{'為示範資料，非真實' if is_demo else '來自農村水保署與中央氣象署開放資料'}）：
- 官方目前已發布土石流警戒：紅色 {ctx['d_r']} 條、黃色 {ctx['d_y']} 條
- 官方大規模崩塌警戒：{ctx['ls']}
- 全國監測溪流 {ctx['n']} 條；目前雨量已達警戒基準值（推估）{ctx['base']} 條
- 情境推演：警戒基準值調降 {delta} mm、未來 24 小時再降雨 {add_rain} mm（保守推估，未計雨量衰減）→ 推估達基準值 {ctx['scen']} 條
- 推估達基準值最多的縣市：{ctx['cty']}
- 最接近或已達基準值的溪流：{ctx['top']}
- 水庫面（{'示範資料，非真實' if res_is_demo else '水利署水庫水情，蓄水率＝有效蓄水量÷有效容量'}；共 {rctx['n']} 座）：蓄水率低於 {low_pct}% 的水庫：{rctx['low']}
- 水庫與上游坡地關聯（以水利署水庫集水區多邊形判斷，離槽水庫含取水口集水區；無多邊形者以半徑 {radius_km} km 近似；溪流位置為參考雨量站座標的近似）：{rctx['text']}
- 水庫運轉：排砂或溢洪的水庫：{rctx['ops']}；以目前取水速度粗估可支撐天數最短的水庫（不計入流，非供水保證）：{rctx['short']}
法規背景：依災害防救法第 24 條，黃色警戒應進行疏散避難勸告；紅色警戒應勸告或強制撤離，並作適當安置。

請只使用上述數據，不要編造未提供的數字或地名；數據不足之處請明確說明不確定性。
請以繁體中文撰寫，嚴格包含以下結構：
1.【風險判讀】：官方警戒與推估之間的差異代表什麼（推估不等於官方警戒），並說明哪些縣市最需要提早準備。
2.【疏散避難勸告建議】：依警戒等級，說明優先勸告的縣市／鄉鎮（只能引用上述資料），以及黃、紅色各應採取的行動。
3.【資源預置】：搶修機具、避難收容、道路與通訊備援的預置重點。
4.【後續監控與重新評估】：哪些數據更新後應重新評估（雨量、警戒發布、地震），以及哪些決定需待更新後再確認。
5.【水源與供水影響評估】：僅依上述水庫蓄水率、集水區雨量與上游警戒，指出哪些水庫需優先關注及原因；濁度、淨水場與供水區資料並未提供，且防洪運轉狀態代碼意義未知，不得解讀，請明確說明無法據此評估供水中斷風險，不得推測。
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

**3. 水庫與上游坡地**
- 蓄水率低於 {low_pct}% 的水庫：{rctx['low']}。
- 需關注：{rctx['text']}。
- 本系統沒有濁度、取水量與供水區資料，無法評估供水中斷風險。

**4. 後續重新評估**
- 雨量與警戒資料每數分鐘更新；發生地震或雨勢預報改變時，請調整分頁①的情境重新推演。
""")

# ==========================================
# 8. 分頁④：水庫與供水連動
# ==========================================
with tab4:
    st.markdown("**這一頁回答：上游坡地的雨量與警戒，會不會影響水庫？**　"
                "把每座水庫上游集水區內的土石流溪流、大規模崩塌區放在一起看（水利署集水區多邊形；離槽水庫含取水口集水區）。"
                "溪流位置仍是參考雨量站座標的近似，且**無法評估供水是否中斷**（缺濁度、取水量、供水區資料，見下方說明）。")
    if res_is_demo:
        st.warning("⚠️ 水庫蓄水率目前是**示範資料（非真實）**。")
    if is_demo:
        st.warning("⚠️ 坡地警戒與雨量目前是**示範資料（非真實）**，下方的關聯結果不代表實際狀況。")
    link, pairs = get_link()
    ds = scenario_deb()
    if not has_xy:
        st.info("沒有雨量站座標，無法關聯溪流（仍可關聯大規模崩塌潛勢區）。請設定氣象署授權碼。")

    n_alert = int((link["rank"] >= 1).sum())
    n_prio = int((link["rank"] >= 2).sum())
    n_low = int((link["pct"] < low_pct).sum())
    n_stale = int((link["age_h"] > 24).sum())
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("水庫數", len(link))
    m2.metric("上游有警戒或推估達標", n_alert)
    m3.metric("關注等級 ≥ 2（含紅色警戒）", n_prio)
    m4.metric(f"蓄水率 < {low_pct}%／資料過期", f"{n_low} ／ {n_stale}")

    show_catch = st.checkbox("顯示集水區範圍", value=True)
    layers = []
    if show_sat:
        mm = satellite_mosaic(119.8, 21.7, 122.3, 25.5, 9)
        if mm[2] > 0:
            layers.append(bitmap_layer(mm))
    if show_catch and not catch_df.empty:
        layers.append(pdk.Layer("PolygonLayer", id="catchments", data=catch_df, get_polygon="polygon",
                                get_fill_color=[30, 100, 200, 28], get_line_color=[30, 100, 200, 200],
                                line_width_min_pixels=1, stroked=True, filled=True, pickable=True,
                                parameters={"depthTest": False}))
    stn4 = aggregate_stations(ds) if has_xy else pd.DataFrame()
    if not stn4.empty:
        layers.append(pdk.Layer("ColumnLayer", id="stations4", data=stn4, get_position=["lon", "lat"],
                                get_elevation="elev", elevation_scale=1, radius=1800, get_fill_color="fill",
                                pickable=True, auto_highlight=True))
    lsd4 = ls_layer_df(ls, alerts)
    if not lsd4.empty:
        layers.append(pdk.Layer("ScatterplotLayer", id="landslides4", data=lsd4, get_position=["lon", "lat"],
                                get_radius=3000, radius_min_pixels=7, get_fill_color="fill", pickable=True,
                                parameters={"depthTest": False}))
    if not pairs.empty:
        layers.append(pdk.Layer("ArcLayer", id="links", data=pairs, get_source_position=["slon", "slat"],
                                get_target_position=["tlon", "tlat"], get_source_color="c", get_target_color="c",
                                get_width=2))
    rl = res_layer_df(link)
    hal = rl[rl["rank"] >= 2]
    if not hal.empty:
        layers.append(pdk.Layer("ScatterplotLayer", id="res-halo", data=hal, get_position=["lon", "lat"],
                                get_radius=9000, radius_min_pixels=16, get_fill_color=[230, 30, 30, 70],
                                parameters={"depthTest": False}))
    layers.append(pdk.Layer("ScatterplotLayer", id="reservoirs", data=rl, get_position=["lon", "lat"],
                            get_radius=4500, radius_min_pixels=11, get_fill_color="fill", get_line_color=[255, 255, 255, 255],
                            stroked=True, line_width_min_pixels=2, pickable=True, parameters={"depthTest": False}))
    st.pydeck_chart(deck_for(layers))
    st.caption("藍色圓點＝水庫（越深蓄水率越高；蓄水率不用綠黃紅，避免與水情燈號混淆）；紅色外圈＝關注等級 ≥ 2。"
               "弧線連到上游「已達 80%／有官方警戒」的雨量站（紅＝含官方紅色警戒）。柱與崩塌區點的意義同分頁①。"
               "藍色半透明範圍＝水利署集水區（含離槽水庫取水口集水區）；水庫圓點位置為壩址約略座標（估計值，非蓄水範圍中心）。沒有集水區多邊形的水庫改以半徑近似，表格會標示判斷方式。")

    st.markdown("##### 🚰 水庫 × 上游坡地一覽（依關注等級排序）")
    tb = link.sort_values(["rank", "pct"], ascending=[False, True])
    st.dataframe(pd.DataFrame({
        "水庫": tb["name"], "關注等級": tb["rank"].map({3: "3 優先", 2: "2 關注", 1: "1 留意", 0: "—"}),
        "原因": tb["flags"], "運轉狀態": tb["ops"], "判斷方式": tb["method"], "涵蓋範圍": tb["scope"], "蓄水率 %": tb["pct"].astype("Int64"),
        "可支撐天數(粗估)": tb["days"].map(lambda v: "—" if pd.isna(v) else (">999" if v > 999 else f"{v:.0f}")),
        "天數依據": tb["draw_src"].replace("", "—"),
        "觀測時間": tb["obs"].dt.strftime("%m/%d %H:%M").fillna("—"),
        "集水區近3h (mm)": tb["rain3"], "集水區近24h (mm)": tb["rain24"], "取水口集水區降雨": tb["intake_rain"].replace("", "—"),
        "上游溪流": tb["n_deb"], "推估達標": tb["n_reach"],
        "官方紅": tb["deb_r"], "官方黃": tb["deb_y"], "崩塌區 紅/黃": tb["ls_r"].astype(str) + "／" + tb["ls_y"].astype(str),
        "涉及縣市": tb["county"],
    }), hide_index=True, use_container_width=True)
    st.caption(f"關注等級規則（透明、可在側邊欄調整，非官方指標）：上游有官方紅色警戒＝2；有黃色警戒或推估達標溪流＝1；"
               f"以上任一成立且蓄水率 < {low_pct}% 再加 1。官方警戒數以溪流筆數計；同一條溪流落在多座水庫的半徑內時，會分別計入各水庫。")

    st.markdown("##### 🔍 單一水庫：上游最接近基準值的溪流")
    pick = st.selectbox("選擇水庫", link["name"].tolist())
    pr = link[link["name"] == pick].iloc[0]
    if has_xy:
        dd4 = ds.dropna(subset=["lat", "lon", "ratio"])
        nr = in_upstream(pick, pr["alat"], pr["alon"], dd4, radius_km, catch)
        nr = nr.sort_values("ratio", ascending=False).head(10)
        if nr.empty:
            st.caption("上游沒有可關聯的溪流。")
        else:
            st.dataframe(pd.DataFrame({
                "溪流編號": nr["DebrisNo"], "縣市": nr["County"], "鄉鎮": nr["Town"], "參考雨量站": nr["STName1"],
                "推估百分比": (nr["ratio"] * 100).round(0).astype("Int64").astype(str) + "%",
                "官方警戒": nr["off"].map({"r": "🔴 紅色", "y": "🟡 黃色"}).fillna("—")}),
                hide_index=True, use_container_width=True)
    det = []
    if pd.notna(pr["inflow"]) or pd.notna(pr["totalout"]):
        det.append(f"進流量 {pr['inflow']:.1f} cms、總出流量 {pr['totalout']:.1f} cms" if pd.notna(pr["inflow"]) and pd.notna(pr["totalout"])
                   else "進出流量資料不完整")
    if pr["ops"] != "—": det.append(f"運轉：{pr['ops']}")
    if pd.notna(pr["days"]): det.append(f"依{pr['draw_src']}粗估可支撐約 {pr['days']:.0f} 天（不計入流，非供水保證）")
    if pr["statustype"]: det.append(f"防洪運轉狀態代碼：{pr['statustype']}（代碼意義待確認，不做解讀）")
    if pr["intake_rain"]: det.append(f"取水口集水區降雨：{pr['intake_rain']}")
    if det: st.caption("　｜　".join(det))
    if pd.notna(pr["rain24"]):
        st.caption(f"{pick} 集水區平均雨量：近 3 小時 {pr['rain3']:.1f} mm、近 24 小時 {pr['rain24']:.1f} mm（最後有效時段 {pr['rain_time']}）。")

    with st.expander("⚠️ 這個分析還缺什麼資料（所以不能回答「供水會不會中斷」）"):
        st.markdown("""
- **其他引水關係**：已納入 6 座離槽水庫的取水口集水區（鯉魚潭←士林攔河堰、寶山第二←上坪堰、永和山←田美攔河堰、湖山←桶頭堰、仁義潭←竹山堰、日月潭←栗栖壩）。水利署圖層另有武界壩等，與水庫的引水關係未納入，可在 `CATCH_INTAKES` 補充；阿公店←月眉堰等不在目前清單。
- **溪流與崩塌區的實際座標**：溪流位置目前是參考雨量站座標的近似，靠近集水區邊界時可能誤判；集水區圖資為 2023 年建置，且每 10 年左右才檢討。
- **水質／濁度**：坡地崩塌影響供水的主要途徑是濁度與泥砂，需水庫入庫水質或淨水場原水濁度。
- **供水區對照**：哪座水庫供應哪些淨水場、哪些用戶，需自來水公司或水利署的供水系統資料；澄清湖與澎湖水庫不在目前的水利署資料集內。
- **定量降水預報**：情境推演的「未來 24 小時再降雨」目前是手動輸入，可改接氣象署預報。
- **防洪運轉狀態代碼（statustype）的意義**：資料有 -1、3、4 等值，需水利署代碼說明才能解讀。
- **河川水位的警戒值與歷史序列**：目前河川即時水位只有最新一筆且缺警戒水位，無法判斷高低；集水區內的測站也很少。
- **集水區降雨資料的固定網址與更新頻率**：目前只有你提供的檔案，需設定 `CATCHMENT_URL`。
""")

st.divider()
st.caption("資料來源：農業部農村發展及水土保持署（土石流及大規模崩塌防災資訊網開放資料）、交通部中央氣象署開放資料平臺、經濟部水利署（水庫水情、集水區降雨）；"
           "底圖影像：Esri, Maxar, Earthstar Geographics, and the GIS User Community。本系統為展示用途，非官方預警系統。")
