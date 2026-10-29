"""
把「河川水位測站基本資料」與水利署「警戒水位基本資料」合併成 data/river_alert_stations.csv，
供 slope_twin.py 比對即時河川水位與警戒值。
用法：  python build_alert_stations.py 河川水位測站基本資料.xlsx 警戒水位.json
需要：  pip install pandas openpyxl pyproj
配對：  警戒水位資料沒有測站代號，只能用「水位站名＋流域名稱」對測站基本資料（現存站）；
        對不上的會列出，請人工檢視（不做只靠站名的補配：不同河川常有同名橋，曾因此誤配出假的超警戒）。
座標：  測站座標為 TWD97 TM2 121（EPSG:3826），輸出為 WGS84。
警戒值：一級最高、三級最低；單位公尺，與即時水位同為標高（EL.）——此點以數值比對推斷（166 站的「一級警戒值減即時水位」介於 1.4～15.6 m），尚未由資料說明確認。
        不要用測站基本資料的「高程」欄位做合理性檢查：它與警戒值的基準不一致（甚至有 0.000），會誤殺正確的配對。
"""
import json, sys
import numpy as np
import pandas as pd
from pyproj import Transformer


def main(xlsx, alert_json):
    A = pd.DataFrame(json.load(open(alert_json, encoding="utf-8-sig")))
    for c in ["一級警戒水位", "二級警戒水位", "三級警戒水位"]:
        A[c] = pd.to_numeric(A[c].replace("", np.nan), errors="coerce")
    A["站名"], A["流域"] = A["水位站名"].str.strip(), A["流域名稱"].str.strip()
    S = pd.read_excel(xlsx, sheet_name="水位流量含砂量")
    S = S[S["存廢狀態"] == "現存"].copy()
    S["站號"] = S["站號"].astype(str).str.strip()
    S["站名"], S["流域"] = S["站名"].astype(str).str.strip(), S["流域"].astype(str).str.strip()
    cols = ["站名", "流域", "站號", "河川", "TWD97M2(X坐標)", "TWD97M2(Y坐標)"]
    m = A.merge(S[cols], on=["站名", "流域"], how="left")
    still = m[m["站號"].isna()]
    print(f"警戒水位站 {len(A)} 筆，配到 {len(m) - len(still)} 筆；未配到 {len(still)} 筆：")
    print("、".join(f"{r['站名']}({r['流域']})" for _, r in still.iterrows()))
    m = m.dropna(subset=["站號", "TWD97M2(X坐標)", "TWD97M2(Y坐標)"])
    lon, lat = Transformer.from_crs(3826, 4326, always_xy=True).transform(m["TWD97M2(X坐標)"].values, m["TWD97M2(Y坐標)"].values)
    out = pd.DataFrame({"id": m["站號"].values, "name": m["站名"].values, "basin": m["流域"].values, "river": m["河川"].astype(str).str.strip().values,
                        "lat": np.round(lat, 5), "lon": np.round(lon, 5),
                        "l1": m["一級警戒水位"].values, "l2": m["二級警戒水位"].values, "l3": m["三級警戒水位"].values})
    out.to_csv("data/river_alert_stations.csv", index=False, encoding="utf-8")
    print(f"輸出 {len(out)} 站 → data/river_alert_stations.csv（其中有三級警戒值 {int(out['l3'].notna().sum())} 站）")


if __name__ == "__main__":
    main(*sys.argv[1:3])
