# slope_twin（坡地數位孿生 × 水庫與供水連動）

```
slope_twin.py                       Streamlit 主程式（分頁④ 為水庫與供水連動）
requirements.txt                    相依套件（新增 shapely>=2.0）
data/reservoir_catchments.geojson   水利署水庫集水區（80 筆，含離槽水庫取水口；建置 2023-05）
data/river_basins.geojson           水利署河川流域範圍（143 個，簡化）
data/main_rivers.geojson            水利署主河道（總長 ≥ 40 km、位於水庫所在流域者，約 1 MB，地圖上預設不顯示）
data/river_alert_stations.csv       河川水位站（166 站，含一～三級警戒值與座標）
build_catchments.py                 由水庫集水區 KML 重新產生 reservoir_catchments.geojson
build_alert_stations.py             由測站基本資料 xlsx 與警戒水位 JSON 重新產生 river_alert_stations.csv（需 pandas、openpyxl、pyproj）
build_rivers.py                     由流域與主河道 Shapefile 重新產生後兩個 GeoJSON（需 pyshp、pyproj、shapely）
```

- 部署時 `data/` 資料夾必須和 `slope_twin.py` 放在一起。
- Secrets／環境變數：氣象署與水保署金鑰同原專案；新增 `CATCHMENT_URL`（集水區降雨，選用）、`WRA_REALTIME_URL`／`WRA_DAILY_URL`（水利署網址，選用，預設值尚未驗證）、`RIVER_STAGE_URL`（河川即時水位資料網址，選用；未設定時略過水位比對）。
- 重新產生河川資料：`python build_rivers.py 河川範圍/basin/basin.shp 主河道/riverlin/riverlin.shp`
- 重新產生集水區：`python build_catchments.py RESERVOIR.kml`
- 「支流」圖層（河川區域多邊形）尚未使用；主河道是零碎線段，無法判斷上下游。
