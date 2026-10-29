"""
把水利署的河川流域與主河道 Shapefile 轉成精簡 GeoJSON，供 slope_twin.py 使用。
用法：  python build_rivers.py <流域.shp 路徑> <主河道.shp 路徑>
  例：  python build_rivers.py 河川範圍/basin/basin.shp 主河道/riverlin/riverlin.shp
需要：  pip install pyshp pyproj shapely
產出：  data/river_basins.geojson   全臺 143 個流域（簡化）
        data/main_rivers.geojson   主河道線（只保留有名稱、總長 ≥ 40 km、且位於水庫所在流域者；簡化）
座標：  來源為 TWD97 TM2 121（EPSG:3826），輸出為 WGS84。
欄位：  BASIN_NO／BASIN_NAME（流域）；NAME（主河道名稱）。支流圖層（RIVER_NAME 等）未使用。
"""
import json, sys
import numpy as np
import shapefile
import shapely
from pyproj import Transformer
from shapely.geometry import mapping, shape
from shapely.ops import linemerge

MIN_KM = 40      # 主河道總長低於此值（公里）者不輸出，讓檔案維持精簡
T = Transformer.from_crs(3826, 4326, always_xy=True)
tr = lambda g: shapely.transform(g, lambda c: np.column_stack(T.transform(c[:, 0], c[:, 1])))


def r4(o):
    return [r4(x) for x in o] if isinstance(o, (list, tuple)) else round(o, 4)


def dump(path, feats):
    for f in feats:
        f["geometry"]["coordinates"] = r4(f["geometry"]["coordinates"])
    json.dump({"type": "FeatureCollection", "features": feats}, open(path, "w", encoding="utf-8"),
              ensure_ascii=False, separators=(",", ":"))


def main(basin_shp, river_shp):
    rb = shapefile.Reader(basin_shp, encoding="utf-8")
    basins = [(r["BASIN_NO"], r["BASIN_NAME"], tr(shape(s.__geo_interface__))) for r, s in zip(rb.records(), rb.shapes())]
    dump("data/river_basins.geojson", [
        {"type": "Feature", "properties": {"no": no, "name": nm}, "geometry": mapping(g.simplify(0.001))}
        for no, nm, g in basins])
    # 水庫集水區所在的流域（有重疊者）
    cat = [shape(f["geometry"]) for f in json.load(open("data/reservoir_catchments.geojson", encoding="utf-8"))["features"]]
    cu = shapely.union_all(cat)
    rel = shapely.union_all([g for _, _, g in basins if g.intersects(cu)])
    rl = shapefile.Reader(river_shp, encoding="utf-8")
    feats = []
    for r, s in zip(rl.records(), rl.shapes()):
        name = (r["NAME"] or "").strip()
        if not name:
            continue
        g = tr(shape(s.__geo_interface__))
        if not g.intersects(rel):
            continue
        if g.geom_type == "MultiLineString":
            g = linemerge(shapely.set_precision(g, 0.0001))   # 先對齊到約 10 公尺格點，讓相接的小段能合併
        if g.length * 111 < MIN_KM:          # 略過太短的小溪、圳路，讓檔案維持在約 1 MB 以內
            continue
        g = g.simplify(0.002)
        if not g.is_empty:
            feats.append({"type": "Feature", "properties": {"name": name, "from": (r["FROM"] or "").strip()}, "geometry": mapping(g)})
    dump("data/main_rivers.geojson", feats)
    print(f"流域 {len(basins)} 個；主河道 {len(feats)} 條")


if __name__ == "__main__":
    main(*sys.argv[1:3])
