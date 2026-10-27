"""
把水利署「水庫集水區」KML 轉成精簡的 GeoJSON，供 slope_twin.py 做空間判斷。
用法：  python build_catchments.py RESERVOIR.kml        （需要 shapely>=2.0）
產出：  data/reservoir_catchments.geojson
來源：  水利空間資訊服務平台「水庫集水區」，KML 座標為 WGS84；簡化容許誤差約 30 公尺。
"""
import json, sys
import xml.etree.ElementTree as ET
from shapely.geometry import MultiPolygon, Polygon, mapping

K = "{http://www.opengis.net/kml/2.2}"
TOL = 0.0003        # 度，約 30 公尺


def ring(text):
    return [tuple(map(float, p.split(",")[:2])) for p in text.split()]


def main(path):
    root = ET.parse(path).getroot()
    feats = []
    for pm in root.iter(K + "Placemark"):
        name = (pm.findtext(K + "name") or "").strip()
        sd = {e.get("name"): (e.text or "") for e in pm.iter(K + "SimpleData")}
        polys = []
        for pg in pm.iter(K + "Polygon"):
            outer = ring(pg.find(f".//{K}outerBoundaryIs//{K}coordinates").text)
            holes = [ring(h.find(f".//{K}coordinates").text) for h in pg.findall(f".//{K}innerBoundaryIs")]
            polys.append(Polygon(outer, holes))
        if not polys:
            continue
        g = MultiPolygon(polys) if len(polys) > 1 else polys[0]
        g = g.buffer(0).simplify(TOL, preserve_topology=True)
        feats.append({"type": "Feature", "geometry": json.loads(json.dumps(mapping(g))),
                      "properties": {"name": name, "class": sd.get("CLASS_1", ""), "state": sd.get("STATE", ""),
                                     "unit": sd.get("UNIT", ""), "type": sd.get("TYPE", ""),
                                     "area_km2": round(float(sd.get("area", 0) or 0) / 1e6, 2)}})

    def r5(o):
        return [r5(x) for x in o] if isinstance(o, (list, tuple)) else round(o, 5)
    for f in feats:
        f["geometry"]["coordinates"] = r5(f["geometry"]["coordinates"])
    out = {"type": "FeatureCollection", "features": feats}
    json.dump(out, open("data/reservoir_catchments.geojson", "w", encoding="utf-8"), ensure_ascii=False, separators=(",", ":"))
    print(f"{len(feats)} 筆 → data/reservoir_catchments.geojson")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "RESERVOIR.kml")
