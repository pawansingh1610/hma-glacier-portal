"""
Glacier Data Portal — Flask Backend CLOUD VERSION v1
Data loaded from Hugging Face: pawansingh1610/hma-glacier-data
DO NOT REPLACE WITH LOCAL VERSION
"""

import os, math, warnings, traceback, time, pickle, hashlib
import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Point
from flask import Flask, jsonify, request, send_from_directory, Response
from flask_cors import CORS
from urllib.parse import unquote
from huggingface_hub import snapshot_download

warnings.filterwarnings("ignore")

# ─── DOWNLOAD DATA FROM HUGGING FACE ──────────────────────────────────────
HF_REPO    = "pawansingh1610/hma-glacier-data"
CACHE_ROOT = "/tmp/glacier_data"

def ensure_data():
    marker = os.path.join(CACHE_ROOT, ".downloaded")
    if os.path.exists(marker):
        print("✓ Data already cached at", CACHE_ROOT)
        return
    print("⬇ Downloading data from Hugging Face (~600MB)… please wait")
    t0 = time.time()
    snapshot_download(
        repo_id=HF_REPO,
        repo_type="dataset",
        local_dir=CACHE_ROOT,
        ignore_patterns=["*.pkl"],   # skip old cache files
    )
    open(marker, "w").close()
    print(f"✓ Download complete in {time.time()-t0:.0f}s")

ensure_data()

# ─── PATHS (now pointing to /tmp/glacier_data) ────────────────────────────
DATA_DIR     = os.path.join(CACHE_ROOT, "data")
WIDE_DIR     = os.path.join(DATA_DIR, "wide_panels")
RGI_BASE     = os.path.join(CACHE_ROOT, "rgi")
HMA_SHP      = os.path.join(CACHE_ROOT, "hma", "HMA_region.shp")
META_CSV     = os.path.join(DATA_DIR, "ALL_GLACIER_METADATA.csv")
YEARLY_CSV   = os.path.join(DATA_DIR, "ALL_YEARLY_DATA.csv")
FIELD_MB_DIR = os.path.join(CACHE_ROOT, "field_mb")
CACHE_DIR    = os.path.join(CACHE_ROOT, ".portal_cache")
os.makedirs(CACHE_DIR, exist_ok=True)

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")

import json

class NaNSafeEncoder(json.JSONEncoder):
    def encode(self, obj):
        return super().encode(self._fix(obj))
    def _fix(self, obj):
        if isinstance(obj, float):
            if math.isnan(obj) or math.isinf(obj): return None
            return obj
        if isinstance(obj, dict):  return {k: self._fix(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)): return [self._fix(v) for v in obj]
        return obj

app = Flask(__name__, static_folder=STATIC_DIR)
app.json_encoder = NaNSafeEncoder
CORS(app)

def safe(val):
    if val is None: return None
    try:
        f = float(val)
        if math.isnan(f) or math.isinf(f): return None
        return f
    except (ValueError, TypeError): return None

def safe_str(val, fallback=""):
    if val is None: return fallback
    try:
        if isinstance(val, float) and (math.isnan(val) or math.isinf(val)): return fallback
        s = str(val).strip()
        return s if s and s.lower() not in ("nan","none","") else fallback
    except: return fallback

def timed(label):
    class Timer:
        def __enter__(self): self.t = time.time(); return self
        def __exit__(self, *a): print(f"  ✓ {label}: {time.time()-self.t:.2f}s")
    return Timer()

# ═══════════════════════════════════════════════════════════════════════════
#  LOAD WIDE PANELS
# ═══════════════════════════════════════════════════════════════════════════
PANEL_FILES = {
    "mb_annual_mwe":     "mb_annual_mwe_wide.csv",
    "mb_cumulative_mwe": "mb_cumulative_mwe_wide.csv",
    "volume_km3":        "volume_km3_wide.csv",
    "area_km2":          "area_km2_wide.csv",
    "thickness_m":       "thickness_m_wide.csv",
    "length_km":         "length_km_wide.csv",
    "ela_m":             "ela_m_wide.csv",
    "aar":               "aar_wide.csv",
}

print("═"*60); print("  LOADING WIDE PANELS"); print("═"*60)
panels = {}
all_glacier_ids = set()

with timed("Load all panels"):
    for field, fname in PANEL_FILES.items():
        fpath = os.path.join(WIDE_DIR, fname)
        if not os.path.isfile(fpath): continue
        df = pd.read_csv(fpath, index_col=0)
        sample_idx = str(df.index[0]) if len(df.index) else ""
        if sample_idx.startswith("RGI"): df = df.T
        df.index = pd.to_numeric(df.index, errors="coerce")
        df = df[df.index.notna()]
        df.index = df.index.astype(int)
        df.index.name = "year"
        df = df.apply(pd.to_numeric, errors="coerce")
        panels[field] = df
        all_glacier_ids |= set(df.columns)
        print(f"    {field}: {df.shape[1]:,} glaciers × {df.shape[0]} yrs")

print(f"  Total unique glacier IDs: {len(all_glacier_ids):,}")
_panel_glacier_sets = {field: set(df.columns) for field, df in panels.items()}

# ── Names ──
print("Loading glacier names …")
_name_map = {}
with timed("Names"):
    if os.path.isfile(META_CSV):
        try:
            meta = pd.read_csv(META_CSV, usecols=lambda c: c in ["rgi_id","glacier_name","RGIId","Name"], low_memory=False)
            id_col = "rgi_id" if "rgi_id" in meta.columns else "RGIId"
            nm_col = "glacier_name" if "glacier_name" in meta.columns else "Name"
            if id_col in meta.columns and nm_col in meta.columns:
                sub = meta[[id_col,nm_col]].dropna().drop_duplicates(id_col)
                _name_map = dict(zip(sub[id_col], sub[nm_col]))
        except: pass
    print(f"    {len(_name_map):,} names loaded")

# ═══════════════════════════════════════════════════════════════════════════
#  FIELD MASS BALANCE
# ═══════════════════════════════════════════════════════════════════════════
print("═"*60); print("  LOADING FIELD MASS BALANCE"); print("═"*60)
_field_mb_index = {}

def _load_field_mb():
    if not os.path.isdir(FIELD_MB_DIR):
        print(f"  ⚠ Field MB dir not found: {FIELD_MB_DIR}"); return
    csv_files = [f for f in os.listdir(FIELD_MB_DIR) if f.lower().endswith(".csv")]
    print(f"  Found {len(csv_files)} CSV files")
    for fname in csv_files:
        fpath = os.path.join(FIELD_MB_DIR, fname)
        try:
            df = pd.read_csv(fpath, low_memory=False)
            df.columns = df.columns.str.strip().str.lower()
            rgi_col  = next((c for c in df.columns if "rgiid" in c or c=="rgi_id"), None)
            year_col = next((c for c in df.columns if c=="year"), None)
            if not rgi_col or not year_col: continue
            df = df.dropna(subset=[rgi_col,year_col])
            df[year_col] = pd.to_numeric(df[year_col], errors="coerce")
            df = df[df[year_col].notna()]
            df[year_col] = df[year_col].astype(int)
            for _, row in df.iterrows():
                rid = str(row[rgi_col]).strip()
                if not rid: continue
                entry = {
                    "year":   int(row[year_col]),
                    "annual": safe(row.get("annual_balance")),
                    "winter": safe(row.get("winter_balance")),
                    "summer": safe(row.get("summer_balance")),
                    "ela":    safe(row.get("ela")),
                    "aar":    safe(row.get("aar")),
                    "source": safe_str(row.get("source_file") or row.get("references") or fname),
                }
                _field_mb_index.setdefault(rid, []).append(entry)
        except Exception as e:
            print(f"    ⚠ {fname}: {e}")
    for rid in _field_mb_index:
        _field_mb_index[rid].sort(key=lambda x: x["year"])
    print(f"  ✓ {len(_field_mb_index)} glaciers with field MB data")

with timed("Field MB"): _load_field_mb()

# ═══════════════════════════════════════════════════════════════════════════
#  RGI SHAPEFILES
# ═══════════════════════════════════════════════════════════════════════════
print("═"*60); print("  LOADING RGI SHAPEFILES"); print("═"*60)
RGI_PATHS = {
    "RGI60-13": "13_rgi60_CentralAsia/13_rgi60_CentralAsia.shp",
    "RGI60-14": "14_rgi60_SouthAsiaWest/14_rgi60_SouthAsiaWest.shp",
    "RGI60-15": "15_rgi60_SouthAsiaEast/15_rgi60_SouthAsiaEast.shp",
}
gdf_list = []
with timed("Read shapefiles"):
    for rkey, rpath in RGI_PATHS.items():
        full = os.path.join(RGI_BASE, rpath)
        if os.path.isfile(full):
            g = gpd.read_file(full).to_crs("EPSG:4326")
            gdf_list.append(g)
            print(f"    {rkey}: {len(g):,} features")

glaciers_gdf = pd.concat(gdf_list, ignore_index=True) if gdf_list else gpd.GeoDataFrame()
glaciers_gdf = glaciers_gdf[glaciers_gdf["RGIId"].isin(all_glacier_ids)].copy()

if "CenLon" in glaciers_gdf.columns:
    glaciers_gdf["lon"] = glaciers_gdf["CenLon"].astype(float)
    glaciers_gdf["lat"] = glaciers_gdf["CenLat"].astype(float)
else:
    c = glaciers_gdf.geometry.centroid
    glaciers_gdf["lon"] = c.x; glaciers_gdf["lat"] = c.y

glaciers_gdf["glacier_name"] = glaciers_gdf["RGIId"].map(_name_map)
print(f"  {len(glaciers_gdf):,} glaciers matched")

# ═══════════════════════════════════════════════════════════════════════════
#  HMA REGIONS
# ═══════════════════════════════════════════════════════════════════════════
print("═"*60); print("  HMA REGIONS"); print("═"*60)
_hma_geojson = '{"type":"FeatureCollection","features":[]}'
_hma_center  = {"lat":35.0,"lon":82.0}
hma_gdf = None

def _cache_key():
    h = hashlib.md5()
    h.update(str(sorted(list(glaciers_gdf["RGIId"][:100]))).encode())
    h.update(str(len(glaciers_gdf)).encode())
    if os.path.isfile(HMA_SHP): h.update(str(os.path.getmtime(HMA_SHP)).encode())
    return h.hexdigest()[:12]

cache_file = os.path.join(CACHE_DIR, f"region_map_{_cache_key()}.pkl")

try:
    with timed("Load HMA shapefile"):
        hma_gdf = gpd.read_file(HMA_SHP).to_crs("EPSG:4326")

    KNOWN = ["central himalaya","east himalaya","east kun lun","east tien shan",
             "hengduan shan","hindu kush","hissar alay","inner tibet","karakoram",
             "pamir","qilian shan","southeast tibet","west himalaya","west kun lun","west tien shan"]
    hma_name_col = None
    for col in hma_gdf.columns:
        if col=="geometry": continue
        vals = hma_gdf[col].dropna().astype(str).str.lower().tolist()
        if sum(1 for v in vals if any(k in v for k in KNOWN))>=3:
            hma_name_col=col; break
    if not hma_name_col:
        for col in hma_gdf.columns:
            if col!="geometry" and hma_gdf[col].dtype==object:
                hma_name_col=col; break
    hma_gdf["hma_region"] = hma_gdf[hma_name_col].astype(str).str.strip() if hma_name_col else ("Region_"+hma_gdf.index.astype(str))

    if os.path.isfile(cache_file):
        print("    Loading spatial join from cache …")
        with open(cache_file,"rb") as f: region_map_dict = pickle.load(f)
        glaciers_gdf["region_key"] = glaciers_gdf["RGIId"].map(region_map_dict).fillna("Unknown")
    else:
        with timed("Spatial join"):
            pts = gpd.GeoDataFrame(
                glaciers_gdf[["RGIId","lon","lat"]],
                geometry=gpd.points_from_xy(glaciers_gdf["lon"],glaciers_gdf["lat"]),
                crs="EPSG:4326",
            )
            joined = gpd.sjoin(pts, hma_gdf[["hma_region","geometry"]], how="left", predicate="within")
            unmatched = joined[joined["hma_region"].isna()]["RGIId"].tolist()
            if unmatched:
                um = pts[pts["RGIId"].isin(unmatched)]
                near = gpd.sjoin_nearest(um, hma_gdf[["hma_region","geometry"]], how="left")
                for _, r in near.iterrows():
                    joined.loc[joined["RGIId"]==r["RGIId"],"hma_region"] = r["hma_region"]
            joined = joined.drop_duplicates(subset="RGIId",keep="first")
            region_map_dict = dict(zip(joined["RGIId"],joined["hma_region"]))
            glaciers_gdf["region_key"] = glaciers_gdf["RGIId"].map(region_map_dict).fillna("Unknown")
            with open(cache_file,"wb") as f: pickle.dump(region_map_dict,f)

    with timed("HMA GeoJSON"):
        hma_out = hma_gdf.copy()
        hma_out["geometry"] = hma_out["geometry"].simplify(0.005,preserve_topology=True)
        _hma_geojson = hma_out.to_json()

    hma_cent = hma_gdf.geometry.union_all().centroid
    _hma_center = {"lat":round(hma_cent.y,3),"lon":round(hma_cent.x,3)}

except Exception as e:
    traceback.print_exc()
    def _rkey(rgi_id):
        p = rgi_id.split("-")
        return "RGI60-"+p[1].split(".")[0] if len(p)>1 else "Unknown"
    glaciers_gdf["region_key"] = glaciers_gdf["RGIId"].apply(_rkey)

_glacier_region = dict(zip(glaciers_gdf["RGIId"],glaciers_gdf["region_key"]))

# ═══════════════════════════════════════════════════════════════════════════
#  BUILD CACHES
# ═══════════════════════════════════════════════════════════════════════════
print("═"*60); print("  BUILDING CACHES"); print("═"*60)

with timed("Info cache"):
    _info_cache = {}
    area_panel = panels.get("area_km2")
    vol_panel  = panels.get("volume_km3")
    mb_panel   = panels.get("mb_annual_mwe")
    for _, row in glaciers_gdf.iterrows():
        rid = row["RGIId"]
        yr_min,yr_max = None,None
        for df in panels.values():
            if rid in df.columns:
                valid = df[rid].dropna()
                if not valid.empty:
                    mn,mx = int(valid.index.min()),int(valid.index.max())
                    yr_min = mn if yr_min is None else min(yr_min,mn)
                    yr_max = mx if yr_max is None else max(yr_max,mx)
        lat_area = None
        if area_panel is not None and rid in area_panel.columns:
            v = area_panel[rid].dropna(); lat_area = safe(v.iloc[-1]) if not v.empty else None
        lat_vol = None
        if vol_panel is not None and rid in vol_panel.columns:
            v = vol_panel[rid].dropna(); lat_vol = safe(v.iloc[-1]) if not v.empty else None
        lat_mb = None
        if mb_panel is not None and rid in mb_panel.columns:
            v = mb_panel[rid].dropna(); lat_mb = safe(v.iloc[-1]) if not v.empty else None
        _info_cache[rid] = {
            "rgi_id":rid,
            "name":safe_str(row.get("glacier_name") or row.get("Name"),fallback=rid),
            "lat":round(float(row["lat"]),5),"lon":round(float(row["lon"]),5),
            "area_km2_shp":safe(row.get("Area")),
            "zmin":safe(row.get("Zmin")),"zmax":safe(row.get("Zmax")),
            "zmed":safe(row.get("Zmed")),"slope":safe(row.get("Slope")),
            "region":row["region_key"],
            "year_min":yr_min,"year_max":yr_max,
            "latest_area_km2":lat_area,"latest_volume_km3":lat_vol,"latest_mb_annual":lat_mb,
        }
    print(f"    {len(_info_cache):,} entries")

with timed("Centroids + search index"):
    _centroids = glaciers_gdf[["RGIId","lat","lon","region_key","Area","glacier_name"]].copy()
    _centroids["name_lower"] = _centroids["glacier_name"].fillna("").str.lower()
    _centroids["id_lower"]   = _centroids["RGIId"].str.lower()

_hma_summary = {
    "count":len(glaciers_gdf),
    "area_total":round(float(glaciers_gdf["Area"].sum()),1),
    "lat":_hma_center["lat"],"lon":_hma_center["lon"],
    "label":"High Mountain Asia",
}

_hma_regions_list = []
for rname,sub in glaciers_gdf.groupby("region_key"):
    bbox = [round(float(sub["lon"].min()),3),round(float(sub["lat"].min()),3),
            round(float(sub["lon"].max()),3),round(float(sub["lat"].max()),3)]
    _hma_regions_list.append({
        "name":rname,"count":len(sub),
        "area":round(float(sub["Area"].sum()),1),
        "lat":round(float(sub["lat"].mean()),4),
        "lon":round(float(sub["lon"].mean()),4),
        "bbox":bbox,
    })
_hma_regions_list.sort(key=lambda x: x["name"])

_region_glacier_ids = {}
for rname,sub in glaciers_gdf.groupby("region_key"):
    _region_glacier_ids[rname] = set(sub["RGIId"])

# ═══════════════════════════════════════════════════════════════════════════
#  STATS
# ═══════════════════════════════════════════════════════════════════════════
def _compute_stats(gid_set):
    mb_df=panels.get("mb_annual_mwe"); vol_df=panels.get("volume_km3"); ar_df=panels.get("area_km2")
    mb_cols  = [g for g in gid_set if mb_df  is not None and g in mb_df.columns]
    vol_cols = [g for g in gid_set if vol_df is not None and g in vol_df.columns]
    ar_cols  = [g for g in gid_set if ar_df  is not None and g in ar_df.columns]
    all_yrs = set()
    if mb_cols:  all_yrs|=set(mb_df.index)
    if vol_cols: all_yrs|=set(vol_df.index)
    if ar_cols:  all_yrs|=set(ar_df.index)
    if not all_yrs: return {"years":[],"mean_mb":[],"total_volume":[],"total_area":[],"n_glaciers":[]}
    years=sorted(all_yrs)
    mb_sub  = mb_df[mb_cols].reindex(years)   if mb_cols  else None
    vol_sub = vol_df[vol_cols].reindex(years)  if vol_cols else None
    ar_sub  = ar_df[ar_cols].reindex(years)    if ar_cols  else None
    mean_mb_arr = mb_sub.mean(axis=1).tolist()  if mb_sub  is not None else [None]*len(years)
    tot_vol_arr = vol_sub.sum(axis=1).tolist()   if vol_sub is not None else [None]*len(years)
    tot_ar_arr  = ar_sub.sum(axis=1).tolist()    if ar_sub  is not None else [None]*len(years)
    n_arr=[]
    for yr in years:
        ct=0
        if mb_sub  is not None and yr in mb_sub.index:  ct=max(ct,int(mb_sub.loc[yr].notna().sum()))
        if vol_sub is not None and yr in vol_sub.index: ct=max(ct,int(vol_sub.loc[yr].notna().sum()))
        if ar_sub  is not None and yr in ar_sub.index:  ct=max(ct,int(ar_sub.loc[yr].notna().sum()))
        n_arr.append(ct)
    return {"years":[int(y) for y in years],"mean_mb":[safe(v) for v in mean_mb_arr],
            "total_volume":[safe(v) for v in tot_vol_arr],"total_area":[safe(v) for v in tot_ar_arr],"n_glaciers":n_arr}

print("═"*60); print("  COMPUTING STATS"); print("═"*60)
with timed("HMA stats"):   _hma_stats = _compute_stats(all_glacier_ids)
with timed("Region stats"):
    _region_stats = {rname:_compute_stats(gids) for rname,gids in _region_glacier_ids.items()}
print("═"*60); print("  ALL READY"); print("═"*60)

# ═══════════════════════════════════════════════════════════════════════════
#  API ROUTES (identical to original)
# ═══════════════════════════════════════════════════════════════════════════
@app.route("/api/hma_outline")
def api_hma_outline(): return Response(_hma_geojson,mimetype="application/json")

@app.route("/api/hma_regions")
def api_hma_regions(): return jsonify(_hma_regions_list)

@app.route("/api/hma_summary")
def api_hma_summary(): return jsonify(_hma_summary)

@app.route("/api/clusters")
def api_clusters():
    cells=int(request.args.get("cells",14)); bbox_s=request.args.get("bbox"); region=request.args.get("region","")
    df=_centroids
    if region and region!="all": df=df[df["region_key"]==region]
    if bbox_s:
        try:
            minlon,minlat,maxlon,maxlat=map(float,bbox_s.split(","))
            df=df[(df["lon"]>=minlon)&(df["lon"]<=maxlon)&(df["lat"]>=minlat)&(df["lat"]<=maxlat)]
        except: pass
    if df.empty: return jsonify([])
    lon_min,lon_max=df["lon"].min(),df["lon"].max()
    lat_min,lat_max=df["lat"].min(),df["lat"].max()
    dlon=(lon_max-lon_min)/cells or 1; dlat=(lat_max-lat_min)/cells or 1
    df2=df.copy()
    df2["cx"]=((df2["lon"]-lon_min)/dlon).astype(int).clip(0,cells-1)
    df2["cy"]=((df2["lat"]-lat_min)/dlat).astype(int).clip(0,cells-1)
    return jsonify([{"count":len(s),"lat":round(float(s["lat"].mean()),4),
                     "lon":round(float(s["lon"].mean()),4),"region":s["region_key"].mode()[0]}
                    for (_,_),s in df2.groupby(["cx","cy"])])

@app.route("/api/polygons")
def api_polygons():
    bbox_s=request.args.get("bbox"); region=request.args.get("region","")
    if not bbox_s: return jsonify({"error":"bbox required"}),400
    try: minlon,minlat,maxlon,maxlat=map(float,bbox_s.split(","))
    except: return jsonify({"error":"bad bbox"}),400
    gdf=glaciers_gdf
    if region and region!="all": gdf=gdf[gdf["region_key"]==region]
    vis=gdf[(gdf["lon"]>=minlon)&(gdf["lon"]<=maxlon)&(gdf["lat"]>=minlat)&(gdf["lat"]<=maxlat)].copy()
    if vis.empty: return Response('{"type":"FeatureCollection","features":[]}',mimetype="application/json")
    if len(vis)>600: vis=vis.nlargest(600,"Area")
    span=max(maxlon-minlon,maxlat-minlat)
    vis["geometry"]=vis["geometry"].simplify(max(0.0001,span*0.002),preserve_topology=True)
    keep=[c for c in ["RGIId","glacier_name","Area","region_key","geometry"] if c in vis.columns]
    out=vis[keep].rename(columns={"RGIId":"id","glacier_name":"name","Area":"area","region_key":"region"})
    return Response(out.to_json(),mimetype="application/json")

@app.route("/api/search")
def api_search():
    q=request.args.get("q","").strip().lower(); by=request.args.get("by","both")
    limit=min(int(request.args.get("limit",30)),100); region=request.args.get("region","")
    if len(q)<2: return jsonify([])
    df=_centroids
    if region and region!="all": df=df[df["region_key"]==region]
    if by=="name": mask=df["name_lower"].str.contains(q,na=False)
    elif by=="id": mask=df["id_lower"].str.contains(q,na=False)
    else: mask=df["name_lower"].str.contains(q,na=False)|df["id_lower"].str.contains(q,na=False)
    hits=df[mask].head(limit)
    return jsonify([{"id":r["RGIId"],"name":r["glacier_name"] if pd.notna(r["glacier_name"]) else r["RGIId"],
        "lat":round(float(r["lat"]),5),"lon":round(float(r["lon"]),5),
        "area":round(float(r["Area"]),3) if pd.notna(r.get("Area")) else None,
        "region":r["region_key"]} for _,r in hits.iterrows()])

@app.route("/api/glacier/<rgi_id>/info")
def api_info(rgi_id):
    info=_info_cache.get(rgi_id)
    if not info: return jsonify({"error":"Not found"}),404
    return jsonify(info)

@app.route("/api/glacier/<rgi_id>/timeseries")
def api_timeseries(rgi_id):
    t0=time.time()
    all_years=set()
    for field,df in panels.items():
        if rgi_id in df.columns:
            valid=df[rgi_id].dropna()
            if not valid.empty: all_years|=set(valid.index)
    if not all_years: return jsonify({"error":"Not found","rgi_id":rgi_id}),404
    years=sorted(all_years)
    def get_col(field):
        if field not in panels or rgi_id not in panels[field].columns: return [None]*len(years)
        col=panels[field][rgi_id]
        return [safe(col.get(yr)) for yr in years]
    result={
        "rgi_id":rgi_id,"years":[int(y) for y in years],
        "mb_annual":get_col("mb_annual_mwe"),"mb_cumulative":get_col("mb_cumulative_mwe"),
        "volume_km3":get_col("volume_km3"),"area_km2":get_col("area_km2"),
    }
    return jsonify(result)

@app.route("/api/glacier/<rgi_id>/field_mb")
def api_field_mb(rgi_id):
    records=_field_mb_index.get(rgi_id)
    if not records:
        lower=rgi_id.lower()
        for k,v in _field_mb_index.items():
            if k.lower()==lower: records=v; break
    if not records: return jsonify({"rgi_id":rgi_id,"has_data":False,"records":[]})
    return jsonify({"rgi_id":rgi_id,"has_data":True,"n":len(records),"records":records})

@app.route("/api/stats/hma")
def api_stats_hma(): return jsonify(_hma_stats)

@app.route("/api/stats/region/<path:region_name>")
def api_stats_region(region_name):
    decoded=unquote(region_name)
    s=_region_stats.get(decoded)
    if s: return jsonify(s)
    lower=decoded.lower()
    for rn,st in _region_stats.items():
        if rn.lower()==lower: return jsonify(st)
    return jsonify({"error":f"Region '{decoded}' not found"}),404

@app.route("/")
def index(): return send_from_directory(STATIC_DIR,"index.html")

@app.route("/<path:path>")
def static_files(path): return send_from_directory(STATIC_DIR,path)

if __name__=="__main__":
    print(f"\nPortal up → http://localhost:5000\n")
    app.run(debug=False,port=5000,threaded=True)
