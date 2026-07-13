"""
Interactive OpenStreetMap of pseudo-absence points
Three layers: Random PA | Heuristic PA | SA PA  + Fire points
Uses Folium + OpenStreetMap tiles
"""
import numpy as np
import pandas as pd
import folium
from folium.plugins import MarkerCluster, MiniMap, LocateControl

# ── Load points ────────────────────────────────────────────
df       = pd.read_csv('../../data/Fire_points_dataset_final_csv.csv').dropna()
fire_pts = df[['LONGITUDE','LATITUDE']].values
heur_pts = np.load('../../outputs/heur_pts.npy')
rand_pts = np.load('../../outputs/rand_pts.npy')
sa_pts   = np.load('../../outputs/sa_pts.npy')

# Centre of study area
ctr_lat = fire_pts[:,1].mean()
ctr_lon = fire_pts[:,0].mean()

# ── Create map ─────────────────────────────────────────────
m = folium.Map(
    location=[ctr_lat, ctr_lon],
    zoom_start=7,
    tiles='OpenStreetMap',
    control_scale=True,
    prefer_canvas=True,
)

# ── Helper: add a CircleMarker layer ───────────────────────
def add_layer(pts, name, color, fill_color, radius=3, opacity=0.7):
    fg = folium.FeatureGroup(name=name, show=True)
    for lon, lat in pts:
        folium.CircleMarker(
            location=[lat, lon],
            radius=radius,
            color=color,
            fill=True,
            fill_color=fill_color,
            fill_opacity=opacity,
            weight=0.5,
            popup=folium.Popup(f'{name}<br>Lat: {lat:.4f}<br>Lon: {lon:.4f}', max_width=180),
        ).add_to(fg)
    fg.add_to(m)
    return fg

# Add layers (order matters for z-index: bottom → top)
add_layer(rand_pts, f'Random PA (n={len(rand_pts):,})',       '#2ca02c', '#2ca02c', radius=3,   opacity=0.55)
add_layer(sa_pts,   f'SA PA (n={len(sa_pts):,})',             '#ff7f0e', '#ff7f0e', radius=3,   opacity=0.55)
add_layer(heur_pts, f'Heuristic PA (n={len(heur_pts):,})',    '#1f77b4', '#1f77b4', radius=3,   opacity=0.55)
add_layer(fire_pts, f'Fire Points (n={len(fire_pts):,})',     '#d62728', '#d62728', radius=3.5, opacity=0.75)

# ── Bounding box of study area ─────────────────────────────
lon_min = fire_pts[:,0].min() - 0.3
lon_max = fire_pts[:,0].max() + 0.3
lat_min = fire_pts[:,1].min() - 0.3
lat_max = fire_pts[:,1].max() + 0.3

folium.Rectangle(
    bounds=[[lat_min, lon_min], [lat_max, lon_max]],
    color='#555555',
    weight=1.5,
    fill=False,
    dash_array='6',
    tooltip='Study area extent',
).add_to(m)

# ── Layer control + minimap ────────────────────────────────
folium.LayerControl(position='topright', collapsed=False).add_to(m)
MiniMap(toggle_display=True, position='bottomleft').add_to(m)

# ── Legend ─────────────────────────────────────────────────
legend_html = """
<div style="
    position: fixed;
    bottom: 40px; right: 15px;
    z-index: 9999;
    background: rgba(255,255,255,0.95);
    border: 2px solid #aaa;
    border-radius: 8px;
    padding: 12px 16px;
    font-family: Arial, sans-serif;
    font-size: 13px;
    box-shadow: 2px 2px 6px rgba(0,0,0,0.25);
    min-width: 210px;
">
<b style="font-size:14px;">Pseudo-Absence Methods</b><br><br>
<span style="color:#d62728;">&#9679;</span>&nbsp;<b>Fire Points</b>&nbsp;(n={nf:,})<br>
<span style="color:#1f77b4;">&#9679;</span>&nbsp;<b>Heuristic PA</b> (BP_V6+CS_V5)<br>
<span style="color:#ff7f0e;">&#9679;</span>&nbsp;<b>SA PA</b> (Simulated Annealing)<br>
<span style="color:#2ca02c;">&#9679;</span>&nbsp;<b>Random PA</b><br>
<br>
<hr style="margin:6px 0;">
<span style="font-size:11px;color:#555;">
Use layer control (top-right)<br>
to toggle each method on/off.
</span>
</div>
""".format(nf=len(fire_pts))

m.get_root().html.add_child(folium.Element(legend_html))

# ── Title bar ──────────────────────────────────────────────
title_html = """
<div style="
    position: fixed;
    top: 10px; left: 50%; transform: translateX(-50%);
    z-index: 9999;
    background: rgba(255,255,255,0.95);
    border: 2px solid #888;
    border-radius: 8px;
    padding: 8px 20px;
    font-family: Arial, sans-serif;
    font-size: 15px;
    font-weight: bold;
    color: #222;
    box-shadow: 2px 2px 6px rgba(0,0,0,0.2);
    pointer-events: none;
">
Alberta Wildfire Dataset — Pseudo-Absence Methods on OpenStreetMap
</div>
"""
m.get_root().html.add_child(folium.Element(title_html))

# ── Save ───────────────────────────────────────────────────
out = '../../outputs/pa_openstreetmap.html'
m.save(out)
print(f"Map saved → {out}")
print(f"Points: fire={len(fire_pts):,}  heuristic={len(heur_pts):,}  SA={len(sa_pts):,}  random={len(rand_pts):,}")
