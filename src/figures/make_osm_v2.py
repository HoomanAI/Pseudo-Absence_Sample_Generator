import numpy as np, pandas as pd, folium
from folium.plugins import MiniMap
OUT='../../outputs/'
fire=pd.read_csv('../../data/Fire_points_dataset_final_csv.csv').dropna()[['LONGITUDE','LATITUDE']].values
heur=np.load(OUT+'heur_pts.npy'); rand=np.load(OUT+'rand_pts.npy')
sa=np.load(OUT+'sa_pts.npy');     gan=np.load(OUT+'gan_pts.npy')

clat=np.mean(np.vstack([fire,heur,rand,sa,gan])[:,1])
clon=np.mean(np.vstack([fire,heur,rand,sa,gan])[:,0])
m=folium.Map(location=[clat,clon],zoom_start=6,tiles='OpenStreetMap')
MiniMap(toggle_display=True).add_to(m)

layers=[
    ('Fire Points',       fire,  '#d62728', 3, 0.8),
    ('Heuristic BP_V6',   heur,  '#1f77b4', 3, 0.5),
    ('Simulated Annealing', sa,  '#ff7f0e', 3, 0.5),
    ('GAN (CSGN)',         gan,  '#9467bd', 3, 0.5),
    ('Random',            rand,  '#2ca02c', 3, 0.5),
]
for lbl,pts,col,r,a in layers:
    fg=folium.FeatureGroup(name=lbl,show=(lbl in ('Fire Points','GAN (CSGN)')))
    sub=pts[np.random.choice(len(pts),min(1000,len(pts)),replace=False)]
    for pt in sub:
        folium.CircleMarker([pt[1],pt[0]],radius=r,color=col,fill=True,fill_color=col,
                            fill_opacity=a,weight=0).add_to(fg)
    fg.add_to(m)

legend_html="""
<div style='position:fixed;bottom:30px;left:30px;z-index:1000;background:white;
padding:12px 16px;border-radius:8px;border:1px solid #ccc;font-size:13px;box-shadow:3px 3px 6px rgba(0,0,0,0.2)'>
<b>Pseudo-Absence Methods</b><br>
<span style='color:#d62728'>&#9679;</span> Fire Points (n=3370)<br>
<span style='color:#1f77b4'>&#9679;</span> Heuristic BP_V6 (n=5500)<br>
<span style='color:#ff7f0e'>&#9679;</span> Simulated Annealing (n=5500)<br>
<span style='color:#9467bd'>&#9679;</span> GAN / CSGN (n=5500)<br>
<span style='color:#2ca02c'>&#9679;</span> Random (n=5500)
</div>"""
m.get_root().html.add_child(folium.Element(legend_html))
folium.LayerControl(collapsed=False).add_to(m)
m.save(OUT+'pa_openstreetmap.html')
print('OSM map saved.')
