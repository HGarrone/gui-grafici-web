"""
Script da eseguire UNA VOLTA in locale per creare i file CAP pre-filtrati.
I file generati sono piccoli abbastanza per GitHub (<50 MB ciascuno).
NON caricare cap_ita.geojson su GitHub (è 206 MB).

Esegui con:  python prep_geo.py
"""
import os
import sys

try:
    import geopandas as gpd
except ImportError:
    sys.exit("Errore: geopandas non installato. Esegui: pip install geopandas")

_XGUI   = os.path.join(os.path.dirname(__file__), "xGUI")
cap_src = os.path.join(_XGUI, "cap_ita.geojson")

if not os.path.exists(cap_src):
    sys.exit(f"Errore: file non trovato:\n  {cap_src}")

print("Caricamento cap_ita.geojson (alcuni secondi)…")
gdf = gpd.read_file(cap_src)

name_col = next(
    (c for c in ["cap", "CAP", "codice_postale", "zip", "postalcode", "name", "NAME"]
     if c in gdf.columns),
    gdf.columns[0],
)
cap_str = gdf[name_col].astype(str)

_TORINO_CITY = (
    {"10100"}
    | {str(i) for i in range(10121, 10132)}
    | {str(i) for i in range(10132, 10138)}
    | {str(i) for i in range(10138, 10142)}
    | {str(i) for i in range(10142, 10149)}
    | {str(i) for i in range(10151, 10157)}
)

filtri = {
    "cap_piemvda.geojson":    cap_str.str[:2].isin({"10","11","12","13","14","15","28"}),
    "cap_torino.geojson":     cap_str.str[:2].isin({"10"}),
    "cap_torino_city.geojson": cap_str.isin(_TORINO_CITY),
}

for filename, mask in filtri.items():
    out = os.path.join(_XGUI, filename)
    sub = gdf[mask].copy()
    sub.to_file(out, driver="GeoJSON")
    mb = os.path.getsize(out) / 1_048_576
    print(f"  Salvato  {filename}  ({mb:.1f} MB,  {len(sub)} aree)")

print("\nFatto! Aggiungi a git i file cap_piemvda.geojson, cap_torino.geojson, cap_torino_city.geojson")
print("e tieni cap_ita.geojson SOLO in locale (è già in .gitignore).")
