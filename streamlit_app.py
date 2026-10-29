"""
GUI Grafici — versione web (Streamlit)
Deployment: streamlit run streamlit_app.py
             oppure Streamlit Cloud (share.streamlit.io)
"""

import colorsys
import io
import json
import os
import pathlib
import tempfile
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as mcm
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

import chart_animation as CA

# ── Percorsi ──────────────────────────────────────────────────────────────────
_DIR  = pathlib.Path(__file__).parent
_XGUI = _DIR / "xGUI"

# ── Geopandas (opzionale: solo per mappe) ─────────────────────────────────────
try:
    import geopandas as gpd
    GEO_OK = True
except ImportError:
    GEO_OK = False

# ── Configurazione pagina ──────────────────────────────────────────────────────
st.set_page_config(layout="wide", page_title="GUI Grafici", page_icon="📊")
st.title("📊 GUI Grafici")

# ── Session state ──────────────────────────────────────────────────────────────
_defaults = {"df": None, "col_roles": {}, "fig": None, "last_chart_pkg": None}
for k, v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── Costanti ───────────────────────────────────────────────────────────────────
CHART_TYPES = [
    ("Barre",              "bar"),
    ("Barre impilate",     "bar_stacked"),
    ("Linee",              "line"),
    ("Dispersione (XY)",   "scatter"),
    ("Istogramma",         "histogram"),
    ("Torta",              "pie"),
    ("Heatmap",            "heatmap"),
    ("Boxplot",            "boxplot"),
    ("Radar",              "radar"),
    ("Imbuto (Funnel)",    "funnel"),
    ("Podio",              "podio"),
    ("Violino",            "violin"),
    ("Barre + Linee",      "combo"),
    ("Treemap",            "treemap"),
    ("Mappa coropletica",  "choropleth"),
    ("Mappa bolla",        "bubble_map"),
    ("Mappa bande",        "band_map"),
    ("Mappa pin",          "pin_map"),
]

MAP_KEYS    = {"choropleth", "bubble_map", "band_map", "pin_map"}
# Grafici che usano assi personalizzati (axis off / polare) — _apply_axis_settings saltata
_NO_AXIS_KEYS = {"funnel", "podio", "treemap"}
_FREQ_LABEL   = "— Frequenza —"

_SDG_COLORS = [
    "#E5243B","#DDA63A","#4C9F38","#C5192D","#FF3A21",
    "#26BDE2","#FCC30B","#A21942","#FD6925","#DD1367",
    "#FD9D24","#BF8B2E","#3F7E44","#0A97D9","#56C02B",
    "#00689D","#19486A",
]

# Palette predefinite (abbinamenti generici)
_BRAND_PALETTES = {
    "Oceano":            ["#03045E", "#0077B6", "#00B4D8", "#90E0EF", "#023E8A", "#48CAE4", "#0096C7", "#ADE8F4"],
    "Tramonto":          ["#F94144", "#F3722C", "#F8961E", "#F9C74F", "#90BE6D", "#43AA8B", "#577590", "#277DA1"],
    "Terra":             ["#6B4226", "#A47148", "#D4A373", "#CCD5AE", "#606C38", "#283618", "#BC6C25", "#DDA15E"],
    "Notte e Corallo":   ["#22223B", "#F25C54", "#4A4E69", "#F7B267", "#9A8C98", "#F4845F", "#C9ADA7", "#F79D65"],
    "Foresta":           ["#081C15", "#1B4332", "#2D6A4F", "#40916C", "#52B788", "#74C69D", "#95D5B2", "#B7E4C7"],
    "Contrasto acceso":  ["#264653", "#2A9D8F", "#E9C46A", "#F4A261", "#E76F51", "#8AB17D", "#BB3E03", "#005F73"],
    "Grigi e Accento":   ["#2B2D42", "#8D99AE", "#EF233C", "#5C677D", "#D90429", "#979DAC", "#33415C", "#EDF2F4"],
    "Pastello":          ["#CDB4DB", "#FFC8DD", "#FFAFCC", "#BDE0FE", "#A2D2FF", "#B5E48C", "#FFD6A5", "#CAFFBF"],
    "SDG":               _SDG_COLORS,
}

PALETTES = {
    # ── Predefinite ───────────────────────────────────────────────────────────
    "Oceano":            "Oceano",
    "Tramonto":          "Tramonto",
    "Terra":             "Terra",
    "Notte e Corallo":   "Notte e Corallo",
    "Foresta":           "Foresta",
    "Contrasto acceso":  "Contrasto acceso",
    "Grigi e Accento":   "Grigi e Accento",
    "Pastello":          "Pastello",
    "SDG":               "SDG",
    # ── Qualitative matplotlib ───────────────────────────────────────────────
    "Tab10":     "tab10",
    "Tab20":     "tab20",
    "Set1":      "Set1",
    "Set2":      "Set2",
    "Set3":      "Set3",
    "Paired":    "Paired",
    "Pastel":    "Pastel1",
    "Scuro":     "Dark2",
    # ── Sequenziali / divergenti ─────────────────────────────────────────────
    "Blues":     "Blues",
    "Greens":    "Greens",
    "Reds":      "Reds",
    "Oranges":   "Oranges",
    "Purples":   "Purples",
    "YlOrRd":    "YlOrRd",
    "RdYlGn":    "RdYlGn",
    "Viridis":   "viridis",
    "Plasma":    "plasma",
    "Coolwarm":  "coolwarm",
    "Blu-Viola": "Blu-Viola",
}
_BRAND_PALETTES["Blu-Viola"] = ["#0B3D91", "#3A1C71", "#6A0DAD"]

MAP_SCOPES = [
    ("Piemonte — Province",               "prov_piemonte"),
    ("Piemonte + Val d'Aosta — Province", "prov_piemvda"),
    ("Italia — Province",                 "prov_italy"),
    ("Italia — Regioni",                  "reg_italy"),
    ("Europa — Paesi",                    "europe"),
    ("Piemonte — Comuni",                 "comuni_piemonte"),
    ("Val d'Aosta — Comuni",              "comuni_valdaosta"),
    # CAP: usa file pre-filtrati generati da prep_geo.py (non cap_ita.geojson che è 206 MB)
    ("Piemonte + Val d'Aosta — CAP",       "cap_piemvda"),
    ("Città Metropolitana di Torino — CAP", "cap_torino"),
    ("Torino — CAP",                        "cap_torino_city"),
    ("Torino — Circoscrizioni",             "circ_torino"),
]
MAP_SCOPE_KEYS = {lbl: key for lbl, key in MAP_SCOPES}

_PIEMONTE_PROV = {
    "Torino", "Cuneo", "Alessandria", "Asti", "Novara",
    "Vercelli", "Biella", "Verbano-Cusio-Ossola",
}
_PIEMVDA_PROV = _PIEMONTE_PROV | {"Valle d'Aosta/Vallée d'Aoste", "Aosta"}
_CAP_PREFIXES_PIEMVDA = {"10", "11", "12", "13", "14", "15", "28"}
_CAP_PREFIXES_TORINO  = {"10"}
_CAP_TORINO_CITY = (
    {"10100"}
    | {str(i) for i in range(10121, 10132)}
    | {str(i) for i in range(10132, 10138)}
    | {str(i) for i in range(10138, 10142)}
    | {str(i) for i in range(10142, 10149)}
    | {str(i) for i in range(10151, 10157)}
)

# ── Caricamento geometrie (cached) ─────────────────────────────────────────────
@st.cache_resource(show_spinner="Caricamento geometrie mappa…")
def _load_geo(scope_key: str):
    """Carica il GeoDataFrame per l'ambito scelto. Ritorna None se non disponibile."""
    if not GEO_OK:
        return None

    _PROV_URL  = ("https://raw.githubusercontent.com/openpolis/geojson-italy"
                  "/master/geojson/limits_IT_provinces.geojson")
    _REG_URL   = ("https://raw.githubusercontent.com/openpolis/geojson-italy"
                  "/master/geojson/limits_IT_regions.geojson")
    _PROV_LOC  = _XGUI / "province_italiane.geojson"
    _REG_LOC   = _XGUI / "regioni_italiane.geojson"
    _PIE_COM   = _XGUI / "ondata_confini_amministrativi_api_v2_it_20250101_piemonte_comuni.geo.json"
    _VDA_COM   = _XGUI / "ondata_confini_amministrativi_api_v2_it_20250101_valdaosta_comuni.geo.json"
    _CAP_FILE  = _XGUI / "cap_ita.geojson"
    _EUR_FILE  = _XGUI / "europe.geojson"

    def _name_col(gdf, candidates):
        return next((c for c in candidates if c in gdf.columns), gdf.columns[0])

    try:
        if scope_key in ("prov_piemonte", "prov_piemvda", "prov_italy"):
            src = str(_PROV_LOC) if _PROV_LOC.exists() else _PROV_URL
            gdf = gpd.read_file(src)
            nc  = _name_col(gdf, ["prov_name", "name", "NAME"])
            gdf = gdf.rename(columns={nc: "GEO_NAME"})
            if scope_key == "prov_piemonte":
                gdf = gdf[gdf["GEO_NAME"].isin(_PIEMONTE_PROV)].copy()
            elif scope_key == "prov_piemvda":
                gdf = gdf[gdf["GEO_NAME"].isin(_PIEMVDA_PROV)].copy()
            return gdf.to_crs(epsg=32632)

        elif scope_key == "reg_italy":
            src = str(_REG_LOC) if _REG_LOC.exists() else _REG_URL
            gdf = gpd.read_file(src)
            nc  = _name_col(gdf, ["reg_name", "name", "NAME"])
            gdf = gdf.rename(columns={nc: "GEO_NAME"})
            return gdf.to_crs(epsg=32632)

        elif scope_key == "europe":
            if not _EUR_FILE.exists():
                st.error("europe.geojson non trovato nella cartella xGUI/")
                return None
            gdf = gpd.read_file(str(_EUR_FILE))
            nc  = _name_col(gdf, ["name", "NAME", "SOVEREIGNT", "sovereignt",
                                   "NAME_LONG", "name_long", "admin", "ADMIN",
                                   "COUNTRY", "country"])
            gdf = gdf.rename(columns={nc: "GEO_NAME"})
            gdf["GEO_NAME"] = gdf["GEO_NAME"].astype(str).str.strip()
            if gdf.crs is None:
                gdf = gdf.set_crs(epsg=4326)
            return gdf.to_crs(epsg=3035)

        elif scope_key in ("comuni_piemonte", "comuni_valdaosta"):
            files = []
            if scope_key in ("comuni_piemonte",) and _PIE_COM.exists():
                files.append(str(_PIE_COM))
            if scope_key in ("comuni_valdaosta",) and _VDA_COM.exists():
                files.append(str(_VDA_COM))
            if not files:
                st.error("File comuni non trovato in xGUI/"); return None
            gdfs = [gpd.read_file(f) for f in files]
            gdf  = pd.concat(gdfs, ignore_index=True) if len(gdfs) > 1 else gdfs[0]
            gdf  = gpd.GeoDataFrame(gdf, geometry="geometry")
            nc   = _name_col(gdf, ["name", "comune", "COMUNE",
                                    "denominazione_ita", "DENOMINAZI", "NAME", "nome"])
            gdf  = gdf.rename(columns={nc: "GEO_NAME"})
            if gdf.crs is None:
                gdf = gdf.set_crs(epsg=4326)
            return gdf.to_crs(epsg=32632)

        elif scope_key in ("cap_piemvda", "cap_torino", "cap_torino_city"):
            # Usa file pre-filtrati generati da prep_geo.py
            _cap_files = {
                "cap_piemvda":     _XGUI / "cap_piemvda.geojson",
                "cap_torino":      _XGUI / "cap_torino.geojson",
                "cap_torino_city": _XGUI / "cap_torino_city.geojson",
            }
            _f = _cap_files[scope_key]
            if not _f.exists():
                st.error(
                    f"{_f.name} non trovato in xGUI/. "
                    "Esegui prima `python prep_geo.py` per generare i file CAP filtrati."
                )
                return None
            gdf = gpd.read_file(str(_f))
            nc  = _name_col(gdf, ["cap", "CAP", "GEO_NAME", "codice_postale",
                                   "zip", "postalcode", "name", "NAME"])
            gdf = gdf.rename(columns={nc: "GEO_NAME"})
            if gdf.crs is None:
                gdf = gdf.set_crs(epsg=4326)
            return gdf.to_crs(epsg=32632)

        elif scope_key == "circ_torino":
            _CIRC_FILE = _XGUI / "circoscrizioni.csv"
            if not _CIRC_FILE.exists():
                st.error("circoscrizioni.csv non trovato in xGUI/"); return None
            try:
                import csv as _csv
                from shapely import wkt as _swkt
                rows = []
                with open(str(_CIRC_FILE), encoding="utf-8", newline="") as f:
                    for r in _csv.DictReader(f, delimiter=";"):
                        rows.append(r)
                df_c  = pd.DataFrame(rows)
                geoms = []
                for w in df_c["WKT_GEOM"]:
                    try:    geoms.append(_swkt.loads(w))
                    except: geoms.append(None)
                gdf = gpd.GeoDataFrame(df_c, geometry=geoms, crs="EPSG:3003")
                gdf["DENOM"] = gdf["DENOM"].astype(str).str.strip()
                gdf = gdf.rename(columns={"DENOM": "GEO_NAME"})
                return gdf.to_crs(epsg=32632)
            except Exception as e:
                st.error(f"Errore circoscrizioni: {e}"); return None

    except Exception as e:
        st.error(f"Errore caricamento geometrie: {e}")
    return None


# ── Helper generici ────────────────────────────────────────────────────────────
def _is_num(series: pd.Series) -> bool:
    return pd.to_numeric(series, errors="coerce").notna().mean() > 0.5


def _smart_labels(ax_, labels, override_rot=None):
    """Etichette X con rotazione/wrapping automatici."""
    strs = [str(l) for l in labels]
    n = len(strs)
    if n == 0:
        return
    if override_rot is not None and override_rot >= 0:
        ha = "right" if override_rot > 0 else "center"
        ax_.set_xticklabels(strs, rotation=override_rot, ha=ha, fontsize=8, rotation_mode="anchor")
        return
    max_len = max(len(s) for s in strs)
    if n > 25:
        step = max(1, round(n / 15))
        sparse = [s if i % step == 0 else "" for i, s in enumerate(strs)]
        ax_.set_xticklabels(sparse, rotation=70, ha="right", fontsize=6, rotation_mode="anchor")
    elif n > 10 or max_len > 8:
        rot = 70 if max_len > 12 else 45
        ax_.set_xticklabels(strs, rotation=rot, ha="right", fontsize=7, rotation_mode="anchor")
    elif max_len > 10:
        wrapped = ["\n".join(textwrap.wrap(s, 12)) for s in strs]
        ax_.set_xticklabels(wrapped, rotation=0, ha="center", fontsize=8)
        ax_.tick_params(axis="x", pad=4)
    else:
        ax_.set_xticklabels(strs, rotation=30, ha="right", fontsize=8, rotation_mode="anchor")


def _xkey(v):
    """Chiave di ordinamento: numerica se possibile, altrimenti stringa."""
    try:
        return (0, float(v))
    except (ValueError, TypeError):
        return (1, str(v))


def _sort_df_by_x(df_, x_col):
    """Ordina df_ per x_col: numericamente se tutti i valori sono numerici, altrimenti come stringa."""
    nums = pd.to_numeric(df_[x_col], errors="coerce")
    if nums.notna().all():
        return df_.sort_values(x_col, key=lambda s: pd.to_numeric(s, errors="coerce"))
    return df_.sort_values(x_col, key=lambda s: s.astype(str))


def _sort_cats_desc(df_, group_col, cats, value_col, agg="sum"):
    """Ordina una lista di categorie per valore aggregato discendente (sum/mean/
    median di value_col raggruppato per group_col), oppure per conteggio righe
    se value_col è vuoto/_FREQ_LABEL. Usata dal toggle 'Ordina per valore Y'."""
    if not value_col or value_col == _FREQ_LABEL:
        scores = df_[group_col].value_counts()
    else:
        nums = pd.to_numeric(df_[value_col], errors="coerce")
        scores = df_.assign(_v=nums).groupby(group_col)["_v"].agg(agg)
    return sorted(cats, key=lambda c: scores.get(c, 0) if pd.notna(scores.get(c, 0)) else 0,
                  reverse=True)


def _agg(df_, x_col, y_col, grp_col=None):
    g_cols = [c for c in [x_col, grp_col] if c]
    if y_col and g_cols:
        return df_.groupby(g_cols, sort=False)[y_col].sum().reset_index()
    if x_col:
        return df_[x_col].value_counts().reset_index().rename(
            columns={x_col: x_col, "count": "_n"})
    return df_


def _agg_by_x(df_, x_cols, y_col, sort_y_desc=False):
    """Aggrega df_ per x_cols: frequenza se y_col == _FREQ_LABEL, altrimenti somma numerica.
    x_cols può essere str (singolo) o list (heatmap: [col_cols, row_col]).
    Restituisce df ordinato per il primo x_col (o per valore decrescente se sort_y_desc),
    con colonne [*x_cols, value_col]."""
    if isinstance(x_cols, str):
        x_cols = [x_cols]
    if y_col == _FREQ_LABEL or not y_col:
        out = df_.groupby(x_cols, sort=False).size().reset_index(name="Frequenza")
        val_col = "Frequenza"
    else:
        out = df_.groupby(x_cols, sort=False)[y_col].sum().reset_index()
        val_col = y_col
    if sort_y_desc:
        out = out.sort_values(val_col, ascending=False)
    else:
        # ordina sempre per il primo asse X (numerico se possibile)
        out = _sort_df_by_x(out, x_cols[0])
    return out, val_col


def _lighten_for_dark_bg(c, amount: float = 0.32):
    """Schiarisce/vivacizza un colore (hex string O tupla RGBA 0-1 di
    matplotlib) per restare leggibile su sfondo scuro — stesso principio
    della 'palette_scura' distinta dello script R (analisi_rete_generale),
    ma applicato PROGRAMMATICAMENTE a qualunque palette invece di doverne
    disegnare una a mano per ciascuna. Alza la luminosità in spazio HLS
    mantenendo tonalità/saturazione, così i colori restano riconoscibili
    come "la stessa palette", solo adattata allo sfondo."""
    is_hex = isinstance(c, str)
    if is_hex:
        s = c.lstrip("#")
        if len(s) not in (6, 8):
            return c
        r, g, b = int(s[0:2], 16) / 255, int(s[2:4], 16) / 255, int(s[4:6], 16) / 255
        alpha_suffix = s[6:8] if len(s) == 8 else ""
    else:
        r, g, b = c[0], c[1], c[2]

    h, l, s_ = colorsys.rgb_to_hls(r, g, b)
    l = min(0.82, l + amount * (1 - l))
    s_ = min(1.0, s_ * 1.08)
    r2, g2, b2 = colorsys.hls_to_rgb(h, l, s_)

    if is_hex:
        return f"#{int(round(r2*255)):02X}{int(round(g2*255)):02X}{int(round(b2*255)):02X}{alpha_suffix}"
    return (r2, g2, b2) + tuple(c[3:])

def _resolve_brand(palette_name):
    """Restituisce la lista di colori hex di una palette 'brand' (fissa o personalizzata
    dall'utente nella tab Personalizzazione), oppure None se non è una palette di quel tipo."""
    custom = st.session_state.get("custom_palettes", {})
    if palette_name in custom:
        return custom[palette_name]
    return _BRAND_PALETTES.get(palette_name)


def _colors(n, palette_name):
    brand = _resolve_brand(palette_name)
    if brand:
        cols = [brand[i % len(brand)] for i in range(n)]
    else:
        try:
            cmap = plt.get_cmap(palette_name) if palette_name else plt.get_cmap("tab10")
        except Exception:
            cmap = plt.get_cmap("tab10")
        cols = [cmap(i / max(n - 1, 1)) for i in range(n)]
    if st.session_state.get("opt_palette_mode") == "Scuro":
        cols = [_lighten_for_dark_bg(c) for c in cols]
    return cols


def _get_cmap(palette_name, fallback="Blues"):
    """Restituisce un matplotlib colormap continuo (per mappe/matrici con valori numerici).
    Per le palette brand interpola i colori in un gradiente anziché usare bande discrete
    (ListedColormap produrrebbe uno scalino per ogni colore della palette)."""
    brand = _resolve_brand(palette_name)
    if brand:
        from matplotlib.colors import LinearSegmentedColormap
        return LinearSegmentedColormap.from_list(palette_name, brand)
    try:
        return plt.get_cmap(palette_name) if palette_name else plt.get_cmap(fallback)
    except Exception:
        return plt.get_cmap(fallback)


def _cat_colors(categories, palette_name):
    cats = list(categories)
    cols = _colors(len(cats), palette_name)
    return dict(zip(cats, cols))


_LEGEND_POSITIONS = {
    "Auto (migliore)":  {"loc": "best"},
    "Alto a destra":    {"loc": "upper right", "anchor": (1.0, 1.0)},
    "Alto a sinistra":  {"loc": "upper left",  "anchor": (0.0, 1.0)},
    "Basso a destra":   {"loc": "lower right", "anchor": (1.0, 0.0)},
    "Basso a sinistra": {"loc": "lower left",  "anchor": (0.0, 0.0)},
    "Fuori a destra":   {"loc": "center left",  "anchor": (1.02, 0.5)},
    "Fuori in alto":    {"loc": "lower center", "anchor": (0.5, 1.02)},
    "Fuori in basso":   {"loc": "upper center", "anchor": (0.5, -0.15)},
}


def _place_legend(ax_, opts=None, handles=None, labels=None):
    """Posiziona la legenda in modo configurabile (posizione scelta dall'utente
    + piccolo scostamento manuale in frazione di assi) invece del posizionamento
    automatico fisso di matplotlib, che spesso finisce sopra i dati.
    UNICA funzione condivisa per la legenda: usata sia da _apply_style (tutti i
    tipi di grafico "standard") sia dal blocco 3D delle barre impilate (l'unico
    che disegna la propria legenda a mano) — così una modifica al posizionamento
    si applica ovunque senza dover toccare ogni tipo di grafico singolarmente.
    Se handles/labels non sono passati li prende dagli elementi già disegnati
    sull'asse (funziona per i grafici "standard" con label= sui plot)."""
    o = opts or {}
    if handles is None:
        handles, labels = ax_.get_legend_handles_labels()
    if not handles:
        return None
    pos_key = o.get("legend_loc", "Auto (migliore)")
    pos     = _LEGEND_POSITIONS.get(pos_key, _LEGEND_POSITIONS["Auto (migliore)"])
    kwargs  = {"fontsize": o.get("legend_sz", 8), "framealpha": 0.85, "loc": pos["loc"]}
    if "anchor" in pos:
        dx = float(o.get("legend_offset_x", 0.0))
        dy = float(o.get("legend_offset_y", 0.0))
        ax_x, ax_y = pos["anchor"]
        kwargs["bbox_to_anchor"] = (ax_x + dx, ax_y + dy)
    leg = ax_.legend(handles, labels, **kwargs) if labels else ax_.legend(handles=handles, **kwargs)
    if leg:
        leg.set_title(None)
    return leg

def _grouped_bar(ax, pivot_raw, color_map, horiz=False, gap=0.15, **kw):
    """Disegna barre raggruppate: a ogni valore dell'asse, la larghezza delle barre
    si adatta al numero di categorie EFFETTIVAMENTE presenti per quel valore
    (non al numero totale di categorie nell'intero dataset), così le barre non
    risultano strette quando solo poche categorie compaiono per un dato valore x.
    `pivot_raw` deve avere NaN (non 0) nelle celle dove la combinazione non esiste.
    Restituisce la lista degli indici (valori asse), nello stesso ordine di pivot_raw.index."""
    present = pivot_raw.notna()
    vals    = pivot_raw.fillna(0)
    idx     = list(vals.index)
    cols    = list(vals.columns)
    slot    = 1.0 - gap
    seen    = set()
    for i in range(len(idx)):
        present_cols = [c for c in cols if present.iloc[i][c]]
        n_here = len(present_cols) or 1
        w = slot / n_here
        for j, c in enumerate(present_cols):
            off   = (j - (n_here - 1) / 2) * w
            v     = vals.iloc[i][c]
            color = color_map.get(c, "#888")
            lbl   = str(c) if c not in seen else None
            seen.add(c)
            if horiz:
                ax.barh(i + off, v, height=w * 0.9, color=color, label=lbl, **kw)
            else:
                ax.bar(i + off, v, width=w * 0.9, color=color, label=lbl, **kw)
    if horiz:
        ax.set_yticks(range(len(idx)))
    else:
        ax.set_xticks(range(len(idx)))
    return idx


def _apply_style(ax_, title, xlabel, ylabel, show_legend, show_grid, opts=None):
    o = opts or {}
    title_sz   = o.get("title_sz", 13)
    title_bold = o.get("title_bold", True)
    title_ital = o.get("title_ital", False)
    label_sz   = o.get("label_sz", 10)
    minimal    = o.get("minimal", False)

    if title:
        _TITLE_LOC = {"Sinistra": "left", "Centro": "center", "Destra": "right"}
        ax_.set_title(
            title,
            fontsize=title_sz,
            fontweight="bold" if title_bold else "normal",
            fontstyle="italic" if title_ital else "normal",
            loc=_TITLE_LOC.get(o.get("title_loc", "Centro"), "center"),
            y=1.0 + float(o.get("title_offset_y", 0.0)),
            pad=10,
        )
    if xlabel:
        ax_.set_xlabel(xlabel, fontsize=label_sz)
        _xox, _xoy = float(o.get("xlabel_offset_x", 0.0)), float(o.get("xlabel_offset_y", 0.0))
        if _xox or _xoy:
            # Posizione di default di matplotlib per la xlabel: centrata
            # (0.5) leggermente sotto l'asse — la si sposta SOLO se
            # scostata (a scostamento zero il comportamento resta quello
            # automatico di matplotlib, invariato).
            ax_.xaxis.set_label_coords(0.5 + _xox, -0.12 + _xoy, transform=ax_.transAxes)
    if ylabel:
        ax_.set_ylabel(ylabel, fontsize=label_sz)
        _yox, _yoy = float(o.get("ylabel_offset_x", 0.0)), float(o.get("ylabel_offset_y", 0.0))
        if _yox or _yoy:
            ax_.yaxis.set_label_coords(-0.1 + _yox, 0.5 + _yoy, transform=ax_.transAxes)
    if show_grid:
        ax_.grid(True, alpha=0.3, linestyle="--", axis="y")
    if show_legend:
        _place_legend(ax_, o)

    # Limiti Y
    y_min_s = o.get("y_min", "")
    y_max_s = o.get("y_max", "")
    if y_min_s or y_max_s:
        cur_lo, cur_hi = ax_.get_ylim()
        try:
            ax_.set_ylim(
                float(y_min_s) if y_min_s else cur_lo,
                float(y_max_s) if y_max_s else cur_hi,
            )
        except ValueError:
            pass

    # Stile minimal: rimuove spine superiore e destra
    # BUG corretto: un asse polare (radar) non ha spine "top"/"right" — solo
    # "polar" — quindi ax_.spines["top"] dava KeyError('top') ogni volta che
    # si combinava Radar + Stile minimal.
    if minimal:
        if ax_.name == "polar":
            ax_.spines["polar"].set_visible(False)
        else:
            ax_.spines["top"].set_visible(False)
            ax_.spines["right"].set_visible(False)

    # Mostra / nascondi asse Y (stesso motivo del check "polar" sopra: niente
    # spine "left"/"bottom" su un asse polare)
    if not o.get("show_axes_y", True):
        ax_.yaxis.set_visible(False)
        if ax_.name != "polar":
            ax_.spines["left"].set_visible(False)

    # Mostra / nascondi etichette asse X
    if not o.get("show_axes_x", True):
        ax_.xaxis.set_visible(False)
        if ax_.name != "polar":
            ax_.spines["bottom"].set_visible(False)

    # Sfondo e palette-per-sfondo sono DUE scelte indipendenti (non un unico
    # selettore a 3 stati): "Aggiungi sfondo" decide SOLO se il pannello è
    # trasparente o riempito con un colore a scelta (color picker, non un
    # colore fisso) — "Palette per sfondo" decide SOLO se testi/assi/palette
    # sono tarati per stare su chiaro o su scuro. Restano indipendenti perché
    # un grafico può restare trasparente qui e finire composto su uno sfondo
    # scuro altrove (slide, video) — in quel caso la palette va comunque
    # schiarita anche se qui non si vede un riempimento scuro.
    if o.get("bg_on", False):
        _bg_color = o.get("bg_color") or "white"
        ax_.set_facecolor(_bg_color)
        if ax_.get_figure():
            ax_.get_figure().patch.set_facecolor(_bg_color)

    if o.get("palette_mode") == "Scuro":
        _light = "#E8E6EF"
        ax_.title.set_color(_light)
        ax_.xaxis.label.set_color(_light)
        ax_.yaxis.label.set_color(_light)
        ax_.tick_params(colors=_light)
        for _spine in ax_.spines.values():
            _spine.set_color(_light)
        _leg = ax_.get_legend()
        if _leg:
            _leg.get_frame().set_alpha(0.15)
            for _txt in _leg.get_texts():
                _txt.set_color(_light)

    # Testi liberi aggiunti dall'utente (tab Personalizzazione → "Testi
    # aggiuntivi") — coordinate in frazione degli assi (transAxes, 0-1
    # indipendente dai dati), stesso sistema di riferimento usato dal motore
    # Plotly (xref/yref="paper") per coerenza tra i due motori di disegno.
    for _ann in (o.get("extra_texts") or []):
        try:
            ax_.text(float(_ann.get("x", 0.5)), float(_ann.get("y", 0.9)), str(_ann.get("text", "")),
                      transform=ax_.transAxes, fontsize=int(_ann.get("size", 12)),
                      ha="center", va="center")
        except Exception:
            pass


_GRID_STYLE_MAP = {
    "—  solida":        "-",
    "-- tratteggiata":  "--",
    ":  punteggiata":   ":",
    "-. tratto-punto":  "-.",
}
_SCALE_TYPES = ["Normale", "Log10", "Log₂", "Sqrt"]


def _apply_axis_settings(ax, opts):
    """Applica scala, minor tick, griglia avanzata e dimensioni font assi."""
    import matplotlib.ticker as mticker
    from matplotlib.ticker import MultipleLocator, AutoMinorLocator, LogLocator, NullFormatter
    from matplotlib.ticker import FixedLocator

    for axis_name in ("x", "y"):
        sc     = str(opts.get(f"{axis_name}_scale", "Normale"))
        step_s = str(opts.get(f"{axis_name}_step", "")).strip()
        set_scale = ax.set_xscale if axis_name == "x" else ax.set_yscale
        mpl_axis  = ax.xaxis     if axis_name == "x" else ax.yaxis
        set_lim   = ax.set_xlim  if axis_name == "x" else ax.set_ylim
        is_log    = sc in ("Log10", "Log₂")
        # Salta scala su assi categoriali (FixedLocator = barre con etichette)
        is_cat = isinstance(mpl_axis.get_major_locator(), FixedLocator)
        if not is_cat:
            try:
                if sc == "Log10":
                    set_scale("log", base=10)
                elif sc == "Log₂":
                    set_scale("log", base=2)
                elif sc == "Sqrt":
                    set_scale("function", functions=(
                        lambda v: np.sqrt(np.clip(v, 0, None)),
                        lambda v: np.square(v),
                    ))
            except Exception:
                pass
            # Scala log: assicura limite inferiore > 0 e formatter leggibile
            if is_log:
                lo, hi = (ax.get_xlim() if axis_name == "x" else ax.get_ylim())
                if lo <= 0 < hi:
                    try:
                        dmin, _ = mpl_axis.get_data_interval()
                        pos_lo  = dmin if dmin > 0 else hi / 100
                    except Exception:
                        pos_lo = hi / 100
                    try:
                        set_lim(pos_lo, hi)
                    except Exception:
                        pass
                mpl_axis.set_major_formatter(
                    mticker.FuncFormatter(
                        lambda v, pos: f"{int(v):,}" if v >= 1 and v == int(v) else f"{v:g}"
                    )
                )
            # Passo tick personalizzato (non su log, che usa LogLocator)
            if step_s and not is_log:
                try:
                    mpl_axis.set_major_locator(MultipleLocator(float(step_s)))
                except Exception:
                    pass
            # Minor tick
            if opts.get(f"{axis_name}_minor", False):
                try:
                    if is_log:
                        # Per scala log: minor tick tra le decade (2,3,...9)
                        mpl_axis.set_minor_locator(LogLocator(subs=list(range(2, 10))))
                        mpl_axis.set_minor_formatter(NullFormatter())
                    else:
                        ndiv = int(opts.get(f"{axis_name}_minor_ndiv", 5))
                        mpl_axis.set_minor_locator(AutoMinorLocator(ndiv))
                except Exception:
                    pass

    # ── Griglia avanzata ──────────────────────────────────────────────────────
    _has_adv_grid = any(opts.get(k, False) for k in
                        ("gx_maj", "gx_min", "gy_maj", "gy_min"))
    if _has_adv_grid:
        # Azzera griglia precedente per tutti gli assi / livelli
        ax.grid(False, which="both", axis="both")

    for axis_name, pfx in [("y", "gy"), ("x", "gx")]:
        if opts.get(f"{pfx}_maj", False):
            sty = _GRID_STYLE_MAP.get(str(opts.get(f"{pfx}_maj_sty", "--")), "--")
            try:
                alp = float(opts.get(f"{pfx}_maj_alp", 0.4))
                lw  = float(opts.get(f"{pfx}_maj_lw",  0.8))
                ax.grid(True, which="major", axis=axis_name,
                        linestyle=sty, alpha=alp, linewidth=lw)
            except Exception:
                pass
        if opts.get(f"{pfx}_min", False):
            sty = _GRID_STYLE_MAP.get(str(opts.get(f"{pfx}_min_sty", ":")), ":")
            try:
                alp  = float(opts.get(f"{pfx}_min_alp", 0.25))
                lw   = float(opts.get(f"{pfx}_min_lw",  0.5))
                mpl_a = ax.xaxis if axis_name == "x" else ax.yaxis
                # Minor locator (se non già impostato dalla sezione minor tick)
                if not opts.get(f"{axis_name}_minor", False):
                    ndiv = int(opts.get(f"{pfx}_min_ndiv", 5))
                    sc_n = str(opts.get(f"{axis_name}_scale", "Normale"))
                    if sc_n in ("Log10", "Log₂"):
                        mpl_a.set_minor_locator(LogLocator(subs=list(range(2, 10))))
                        mpl_a.set_minor_formatter(NullFormatter())
                    else:
                        mpl_a.set_minor_locator(AutoMinorLocator(ndiv))
                ax.grid(True, which="minor", axis=axis_name,
                        linestyle=sty, alpha=alp, linewidth=lw)
            except Exception:
                pass

    # ── Font size etichette tick e legenda ────────────────────────────────────
    xtick_sz  = opts.get("xtick_sz")
    ytick_sz  = opts.get("ytick_sz")
    legend_sz = opts.get("legend_sz")
    if xtick_sz:
        ax.tick_params(axis="x", labelsize=int(xtick_sz))
    if ytick_sz:
        ax.tick_params(axis="y", labelsize=int(ytick_sz))
    if legend_sz:
        leg = ax.get_legend()
        if leg:
            for txt in leg.get_texts():
                txt.set_fontsize(int(legend_sz))


def _normalize_geo(series: pd.Series, name_up: dict) -> pd.Series:
    """Normalizza i nomi geografici: maiuscolo + risolve alias."""
    return series.astype(str).str.strip().str.upper().map(name_up).fillna(
        series.astype(str).str.strip()
    )


def _band_split_v(geom, cum_area, x_lo, x_hi):
    """Ricerca binaria: x tale che area(geom ∩ [x_lo, x]) ≈ cum_area."""
    from shapely.geometry import box as _sb
    miny, maxy = geom.bounds[1], geom.bounds[3]
    lo, hi = x_lo, x_hi
    for _ in range(60):
        mid = (lo + hi) / 2
        a = geom.intersection(_sb(x_lo, miny, mid, maxy)).area
        if abs(a - cum_area) / (abs(cum_area) + 1e-10) < 1e-4:
            return mid
        lo, hi = (mid, hi) if a < cum_area else (lo, mid)
    return (lo + hi) / 2


def _band_split_h(geom, cum_area, y_lo, y_hi):
    """Ricerca binaria: y tale che area(geom ∩ [y_lo, y]) ≈ cum_area."""
    from shapely.geometry import box as _sb
    minx, maxx = geom.bounds[0], geom.bounds[2]
    lo, hi = y_lo, y_hi
    for _ in range(60):
        mid = (lo + hi) / 2
        a = geom.intersection(_sb(minx, y_lo, maxx, mid)).area
        if abs(a - cum_area) / (abs(cum_area) + 1e-10) < 1e-4:
            return mid
        lo, hi = (mid, hi) if a < cum_area else (lo, mid)
    return (lo + hi) / 2


# ── Rendering mappe ────────────────────────────────────────────────────────────
def _plot_choropleth(fig, ax, df_filt, geo_col, val_col, gdf, palette, title):
    """Mappa coropletica: regioni colorate per valore numerico."""
    _name_up = {n.upper().strip(): n for n in gdf["GEO_NAME"]}

    agg = df_filt.groupby(geo_col)[val_col].sum().reset_index()
    agg.columns = ["GEO_NAME", "value"]
    agg["value"] = pd.to_numeric(agg["value"], errors="coerce")
    agg["GEO_NAME"] = _normalize_geo(agg["GEO_NAME"], _name_up)

    merged = gdf.merge(agg, on="GEO_NAME", how="left")

    cmap_obj  = _get_cmap(palette, fallback="Blues")
    vmin = agg["value"].min()
    vmax = agg["value"].max()
    if pd.isna(vmin) or pd.isna(vmax) or vmin == vmax:
        vmin, vmax = 0, 1

    merged.plot(
        column="value", ax=ax, cmap=cmap_obj,
        missing_kwds={"color": "#e4e4e4", "label": "Nessun dato"},
        legend=False, edgecolor="#aaaaaa", linewidth=0.5,
        vmin=vmin, vmax=vmax,
    )
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    sm   = mcm.ScalarMappable(cmap=cmap_obj, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, shrink=0.7, pad=0.02)

    if title:
        ax.set_title(title, fontsize=13, fontweight="bold", pad=8)
    ax.axis("off")
    fig.subplots_adjust(left=0.01, right=0.88, top=0.97, bottom=0.02)


def _plot_bubble_map(fig, ax, df_filt, geo_col, val_col, grp_col, gdf, palette, title, show_legend):
    """Mappa bolla: cerchi proporzionali al valore ai centroidi."""
    gdf.plot(color="#e4e4e4", edgecolor="#aaaaaa", linewidth=0.5, ax=ax)
    centroids = gdf.set_index("GEO_NAME").geometry.centroid
    _name_up  = {n.upper().strip(): n for n in gdf["GEO_NAME"]}

    agg_cols  = [geo_col] + ([grp_col] if grp_col else [])
    if val_col:
        agg = df_filt.groupby(agg_cols)[val_col].sum().reset_index()
    else:
        agg = df_filt.groupby(agg_cols).size().reset_index(name="_n")
        val_col = "_n"

    agg["_GEO"] = _normalize_geo(agg[geo_col], _name_up)
    max_val     = pd.to_numeric(agg[val_col], errors="coerce").max() or 1

    if grp_col and grp_col in agg.columns:
        groups  = sorted(agg[grp_col].dropna().unique(), key=_xkey)
        cc      = _cat_colors(groups, palette)
        for grp in groups:
            sub = agg[agg[grp_col] == grp]
            for _, row in sub.iterrows():
                geo = row["_GEO"]
                if geo not in centroids.index:
                    continue
                cx, cy = centroids[geo].x, centroids[geo].y
                v = pd.to_numeric(row[val_col], errors="coerce")
                if pd.isna(v):
                    continue
                size = max(10, (v / max_val) * 1200)
                ax.scatter(cx, cy, s=size, color=cc[grp], alpha=0.65,
                           edgecolors="#333", linewidths=0.4, label=str(grp), zorder=4)
        if show_legend:
            handles = [plt.scatter([], [], s=60, color=cc[g], label=str(g), alpha=0.8)
                       for g in groups]
            ax.legend(handles=handles, fontsize=7, framealpha=0.9)
    else:
        base_color = _get_cmap(palette, fallback="tab10")(0)
        for _, row in agg.iterrows():
            geo = row["_GEO"]
            if geo not in centroids.index:
                continue
            cx, cy = centroids[geo].x, centroids[geo].y
            v = pd.to_numeric(row[val_col], errors="coerce")
            if pd.isna(v):
                continue
            size = max(10, (v / max_val) * 1200)
            ax.scatter(cx, cy, s=size, color=base_color, alpha=0.65,
                       edgecolors="#333", linewidths=0.4, zorder=4)

    if title:
        ax.set_title(title, fontsize=13, fontweight="bold", pad=8)
    ax.axis("off")
    fig.subplots_adjust(left=0.01, right=0.99, top=0.97, bottom=0.02)


def _plot_band_map(fig, ax, df_filt, geo_col, grp_col, val_col, gdf, palette, title, show_legend,
                   orientation="Verticale", equal_area=False):
    """
    Mappa bande: ogni area geografica è divisa in bande per categoria.
    orientation : "Verticale" (bande sinistra→destra) o "Orizzontale" (bande basso→alto)
    equal_area  : False = larghezza/altezza proporzionale al valore (default)
                  True  = area della banda proporzionale al valore
    """
    if not grp_col:
        raise ValueError("Mappa bande richiede 'Raggruppa/Colore'.")

    try:
        from shapely.geometry import box as shapely_box
    except ImportError:
        raise ImportError("shapely non installato: pip install shapely")

    gdf.plot(color="#e4e4e4", edgecolor="#aaaaaa", linewidth=0.5, ax=ax)
    _name_up = {n.upper().strip(): n for n in gdf["GEO_NAME"]}

    cats = sorted(df_filt[grp_col].dropna().unique(), key=_xkey)
    cc   = _cat_colors(cats, palette)

    df2 = df_filt.copy()
    df2["_GEO"] = _normalize_geo(df2[geo_col], _name_up)

    for _, prow in gdf.iterrows():
        gname = str(prow["GEO_NAME"])
        geom  = prow.geometry
        if geom is None or geom.is_empty:
            continue
        sub = df2[df2["_GEO"] == gname]
        if sub.empty:
            continue

        vals = {}
        for cat in cats:
            sub_c = sub[sub[grp_col] == cat]
            vals[cat] = (pd.to_numeric(sub_c[val_col], errors="coerce").sum()
                         if val_col else len(sub_c))
        total = sum(vals.values())
        if total == 0:
            continue

        minx, miny, maxx, maxy = geom.bounds

        if orientation == "Orizzontale":
            # Bande basso → alto (cut by Y)
            if equal_area:
                total_area = geom.area
                cum = 0.0
                y_prev = miny
                for i, cat in enumerate(cats):
                    if i == len(cats) - 1:
                        clip = shapely_box(minx, y_prev, maxx, maxy)
                    else:
                        cum += vals[cat]
                        y_next = _band_split_h(geom, total_area * (cum / total), miny, maxy)
                        clip   = shapely_box(minx, y_prev, maxx, y_next)
                        y_prev = y_next
                    frag = geom.intersection(clip)
                    if not frag.is_empty:
                        gpd.GeoSeries([frag], crs=gdf.crs).plot(
                            ax=ax, color=cc[cat], linewidth=0, zorder=4)
            else:
                height = maxy - miny
                y_cur  = miny
                for cat in cats:
                    sh = height * (vals[cat] / total)
                    if sh < 1:
                        continue
                    clip = shapely_box(minx, y_cur, maxx, y_cur + sh)
                    frag = geom.intersection(clip)
                    if not frag.is_empty:
                        gpd.GeoSeries([frag], crs=gdf.crs).plot(
                            ax=ax, color=cc[cat], linewidth=0, zorder=4)
                    y_cur += sh
        else:
            # Bande sinistra → destra (cut by X) — default
            if equal_area:
                total_area = geom.area
                cum = 0.0
                x_prev = minx
                for i, cat in enumerate(cats):
                    if i == len(cats) - 1:
                        clip = shapely_box(x_prev, miny, maxx, maxy)
                    else:
                        cum += vals[cat]
                        x_next = _band_split_v(geom, total_area * (cum / total), minx, maxx)
                        clip   = shapely_box(x_prev, miny, x_next, maxy)
                        x_prev = x_next
                    frag = geom.intersection(clip)
                    if not frag.is_empty:
                        gpd.GeoSeries([frag], crs=gdf.crs).plot(
                            ax=ax, color=cc[cat], linewidth=0, zorder=4)
            else:
                width = maxx - minx
                x_cur = minx
                for cat in cats:
                    sw = width * (vals[cat] / total)
                    if sw < 1:
                        continue
                    clip = shapely_box(x_cur, miny, x_cur + sw, maxy)
                    frag = geom.intersection(clip)
                    if not frag.is_empty:
                        gpd.GeoSeries([frag], crs=gdf.crs).plot(
                            ax=ax, color=cc[cat], linewidth=0, zorder=4)
                    x_cur += sw

        gpd.GeoSeries([geom], crs=gdf.crs).plot(
            ax=ax, color="none", edgecolor="#555", linewidth=0.8, zorder=5)

    if show_legend:
        import matplotlib.patches as mpatches
        patches = [mpatches.Patch(color=cc[c], label=str(c)) for c in cats]
        ax.legend(handles=patches, fontsize=7, framealpha=0.9,
                  loc="lower right", title=grp_col)

    if title:
        ax.set_title(title, fontsize=13, fontweight="bold", pad=8)
    ax.axis("off")
    fig.subplots_adjust(left=0.01, right=0.99, top=0.97, bottom=0.02)


def _plot_pin_map(fig, ax, df_filt, geo_col, val_col, grp_col, gdf, palette, title, show_legend):
    """Mappa pin: marker ai centroidi, colorati per valore/gruppo."""
    gdf.plot(color="#e4e4e4", edgecolor="#aaaaaa", linewidth=0.5, ax=ax)
    centroids = gdf.set_index("GEO_NAME").geometry.centroid
    _name_up  = {n.upper().strip(): n for n in gdf["GEO_NAME"]}

    df2 = df_filt.copy()
    df2["_GEO"] = _normalize_geo(df2[geo_col], _name_up)

    if grp_col and grp_col in df2.columns:
        groups = sorted(df2[grp_col].dropna().unique(), key=_xkey)
        cc     = _cat_colors(groups, palette)
        for grp in groups:
            sub = df2[df2[grp_col] == grp]
            xs, ys = [], []
            for _, row in sub.iterrows():
                geo = row["_GEO"]
                if geo in centroids.index:
                    xs.append(centroids[geo].x)
                    ys.append(centroids[geo].y)
            if xs:
                ax.scatter(xs, ys, s=60, color=cc[grp], alpha=0.85,
                           edgecolors="#333", linewidths=0.5,
                           label=str(grp), zorder=4, marker="o")
        if show_legend:
            handles = [plt.scatter([], [], s=60, color=cc[g], label=str(g), alpha=0.85)
                       for g in groups]
            ax.legend(handles=handles, fontsize=7, framealpha=0.9)
    else:
        # Colora per valore se disponibile, altrimenti colore fisso
        base_color = _get_cmap(palette, fallback="tab10")(0.2)
        if val_col and val_col in df2.columns:
            agg = df2.groupby("_GEO")[val_col].sum().reset_index()
            vvals = pd.to_numeric(agg[val_col], errors="coerce")
            vmin, vmax = vvals.min(), vvals.max()
            cmap = _get_cmap(palette, fallback="YlOrRd")
            norm = mcolors.Normalize(vmin=vmin, vmax=vmax if vmax != vmin else vmin + 1)
            for _, row in agg.iterrows():
                geo = row["_GEO"]
                if geo not in centroids.index:
                    continue
                v = pd.to_numeric(row[val_col], errors="coerce")
                if pd.isna(v):
                    continue
                c = cmap(norm(v))
                ax.scatter(centroids[geo].x, centroids[geo].y,
                           s=60, color=c, alpha=0.85,
                           edgecolors="#333", linewidths=0.5, zorder=4)
            sm = mcm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            fig.colorbar(sm, ax=ax, shrink=0.6, pad=0.02)
        else:
            for gname, pt in centroids.items():
                ax.scatter(pt.x, pt.y, s=60, color=base_color, alpha=0.85,
                           edgecolors="#333", linewidths=0.5, zorder=4)

    if title:
        ax.set_title(title, fontsize=13, fontweight="bold", pad=8)
    ax.axis("off")
    fig.subplots_adjust(left=0.01, right=0.99, top=0.97, bottom=0.02)


# ── Squarify helpers (implementazione interna, senza libreria esterna) ─────────
def _sq_worst(areas, side):
    """Rapporto d'aspetto peggiore di una riga squarify."""
    s = sum(areas)
    if s <= 0 or side <= 0:
        return float("inf")
    w = s / side
    worst = 0.0
    for a in areas:
        if a <= 0:
            return float("inf")
        h = a / w
        worst = max(worst, max(w / h, h / w))
    return worst


def _squarify_items(items, x0, y0, dx, dy, result):
    """Layout squarify ricorsivo.
    items  = [(area, label, color), ...]  — le aree devono sommare a dx*dy
    result = lista in output: [(x, y, dx, dy, label, color), ...]
    """
    if not items:
        return
    if len(items) == 1:
        result.append((x0, y0, dx, dy, items[0][1], items[0][2]))
        return
    side = min(dx, dy) if min(dx, dy) > 0 else max(dx, dy)
    if side <= 0:
        return
    row, prev_worst = [], float("inf")
    for item in items:
        candidate = row + [item]
        wr = _sq_worst([a for a, _, _ in candidate], side)
        if row and wr > prev_worst:
            break
        row, prev_worst = candidate, wr
    s = sum(a for a, _, _ in row)
    strip = s / side if side > 0 else 0
    if dx >= dy:
        oy = y0
        for a, lbl, col in row:
            h = a / strip if strip > 0 else 0
            result.append((x0, oy, strip, h, lbl, col))
            oy += h
        _squarify_items(items[len(row):], x0 + strip, y0, dx - strip, dy, result)
    else:
        ox = x0
        for a, lbl, col in row:
            w = a / strip if strip > 0 else 0
            result.append((ox, y0, w, strip, lbl, col))
            ox += w
        _squarify_items(items[len(row):], x0, y0 + strip, dx, dy - strip, result)


def _render_chart_on_ax(ax, df_sub, chart_key, x, y, g,
                         palette, show_nums, xtick_rot, data_lbl_sz, extra=None):
    """Renderizza grafico su ax; restituisce ax (può essere nuovo asse polare per radar)."""
    ex          = extra or {}
    horiz       = bool(ex.get("horiz",      False))
    stacked_3d  = bool(ex.get("stacked_3d", False))
    show_line   = bool(ex.get("show_line",  False))
    line_col    = ex.get("line_col") or None
    podio_n     = max(1, min(10, int(ex.get("podio_n", 3))))
    show_legend = bool(ex.get("show_legend",  True))
    show_xlabels = bool(ex.get("show_xlabels", True))
    y_scale_3d  = str(ex.get("y_scale", "Normale"))
    sort_y_desc = bool(ex.get("sort_y_desc", False))
    highlight_set = {str(v) for v in (ex.get("highlight_cats") or [])}
    highlight_color = ex.get("highlight_color") or "#E63946"
    is_freq    = (y == _FREQ_LABEL or not y)

    def _with_highlight(cols_, labels_):
        """Sostituisce il colore di palette con quello evidenziato per le
        categorie selezionate dall'utente (solo bar chart singolo, colore per
        categoria X — non ha un mapping univoco per barre raggruppate/impilate,
        dove il colore è per serie/gruppo e non per categoria)."""
        if not highlight_set:
            return cols_
        return [highlight_color if str(lbl) in highlight_set else c
                for lbl, c in zip(labels_, cols_)]

    # ── BARRE (verticali / orizzontali) ──────────────────────────────────────
    if chart_key == "bar":
        if not x:
            raise ValueError("Seleziona almeno Asse X.")
        if not is_freq and g:
            agg   = _agg(df_sub, x, y, g)
            pivot = agg.pivot_table(index=x, columns=g, values=y, aggfunc="sum")
            if sort_y_desc:
                pivot = pivot.loc[pivot.sum(axis=1).sort_values(ascending=False).index]
            cols  = _colors(len(pivot.columns), palette)
            cmap_ = dict(zip(pivot.columns, cols))
            idx   = _grouped_bar(ax, pivot, cmap_, horiz=horiz)
            if horiz:
                ax.set_yticklabels([str(v) for v in idx], fontsize=7)
                ax.set_xlabel(y or ""); ax.set_ylabel(x or "")
            else:
                ax.set_xlabel(x or ""); ax.set_ylabel(y or "")
                _smart_labels(ax, idx, xtick_rot)
            if show_nums:
                for cnt in ax.containers:
                    ax.bar_label(cnt, fmt="%.4g", fontsize=data_lbl_sz, padding=2)
        elif not is_freq:
            agg_df, val_col = _agg_by_x(df_sub, x, y, sort_y_desc=sort_y_desc)
            vals = pd.to_numeric(agg_df[val_col], errors="coerce").fillna(0)
            cols = _with_highlight(_colors(len(agg_df), palette), agg_df[x])
            if horiz:
                ax.barh(range(len(agg_df)), vals, color=cols)
                ax.set_yticks(range(len(agg_df)))
                ax.set_yticklabels(agg_df[x].astype(str), fontsize=7)
            else:
                ax.bar(range(len(agg_df)), vals, color=cols)
                ax.set_xticks(range(len(agg_df)))
                _smart_labels(ax, agg_df[x].astype(str), xtick_rot)
            if show_nums and ax.containers:
                ax.bar_label(ax.containers[0], fmt="%.4g", fontsize=data_lbl_sz, padding=2)
        else:
            # Frequenza: con o senza g
            if g:
                frq = df_sub.groupby([x, g], sort=False).size().reset_index(name="Frequenza")
                pivot = frq.pivot_table(index=x, columns=g, values="Frequenza")
                if sort_y_desc:
                    pivot = pivot.loc[pivot.sum(axis=1).sort_values(ascending=False).index]
                cols  = _colors(len(pivot.columns), palette)
                cmap_ = dict(zip(pivot.columns, cols))
                idx   = _grouped_bar(ax, pivot, cmap_, horiz=horiz)
                if horiz:
                    ax.set_yticklabels([str(v) for v in idx], fontsize=7)
                    ax.set_xlabel("Frequenza"); ax.set_ylabel(x or "")
                else:
                    ax.set_xlabel(x or ""); ax.set_ylabel("Frequenza")
                    _smart_labels(ax, idx, xtick_rot)
                if show_nums:
                    for cnt in ax.containers:
                        ax.bar_label(cnt, fmt="%d", fontsize=data_lbl_sz, padding=2)
            else:
                agg_df, _ = _agg_by_x(df_sub, x, _FREQ_LABEL, sort_y_desc=sort_y_desc)
                vals = pd.to_numeric(agg_df["Frequenza"], errors="coerce").fillna(0)
                cols = _with_highlight(_colors(len(agg_df), palette), agg_df[x])
                if horiz:
                    ax.barh(range(len(agg_df)), vals, color=cols)
                    ax.set_yticks(range(len(agg_df)))
                    ax.set_yticklabels(agg_df[x].astype(str), fontsize=7)
                else:
                    ax.bar(range(len(agg_df)), vals, color=cols)
                    ax.set_xticks(range(len(agg_df)))
                    _smart_labels(ax, agg_df[x].astype(str), xtick_rot)
                if show_nums and ax.containers:
                    ax.bar_label(ax.containers[0], fmt="%d", fontsize=data_lbl_sz, padding=2)

    # ── BARRE IMPILATE (normale + 3D mattoncini) ──────────────────────────────
    elif chart_key == "bar_stacked":
        if not x:
            raise ValueError("Barre impilate richiedono almeno Asse X.")
        if stacked_3d:
            import matplotlib.patches as mpatches
            if not g:
                raise ValueError("La variante 3D (Pila mattoncini) richiede anche Raggruppa.")
            if not y:
                raise ValueError("La variante 3D richiede anche Asse Y.")
            groups = sorted(df_sub[g].dropna().unique(), key=_xkey)
            cc     = _cat_colors(groups, palette)
            agg_d  = _agg(df_sub, x, y, g)
            _piv   = agg_d.pivot_table(index=x, columns=g, values=y, aggfunc="sum").fillna(0)
            if sort_y_desc:
                cats = _piv.sum(axis=1).sort_values(ascending=False).index.tolist()
            else:
                cats = sorted(df_sub[x].dropna().unique(), key=_xkey)
            pivot  = _piv.reindex(
                index=cats,
                columns=[gr for gr in groups if gr in _piv.columns],
                fill_value=0)
            dx_, dy_, bar_w = 0.22, 0.11, 0.6
            ax.axis("off")
            max_val = pivot.sum(axis=1).max() or 1
            n_cats  = len(cats)

            # Trasforma i valori secondo la scala Y scelta (Normale/Log10/Log2/Sqrt),
            # replicando l'effetto di un vero asse Y in scala log su barre impilate:
            # i CONFINI (base e base+val) vengono trasformati, non l'altezza in sé,
            # cosi' la pila resta coerente e i segmenti si comprimono/espandono come
            # farebbe un asse log reale.
            _floor = max_val * 1e-3 if max_val > 0 else 1e-3

            def _scale_v(v):
                if y_scale_3d == "Log10":
                    return np.log10(max(v, _floor))
                if y_scale_3d == "Log₂":
                    return np.log2(max(v, _floor))
                if y_scale_3d == "Sqrt":
                    return np.sqrt(max(v, 0.0))
                return v

            _lo_t   = _scale_v(_floor if y_scale_3d in ("Log10", "Log₂") else 0.0)
            _hi_t   = _scale_v(max_val)
            _span_t = (_hi_t - _lo_t) or 1.0
            y_scale = (n_cats * 1.2) / _span_t

            def _shade(col_hex, factor):
                r_, g_, b_, a_ = mcolors.to_rgba(col_hex)
                return (min(1, r_ * factor), min(1, g_ * factor), min(1, b_ * factor), a_)

            for ci, cat in enumerate(cats):
                x0   = ci * 1.2
                base = 0.0
                for grp in groups:
                    if grp not in pivot.columns:
                        continue
                    val = float(pivot.loc[cat, grp]) if cat in pivot.index else 0.0
                    if val <= 0:
                        continue
                    b_ = (_scale_v(base) - _lo_t) * y_scale
                    top_ = (_scale_v(base + val) - _lo_t) * y_scale
                    h  = top_ - b_
                    c  = cc[grp]
                    ax.add_patch(mpatches.Polygon(
                        [[x0, b_], [x0+bar_w, b_], [x0+bar_w, b_+h], [x0, b_+h]],
                        closed=True, facecolor=c, edgecolor="white", linewidth=0.6, zorder=3))
                    ax.add_patch(mpatches.Polygon(
                        [[x0, b_+h], [x0+bar_w, b_+h],
                         [x0+bar_w+dx_, b_+h+dy_], [x0+dx_, b_+h+dy_]],
                        closed=True, facecolor=_shade(c, 1.35), edgecolor="white",
                        linewidth=0.6, zorder=3))
                    ax.add_patch(mpatches.Polygon(
                        [[x0+bar_w, b_], [x0+bar_w+dx_, b_+dy_],
                         [x0+bar_w+dx_, b_+h+dy_], [x0+bar_w, b_+h]],
                        closed=True, facecolor=_shade(c, 0.65), edgecolor="white",
                        linewidth=0.6, zorder=3))
                    base += val
                if show_xlabels:
                    _rot = max(0, xtick_rot) if xtick_rot and xtick_rot > 0 else 0
                    ax.text(x0 + bar_w / 2, -0.15, str(cat),
                            ha="right" if _rot else "center", va="top",
                            rotation=_rot, fontsize=8, color="#333")
            handles = [mpatches.Patch(facecolor=cc[gr], label=str(gr))
                       for gr in groups if gr in cc]
            if handles and show_legend:
                _place_legend(ax, ex, handles=handles)
            total_w = (n_cats - 1) * 1.2 + bar_w + dx_ + 0.3
            ax.set_xlim(-0.3, total_w)
            ax.set_ylim(-0.4, (n_cats * 1.2) * 1.25 + dy_ + 0.2)
        else:
            if not (y and g):
                raise ValueError("Barre impilate richiedono Asse X, Asse Y e Raggruppa.")
            agg   = _agg(df_sub, x, y, g)
            pivot = agg.pivot_table(index=x, columns=g, values=y, aggfunc="sum").fillna(0)
            if sort_y_desc:
                pivot = pivot.loc[pivot.sum(axis=1).sort_values(ascending=False).index]
            pivot.plot.bar(ax=ax, stacked=True, color=_colors(len(pivot.columns), palette))
            ax.set_xlabel(x or ""); ax.set_ylabel(y or "")
            _smart_labels(ax, pivot.index, xtick_rot)
            if show_nums:
                for cnt in ax.containers:
                    ax.bar_label(cnt, fmt="%.4g", fontsize=data_lbl_sz, padding=2)

    # ── LINEE ─────────────────────────────────────────────────────────────────
    elif chart_key == "line":
        if not x:
            raise ValueError("Seleziona almeno Asse X.")
        x_all = sorted(df_sub[x].dropna().unique(), key=_xkey)
        if sort_y_desc:
            x_all = _sort_cats_desc(df_sub, x, x_all, y)
        x_pos = {str(v): i for i, v in enumerate(x_all)}
        if g:
            groups = sorted(df_sub[g].dropna().unique(), key=_xkey)
            colors = _colors(len(groups), palette)
            for grp, col_ in zip(groups, colors):
                sub            = df_sub[df_sub[g] == grp]
                agg_sub, vcol  = _agg_by_x(sub, x, y)
                xvals = [x_pos.get(str(v)) for v in agg_sub[x]]
                yvals = pd.to_numeric(agg_sub[vcol], errors="coerce").tolist()
                pairs = [(xv, yv) for xv, yv in zip(xvals, yvals)
                         if xv is not None and not pd.isna(yv)]
                # La linea va sempre disegnata in ordine di POSIZIONE X (xv),
                # non nell'ordine delle righe di agg_sub — che resta l'ordine
                # naturale anche quando x_pos è stato rimescolato da
                # sort_y_desc, altrimenti la linea salta avanti e indietro
                # collegando i punti nell'ordine sbagliato.
                pairs.sort(key=lambda p: p[0])
                if pairs:
                    xs_, ys_ = zip(*pairs)
                    ax.plot(list(xs_), list(ys_), marker="o", markersize=4,
                            label=str(grp), color=col_)
        else:
            agg_df, vcol = _agg_by_x(df_sub, x, y)
            xvals = [x_pos.get(str(v)) for v in agg_df[x]]
            yvals = pd.to_numeric(agg_df[vcol], errors="coerce").tolist()
            pairs = [(xv, yv) for xv, yv in zip(xvals, yvals)
                     if xv is not None and not pd.isna(yv)]
            pairs.sort(key=lambda p: p[0])
            if pairs:
                xs_, ys_ = zip(*pairs)
                ax.plot(list(xs_), list(ys_), marker="o", markersize=4,
                        color=_colors(3, palette)[1])
        ax.set_xticks(range(len(x_all)))
        _smart_labels(ax, [str(v) for v in x_all], xtick_rot)

    # ── DISPERSIONE ───────────────────────────────────────────────────────────
    elif chart_key == "scatter":
        if not (x and y):
            raise ValueError("Seleziona Asse X e Asse Y numerici.")
        xv = pd.to_numeric(df_sub[x], errors="coerce")
        yv = pd.to_numeric(df_sub[y], errors="coerce")
        if g:
            groups = sorted(df_sub[g].dropna().unique(), key=_xkey)
            cc     = _cat_colors(groups, palette)
            for grp in groups:
                mask = df_sub[g] == grp
                col_ = cc[grp]
                ax.scatter(xv[mask], yv[mask], s=40, alpha=0.7,
                           color=col_, edgecolors="white", linewidths=0.4,
                           label=str(grp))
                if show_line:
                    tmp = pd.DataFrame({"xv": xv[mask], "yv": yv[mask]}).dropna().sort_values("xv")
                    if len(tmp) >= 2:
                        ax.plot(tmp["xv"].values, tmp["yv"].values,
                                color=col_, alpha=0.5, linewidth=1)
        else:
            col_ = _colors(3, palette)[1]
            ax.scatter(xv, yv, s=40, alpha=0.7, color=col_,
                       edgecolors="white", linewidths=0.4)
            if show_line:
                tmp = pd.DataFrame({"xv": xv, "yv": yv}).dropna().sort_values("xv")
                if len(tmp) >= 2:
                    ax.plot(tmp["xv"].values, tmp["yv"].values,
                            color=col_, alpha=0.5, linewidth=1)

    # ── ISTOGRAMMA ────────────────────────────────────────────────────────────
    elif chart_key == "histogram":
        col_ = y or x
        if not col_:
            raise ValueError("Seleziona una colonna numerica.")
        if g:
            groups = sorted(df_sub[g].dropna().unique(), key=_xkey)
            colors = _colors(len(groups), palette)
            for grp, clr in zip(groups, colors):
                vals = pd.to_numeric(
                    df_sub.loc[df_sub[g] == grp, col_], errors="coerce"
                ).dropna()
                ax.hist(vals, bins=20, alpha=0.55, label=str(grp), color=clr)
        else:
            vals = pd.to_numeric(df_sub[col_], errors="coerce").dropna()
            ax.hist(vals, bins=20, color=_colors(3, palette)[1])

    # ── TORTA ─────────────────────────────────────────────────────────────────
    elif chart_key == "pie":
        if not x:
            raise ValueError("Seleziona Asse X (categorie).")
        if g:
            # ── Torta a doppio anello: anello interno = totale per gruppo,
            #    anello esterno = categorie X suddivise all'interno di ciascun gruppo ──
            if is_freq:
                agg = df_sub.groupby([x, g], sort=False).size().reset_index(name="_v")
                vcol = "_v"
            else:
                agg = df_sub.groupby([x, g], sort=False)[y].sum().reset_index()
                vcol = y
            agg[vcol] = pd.to_numeric(agg[vcol], errors="coerce").fillna(0)
            group_tot = agg.groupby(g)[vcol].sum().sort_values(ascending=False)
            groups    = list(group_tot.index)
            cc        = _cat_colors(groups, palette)

            inner_vals, inner_labels, inner_colors = [], [], []
            outer_vals, outer_labels, outer_colors = [], [], []
            for grp in groups:
                sub = agg[agg[g] == grp].sort_values(vcol, ascending=False).head(8)
                tot = float(sub[vcol].sum())
                if tot <= 0:
                    continue
                inner_vals.append(tot); inner_labels.append(str(grp)); inner_colors.append(cc[grp])
                base  = mcolors.to_rgb(cc[grp])
                n_sub = len(sub)
                for i, (_, row) in enumerate(sub.iterrows()):
                    shade = 0.45 + 0.5 * (1 - i / max(n_sub - 1, 1))
                    tint  = tuple(c * shade + (1 - shade) for c in base)
                    outer_colors.append(tint)
                    outer_vals.append(row[vcol]); outer_labels.append(str(row[x]))

            if not inner_vals:
                raise ValueError("Nessun dato da mostrare per Asse X / Raggruppa selezionati.")

            ax.pie(outer_vals, radius=1.0, labels=outer_labels, labeldistance=1.05,
                   colors=outer_colors, startangle=90,
                   wedgeprops=dict(width=0.35, edgecolor="white", linewidth=0.8),
                   textprops={"fontsize": 7}, autopct="%1.0f%%", pctdistance=0.82)
            ax.pie(inner_vals, radius=0.65, labels=inner_labels, labeldistance=0.38,
                   colors=inner_colors, startangle=90,
                   wedgeprops=dict(width=0.35, edgecolor="white", linewidth=0.8),
                   textprops={"fontsize": 8, "fontweight": "bold"})
            ax.axis("equal")
        else:
            agg_df, vcol = _agg_by_x(df_sub, x, y)
            agg_df = agg_df.sort_values(vcol, ascending=False)
            top    = agg_df.head(12)
            vals   = pd.to_numeric(top[vcol], errors="coerce").fillna(0)
            _, _, autotexts = ax.pie(
                vals.values,
                labels=[str(l) for l in top[x]],
                autopct="%1.1f%%", startangle=90,
                colors=_colors(len(top), palette),
            )
            for t in autotexts:
                t.set_fontsize(8)
            ax.axis("equal")

    # ── HEATMAP (3 input: x=colonne, g=righe, y=valori/colore) ───────────────
    elif chart_key == "heatmap":
        if not (x and g):
            raise ValueError("Heatmap richiede Asse X1 (colonne) e Asse X2 (righe).")
        if is_freq:
            piv = (df_sub.groupby([g, x], sort=False).size()
                   .unstack(x).fillna(0))
            cbar_lbl = "Frequenza"
        else:
            piv = df_sub.pivot_table(index=g, columns=x, values=y,
                                     aggfunc="mean", fill_value=0)
            cbar_lbl = str(y)
        if piv.empty:
            raise ValueError("Heatmap vuota: nessun dato con le selezioni correnti.")
        cmap   = _get_cmap(palette, fallback="YlOrRd")
        vmax   = float(piv.values.max()) or 1.0
        im     = ax.imshow(piv.values, aspect="auto", cmap=cmap, vmin=0, vmax=vmax)
        cols_x = list(range(len(piv.columns)))
        rows_y = list(range(len(piv.index)))
        ax.set_xticks(cols_x)
        ax.set_xticklabels([str(c) for c in piv.columns],
                            rotation=45, ha="right", fontsize=7)
        ax.set_yticks(rows_y)
        ax.set_yticklabels([str(r) for r in piv.index], fontsize=7)
        for i in rows_y:
            for j in cols_x:
                v      = piv.iloc[i, j]
                txtcol = "white" if (v / vmax) > 0.55 else "#111"
                ax.text(j, i, f"{v:,.0f}", ha="center", va="center",
                        fontsize=6, color=txtcol)
        cb = ax.get_figure().colorbar(im, ax=ax, shrink=0.7)
        cb.set_label(cbar_lbl, fontsize=8)

    # ── BOXPLOT ───────────────────────────────────────────────────────────────
    elif chart_key == "boxplot":
        if not y:
            raise ValueError("Seleziona Asse Y (colonna numerica).")
        mp = dict(color="black", linewidth=1.5)
        if x:
            cats = sorted(df_sub[x].dropna().unique(), key=_xkey)
            if sort_y_desc:
                cats = _sort_cats_desc(df_sub, x, cats, y, agg="median")
            data = [pd.to_numeric(
                df_sub.loc[df_sub[x] == c, y], errors="coerce"
            ).dropna() for c in cats]
            bp = ax.boxplot(data, patch_artist=True, medianprops=mp)
            for patch, col_ in zip(bp["boxes"], _colors(len(cats), palette)):
                patch.set_facecolor(col_)
            ax.set_xticks(range(1, len(cats) + 1))
            _smart_labels(ax, [str(c) for c in cats], xtick_rot)
        else:
            vals = pd.to_numeric(df_sub[y], errors="coerce").dropna()
            bp   = ax.boxplot(vals, patch_artist=True, medianprops=mp)
            bp["boxes"][0].set_facecolor(_colors(3, palette)[1])

    # ── RADAR ─────────────────────────────────────────────────────────────────
    elif chart_key == "radar":
        if not (x and y):
            raise ValueError("Radar richiede Asse X (categorie) e Asse Y (valori).")
        cats = sorted(df_sub[x].dropna().unique(), key=_xkey)
        if sort_y_desc:
            cats = _sort_cats_desc(df_sub, x, cats, y, agg="mean")
        N    = len(cats)
        if N < 3:
            raise ValueError("Il radar richiede almeno 3 categorie nell'Asse X.")
        angles   = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
        angles_c = angles + angles[:1]
        # rimpiazza asse normale con asse polare (restituito alla fine)
        fig_  = ax.get_figure()
        pos   = ax.get_position()
        ax.remove()
        ax = fig_.add_axes([pos.x0, pos.y0, pos.width, pos.height], polar=True)
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)

        def _mean_per_cat(sub):
            out = []
            for cat in cats:
                v = pd.to_numeric(sub.loc[sub[x] == cat, y], errors="coerce").mean()
                out.append(0.0 if pd.isna(v) else float(v))
            return out

        if g:
            groups = sorted(df_sub[g].dropna().unique(), key=_xkey)
            cc     = _cat_colors(groups, palette)
            for grp in groups:
                vals_c = _mean_per_cat(df_sub[df_sub[g] == grp]) + [0]
                vals_c[-1] = vals_c[0]
                c = cc[grp]
                ax.plot(angles_c, vals_c, color=c, linewidth=2, label=str(grp))
                ax.fill(angles_c, vals_c, alpha=0.15, color=c)
        else:
            vals_c = _mean_per_cat(df_sub) + [0]
            vals_c[-1] = vals_c[0]
            c = _colors(3, palette)[1]
            ax.plot(angles_c, vals_c, color=c, linewidth=2)
            ax.fill(angles_c, vals_c, alpha=0.2, color=c)
        ax.set_xticks(angles)
        ax.set_xticklabels([str(c) for c in cats], fontsize=8)
        ax.grid(True, alpha=0.35)

    # ── IMBUTO (FUNNEL) ───────────────────────────────────────────────────────
    elif chart_key == "funnel":
        if not x:
            raise ValueError("Seleziona Asse X (etichette) per l'imbuto.")
        import matplotlib.patches as mpatches
        cx, max_h, min_h = 0.5, 0.42, 0.07

        def _conn_f(dst, src, col_):
            ax.annotate("", xy=dst, xytext=src,
                        arrowprops=dict(arrowstyle="-", color=col_, lw=0.85, alpha=0.65), zorder=3)

        if g and not is_freq:
            # ── Funnel con Raggruppa: ogni trapezio è diviso in segmenti per gruppo ──
            cats   = sorted(df_sub[x].dropna().unique(), key=_xkey)
            groups = sorted(df_sub[g].dropna().unique(), key=_xkey)
            totals = {cat: float(pd.to_numeric(df_sub.loc[df_sub[x]==cat, y if y else "_"],
                                               errors="coerce").sum() if y else len(df_sub[df_sub[x]==cat]))
                      for cat in cats}
            # somma reale: se y è una colonna, somma; altrimenti conta le righe
            if y and y != _FREQ_LABEL:
                totals = {cat: float(pd.to_numeric(df_sub.loc[df_sub[x]==cat, y],
                                                   errors="coerce").sum()) for cat in cats}
            else:
                totals = {cat: float((df_sub[x] == cat).sum()) for cat in cats}
            cats   = sorted(cats, key=lambda c: totals.get(c, 0), reverse=True)
            N      = len(cats)
            if N < 2:
                raise ValueError("L'imbuto richiede almeno 2 categorie.")
            if y and y != _FREQ_LABEL:
                piv = (df_sub.groupby([x, g])[y].sum()).unstack(g).reindex(cats).fillna(0)
            else:
                piv = (df_sub.groupby([x, g]).size()).unstack(g).reindex(cats).fillna(0)
            piv    = piv.reindex(columns=[gr for gr in groups if gr in piv.columns])
            v_max  = max(totals.values()) or 1
            halfs  = [min_h + (totals.get(c, 0) / v_max) * (max_h - min_h) for c in cats]
            cc     = _cat_colors(groups, palette)
            ax.axis("off"); ax.set_xlim(-0.6, 1.6); ax.set_ylim(-0.3, N + 0.3)
            for i, cat in enumerate(cats):
                y_top = N - i;  y_bot = N - i - 1;  y_mid = (y_top + y_bot) / 2
                h_top = halfs[i];  h_bot = halfs[i+1] if i+1 < N else min_h
                row_tot = float(piv.loc[cat].sum()) if cat in piv.index else 1.0
                if row_tot <= 0:
                    row_tot = 1.0
                off_top = cx - h_top;  off_bot = cx - h_bot
                full_top = 2 * h_top;  full_bot = 2 * h_bot
                for grp in groups:
                    val = float(piv.loc[cat, grp]) if (cat in piv.index and grp in piv.columns) else 0.0
                    if val <= 0:
                        continue
                    frac   = val / row_tot
                    sw_top = frac * full_top;  sw_bot = frac * full_bot
                    col_   = cc[grp]
                    ax.add_patch(mpatches.Polygon(
                        [[off_top, y_top], [off_top+sw_top, y_top],
                         [off_bot+sw_bot, y_bot], [off_bot, y_bot]],
                        closed=True, facecolor=col_, edgecolor="white", linewidth=0.8, zorder=2))
                    off_top += sw_top;  off_bot += sw_bot
                ax.plot([cx-h_top, cx+h_top], [y_top, y_top],
                        color="white", linewidth=1.2, zorder=4)
                pct = totals.get(cat, 0) / v_max * 100
                _conn_f((-0.04, y_mid), (cx-h_top, y_mid), "#999")
                ax.text(-0.08, y_mid, str(cat),
                        ha="right", va="center", fontsize=9, fontweight="bold",
                        color="#1a1a1a", zorder=5)
                _conn_f((1.04, y_mid), (cx+h_top, y_mid), "#999")
                ax.text(1.08, y_mid, f"{int(round(totals.get(cat,0))):,}  ({pct:.1f} %)",
                        ha="left", va="center", fontsize=9, color="#1a1a1a", zorder=5)
            handles = [mpatches.Patch(facecolor=cc[gr], label=str(gr))
                       for gr in groups if gr in cc]
            if handles:
                ax.legend(handles=handles, fontsize=8, loc="lower right")
        else:
            # ── Funnel semplice (un colore per livello) ────────────────────────
            agg_df, vcol = _agg_by_x(df_sub, x, y)
            agg_df  = agg_df.sort_values(vcol, ascending=False).reset_index(drop=True)
            labels  = list(agg_df[x].astype(str))
            values  = list(pd.to_numeric(agg_df[vcol], errors="coerce").fillna(0))
            N = len(labels)
            if N < 2:
                raise ValueError("L'imbuto richiede almeno 2 categorie.")
            colors = _colors(N, palette)
            v_max  = max(values) or 1
            halfs  = [min_h + (v / v_max) * (max_h - min_h) for v in values]
            ax.axis("off"); ax.set_xlim(-0.6, 1.6); ax.set_ylim(-0.3, N + 0.3)
            for i in range(N):
                col_   = colors[i]
                y_top  = N - i;  y_bot = N - i - 1;  y_mid = (y_top + y_bot) / 2
                h_top  = halfs[i];  h_bot = halfs[i + 1] if i + 1 < N else min_h
                r0, g0, b0, _ = mcolors.to_rgba(col_)
                hi = (min(1, r0*1.4), min(1, g0*1.4), min(1, b0*1.4), 0.7)
                ax.add_patch(mpatches.Polygon(
                    [[cx-h_top, y_top], [cx+h_top, y_top],
                     [cx+h_bot, y_bot], [cx-h_bot, y_bot]],
                    closed=True, facecolor=col_, edgecolor="white", linewidth=0.8, zorder=2))
                ax.plot([cx-h_top, cx+h_top], [y_top, y_top],
                        color=hi, linewidth=2.5, zorder=4, solid_capstyle="round")
                _conn_f((-0.04, y_mid), (cx-h_top, y_mid), col_)
                ax.text(-0.08, y_mid, str(labels[i]),
                        ha="right", va="center", fontsize=9, fontweight="bold",
                        color="#1a1a1a", zorder=5)
                pct = values[i] / values[0] * 100 if values[0] else 0
                _conn_f((1.04, y_mid), (cx+h_top, y_mid), col_)
                ax.text(1.08, y_mid, f"{int(round(values[i])):,}  ({pct:.1f} %)",
                        ha="left", va="center", fontsize=9, color="#1a1a1a", zorder=5)

    # ── PODIO ─────────────────────────────────────────────────────────────────
    elif chart_key == "podio":
        if not x:
            raise ValueError("Seleziona Asse X (nomi) per il podio.")
        from matplotlib.patches import Rectangle
        if y and y in df_sub.columns and y != _FREQ_LABEL:
            nums = pd.to_numeric(df_sub[y], errors="coerce")
            df2  = df_sub.assign(_score=nums).dropna(subset=["_score"])
            df2  = df2.sort_values("_score", ascending=False).reset_index(drop=True)
            score_col = y
        else:
            df2 = df_sub.copy().reset_index(drop=True)
            df2["_score"] = (len(df2) - df2.index).astype(float)
            score_col = None
        n = min(len(df2), podio_n)
        if n == 0:
            raise ValueError("Nessun dato per il podio.")
        df2 = df2.iloc[:n].copy()

        _MEDAL   = [("#D4AF37","#9A7600"), ("#B0B0BC","#6A6A80"), ("#CD7F32","#8B4513")]
        _extra_c = _colors(max(1, n), palette)
        ORDINALS = ("1°","2°","3°","4°","5°","6°","7°","8°","9°","10°")
        _STEP_H  = {0: 1.00, 1: 0.70, 2: 0.50}

        def _gc(ri):
            return _MEDAL[ri] if ri < 3 else (_extra_c[ri % len(_extra_c)], "#555555")
        def _sh(ri):
            return _STEP_H.get(ri, max(0.25, 0.50 - (ri - 2) * 0.06))

        _OFF  = (0, 1, -1, -2, 2, -3, 3, -4, 4, -5)
        ctr   = -min(_OFF[:n])
        col4r = [ctr + _OFF[i] for i in range(n)]
        col_w = 1.0;  sw = 0.86;  card_h = 0.58;  cgap = 0.05
        strip_h = 0.085;  base_h = 0.05;  white_h = card_h - strip_h
        _fs_n = max(4.5, 7.5 - n * 0.22);  _fs_d = max(3.5, 5.8 - n * 0.18)
        _fs_o = max(5.5, 7.5 - n * 0.2)

        ax.set_facecolor("#F0F0F0")
        for ri in range(n):
            ci = col4r[ri];  cx_p = ci*col_w + col_w/2;  x0_p = cx_p - sw/2
            sh = _sh(ri);  fc, ec = _gc(ri)
            ax.add_patch(Rectangle((x0_p, 0), sw, sh, facecolor=fc,
                                    edgecolor=ec, linewidth=1.5, zorder=3))
            ax.add_patch(Rectangle((x0_p, sh-0.022), sw, 0.022,
                                    facecolor="white", alpha=0.22, linewidth=0, zorder=4))
            ax.text(cx_p, sh*0.40, str(ri+1), ha="center", va="center",
                    fontsize=28, fontweight="bold", color="white", alpha=0.28, zorder=4)
            cy = sh + cgap
            ax.add_patch(Rectangle((x0_p, cy), sw, card_h, facecolor="white",
                                    edgecolor=ec, linewidth=2.0, zorder=5))
            ax.add_patch(Rectangle((x0_p, cy+white_h), sw, strip_h,
                                    facecolor=fc, linewidth=0, zorder=6))
            ax.text(cx_p, cy+white_h+strip_h/2, ORDINALS[ri],
                    ha="center", va="center", fontsize=_fs_o,
                    fontweight="bold", color="white", zorder=7)
            name = str(df2.iloc[ri][x]) if x in df2.columns else f"#{ri+1}"
            if len(name) > 30:
                name = name[:29] + "…"
            nw = "\n".join(textwrap.wrap(name, 12)[:2]) or name
            details = []
            if score_col and score_col in df2.columns:
                sv = df2.iloc[ri]["_score"]
                try:
                    sv_s = f"{sv:,.0f}" if float(sv)==int(float(sv)) else f"{sv:.4g}"
                except Exception:
                    sv_s = str(sv)
                details.append(f"{score_col[:12]}: {sv_s}")
            if g and g in df2.columns:
                details.append(str(df2.iloc[ri][g])[:18])
            ny = cy + white_h * (0.68 if details else 0.50)
            ax.text(cx_p, ny, nw, ha="center", va="center",
                    fontsize=_fs_n, fontweight="bold", color="#111111",
                    zorder=7, multialignment="center", clip_on=True)
            if details:
                ax.text(cx_p, cy+white_h*0.18, "\n".join(details),
                        ha="center", va="center", fontsize=_fs_d,
                        color="#555555", style="italic", zorder=7,
                        multialignment="center", clip_on=True)
            if ri == 0:
                ax.text(cx_p, cy+card_h+0.025, "★",
                        ha="center", va="bottom", fontsize=16,
                        color="#D4AF37", fontweight="bold", zorder=7)
        ax.add_patch(Rectangle((-0.07, -base_h), n*col_w+0.14, base_h,
                                facecolor="#2A2A2A", edgecolor="#111111",
                                linewidth=1, zorder=2))
        ax.set_xlim(-0.2, n*col_w+0.2)
        ax.set_ylim(-base_h-0.03, 1.0+card_h+cgap+0.22)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values(): sp.set_visible(False)

    # ── VIOLINO ───────────────────────────────────────────────────────────────
    elif chart_key == "violin":
        if not y:
            raise ValueError("Seleziona Asse Y (valori numerici) per il violino.")
        if x:
            cats      = sorted(df_sub[x].dropna().unique(), key=_xkey)
            if sort_y_desc:
                cats = _sort_cats_desc(df_sub, x, cats, y, agg="median")
            cc        = _cat_colors(cats, palette)
            data_sets = [pd.to_numeric(df_sub.loc[df_sub[x]==c, y],
                                       errors="coerce").dropna().values for c in cats]
            labels    = [str(c) for c in cats]
            clrs      = [cc[c] for c in cats]
        elif g:
            groups    = sorted(df_sub[g].dropna().unique(), key=_xkey)
            if sort_y_desc:
                groups = _sort_cats_desc(df_sub, g, groups, y, agg="median")
            cc        = _cat_colors(groups, palette)
            data_sets = [pd.to_numeric(df_sub.loc[df_sub[g]==gr, y],
                                       errors="coerce").dropna().values for gr in groups]
            labels    = [str(gr) for gr in groups]
            clrs      = [cc[gr] for gr in groups]
        else:
            data_sets = [pd.to_numeric(df_sub[y], errors="coerce").dropna().values]
            labels    = [str(y)];  clrs = [_colors(3, palette)[1]]
        valid = [(d, l, c) for d, l, c in zip(data_sets, labels, clrs) if len(d) >= 2]
        if not valid:
            raise ValueError("Dati insufficienti per il violino (min 2 valori per categoria).")
        data_arr, labels, clrs = zip(*valid)
        positions = list(range(len(data_arr)))
        vp = ax.violinplot(list(data_arr), positions=positions,
                           showmedians=True, showextrema=True, widths=0.7)
        for body, c in zip(vp["bodies"], clrs):
            body.set_facecolor(c);  body.set_edgecolor(c);  body.set_alpha(0.65)
        for part in ("cmedians", "cmins", "cmaxes", "cbars"):
            if part in vp:
                vp[part].set_color("#444");  vp[part].set_linewidth(1.2)
        ax.set_xticks(positions)
        _smart_labels(ax, list(labels), xtick_rot)

    # ── BARRE + LINEE (COMBO) ─────────────────────────────────────────────────
    elif chart_key == "combo":
        if not x:
            raise ValueError("Combo richiede Asse X.")

        if g and y and not is_freq:
            # ── Barre raggruppate per g ────────────────────────────────────────
            cats   = sorted(df_sub[x].dropna().unique(), key=_xkey)
            if sort_y_desc:
                cats = _sort_cats_desc(df_sub, x, cats, y)
            groups = sorted(df_sub[g].dropna().unique(), key=_xkey)
            cc     = _cat_colors(groups, palette)
            agg_d  = _agg(df_sub, x, y, g)
            pivot  = agg_d.pivot_table(index=x, columns=g, values=y,
                                       aggfunc="sum").reindex(cats)
            idx    = _grouped_bar(ax, pivot, cc, horiz=False, alpha=0.82)
            ax.set_xlabel(x or ""); ax.set_ylabel(y or "")
            _smart_labels(ax, idx, xtick_rot)
            if show_nums:
                for cnt in ax.containers:
                    ax.bar_label(cnt, fmt="%.4g", fontsize=data_lbl_sz, padding=2)
            x_idx = {str(c): i for i, c in enumerate(idx)}
        else:
            # ── Barre semplici ─────────────────────────────────────────────────
            agg_df, vcol = _agg_by_x(df_sub, x, y, sort_y_desc=sort_y_desc)
            vals   = pd.to_numeric(agg_df[vcol], errors="coerce").fillna(0)
            colors = _colors(len(agg_df), palette)
            ax.bar(range(len(agg_df)), vals, color=colors, alpha=0.82)
            ax.set_xticks(range(len(agg_df)))
            _smart_labels(ax, agg_df[x].astype(str), xtick_rot)
            if show_nums and ax.containers:
                ax.bar_label(ax.containers[0], fmt="%.4g", fontsize=data_lbl_sz, padding=2)
            x_idx = {str(v): i for i, v in enumerate(agg_df[x])}

        # ── Linea su asse Y2 separato ──────────────────────────────────────────
        if line_col and line_col in df_sub.columns:
            agg_l, lcol = _agg_by_x(df_sub, x, line_col)
            lvals = pd.to_numeric(agg_l[lcol], errors="coerce").fillna(0)
            lx    = [x_idx.get(str(v)) for v in agg_l[x]]
            pairs = [(xi, yv) for xi, yv in zip(lx, lvals) if xi is not None]
            if pairs:
                px, py = zip(*pairs)
                ax2 = ax.twinx()
                ax2.plot(list(px), list(py), color="#E63946", linewidth=2.2,
                         marker="o", markersize=5, zorder=5)
                ax2.set_ylabel(str(line_col), fontsize=9, color="#E63946")
                ax2.tick_params(axis="y", labelcolor="#E63946")
                # scala Y2 esplicita basata su min/max della linea
                lv_min, lv_max = float(lvals.min()), float(lvals.max())
                pad = (lv_max - lv_min) * 0.12 or abs(lv_min) * 0.1 or 1.0
                ax2.set_ylim(lv_min - pad, lv_max + pad)

    # ── TREEMAP ───────────────────────────────────────────────────────────────
    elif chart_key == "treemap":
        if not x:
            raise ValueError("Treemap richiede Asse X (etichette / categorie).")
        import matplotlib.patches as mpatches
        import math as _math

        treemap_style = ex.get("treemap_style", "Classica")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

        def _shade(col_hex, f):
            r_, g_, b_, a_ = mcolors.to_rgba(col_hex)
            return (min(1.0, r_ * f), min(1.0, g_ * f), min(1.0, b_ * f), a_)

        def _lum_txt(col_hex):
            r_, g_, b_, _ = mcolors.to_rgba(col_hex)
            return "white" if 0.299 * r_ + 0.587 * g_ + 0.114 * b_ < 0.55 else "#222"

        def _clip_to_box(txt, x0, y0, w, h):
            # clip_on=True da solo ritaglia solo ai bordi dell'INTERO asse,
            # non al singolo riquadro del treemap — un'etichetta più larga
            # della sua cella continuava a "uscire" sopra le celle vicine.
            # Qui la ritagliamo esplicitamente al rettangolo che le
            # appartiene.
            txt.set_clip_path(mpatches.Rectangle((x0, y0), w, h, transform=ax.transData))
            return txt

        _fig_w_in = ax.get_figure().get_size_inches()[0]
        _axes_w_frac = ax.get_position().width

        def _fit_text(label, box_w_frac, fontsize_pt):
            # Il ritaglio (_clip_to_box) da solo non basta: con testo SCURO
            # (vedi _lum_txt) il clip path smette di funzionare — bug di
            # matplotlib/Agg riprodotto isolatamente (stessa identica forma,
            # stesso identico clip, cambia solo il colore del testo da
            # bianco a scuro e il ritaglio smette di applicarsi). Soluzione
            # robusta che non dipende da quel meccanismo: accorciamo il
            # testo PRIMA di disegnarlo, così non deve mai essere ritagliato
            # per stare nel riquadro — box_w_frac è la larghezza del
            # riquadro in coordinate dati (0..1, essendo xlim (0,1)),
            # convertita in punti reali via le dimensioni vere della figura.
            box_w_pt = box_w_frac * _axes_w_frac * _fig_w_in * 72
            # 0.66 misurato direttamente (matplotlib get_window_extent) sul
            # font bold di default a questa dimensione — 0.58 lasciava un
            # margine residuo che faceva ancora sporgere leggermente il
            # testo più lungo.
            max_chars = max(1, int(box_w_pt / (0.66 * fontsize_pt)))
            if len(label) <= max_chars:
                return label
            if max_chars <= 1:
                return "…"
            return label[:max_chars - 1].rstrip() + "…"

        # ── Classica: squarify ────────────────────────────────────────────────
        if treemap_style == "Classica":
            if g and not is_freq:
                # Due livelli: x = categorie principali, g = sottocategorie
                main_s = (df_sub.groupby(x)[y].sum()
                          if (y and y != _FREQ_LABEL) else df_sub.groupby(x).size())
                main_s = main_s[main_s > 0].sort_values(ascending=False)
                if main_s.empty:
                    raise ValueError("Nessun valore positivo.")
                main_total = float(main_s.sum())
                main_cats  = list(main_s.index)
                main_cc    = _cat_colors(main_cats, palette)
                main_items = [(v / main_total, cat, main_cc[cat])
                              for cat, v in main_s.items()]
                main_rects = []
                _squarify_items(main_items, 0, 0, 1, 1, main_rects)
                for rx, ry, rdx, rdy, cat, col in main_rects:
                    sub = df_sub[df_sub[x] == cat]
                    sub_s = (sub.groupby(g)[y].sum()
                             if (y and y != _FREQ_LABEL) else sub.groupby(g).size())
                    sub_s = sub_s[sub_s > 0].sort_values(ascending=False)
                    if sub_s.empty:
                        ax.add_patch(mpatches.Rectangle(
                            (rx + .002, ry + .002), rdx - .004, rdy - .004,
                            facecolor=col, edgecolor="white", linewidth=1.5, zorder=2,
                            transform=ax.transData))
                    else:
                        m = 0.004
                        st_ = float(sub_s.sum())
                        n_s = len(sub_s)
                        sub_colors = [_shade(col, 0.55 + 0.45 * i / max(n_s - 1, 1))
                                      for i in range(n_s)]
                        sub_items = [
                            (float(v) / st_ * (rdx - 2 * m) * (rdy - 2 * m),
                             str(sc), sub_colors[i])
                            for i, (sc, v) in enumerate(sub_s.items())
                        ]
                        sub_rects = []
                        _squarify_items(sub_items, rx + m, ry + m,
                                        rdx - 2 * m, rdy - 2 * m, sub_rects)
                        for srx, sry, srdx, srdy, scat, scol in sub_rects:
                            ax.add_patch(mpatches.Rectangle(
                                (srx + .001, sry + .001), srdx - .002, srdy - .002,
                                facecolor=scol, edgecolor="white", linewidth=.6, zorder=3,
                                transform=ax.transData))
                            if srdx * srdy > 0.003:
                                _sfs = max(4, min(7, srdx * 55))
                                _clip_to_box(ax.text(srx + srdx / 2, sry + srdy / 2,
                                        _fit_text(scat, srdx, _sfs),
                                        ha="center", va="center", fontsize=_sfs,
                                        color=_lum_txt(scol), fontweight="bold",
                                        zorder=4, clip_on=True), srx, sry, srdx, srdy)
                    ax.add_patch(mpatches.Rectangle(
                        (rx + .001, ry + .001), rdx - .002, rdy - .002,
                        facecolor="none", edgecolor=col, linewidth=2.5, zorder=5,
                        transform=ax.transData))
                    if rdx > .04 and rdy > .04:
                        _mfs = max(5, min(10, rdx * 22))
                        _clip_to_box(ax.text(rx + .008, ry + rdy - .008,
                                _fit_text(str(cat), rdx - .016, _mfs),
                                ha="left", va="top", fontsize=_mfs,
                                color=_lum_txt(col), fontweight="bold", zorder=6,
                                clip_on=True,
                                bbox=dict(boxstyle="round,pad=0.1", facecolor=col,
                                          alpha=0.75, linewidth=0)), rx, ry, rdx, rdy)
            else:
                # Livello singolo
                agg_df, vcol = _agg_by_x(df_sub, x, y)
                vals_num = pd.to_numeric(agg_df[vcol], errors="coerce").fillna(0)
                agg_df   = agg_df[vals_num > 0].copy().reset_index(drop=True)
                vals_num = pd.to_numeric(agg_df[vcol], errors="coerce").fillna(0)
                if agg_df.empty:
                    raise ValueError("Nessun valore positivo per il treemap.")
                total   = float(vals_num.sum())
                colors  = _colors(len(agg_df), palette)
                val_map = {str(agg_df.iloc[i][x]): float(vals_num.iloc[i])
                           for i in range(len(agg_df))}
                items   = [(float(vals_num.iloc[i]) / total,
                            str(agg_df.iloc[i][x]), colors[i])
                           for i in range(len(agg_df))]
                rects   = []
                _squarify_items(items, 0, 0, 1, 1, rects)
                for rx, ry, rdx, rdy, lbl, col in rects:
                    ax.add_patch(mpatches.Rectangle(
                        (rx + .002, ry + .002), rdx - .004, rdy - .004,
                        facecolor=col, edgecolor="white", linewidth=1.5, zorder=2,
                        transform=ax.transData))
                    if rdx * rdy > 0.005:
                        val_v = val_map.get(lbl, 0)
                        _sfs = max(5, min(11, rdx * rdy * 130))
                        _fit_lbl = _fit_text(lbl, rdx, _sfs)
                        lbl_txt = (f"{_fit_lbl}\n{val_v:,.4g}"
                                   if rdx * rdy > 0.015 else _fit_lbl)
                        _clip_to_box(ax.text(rx + rdx / 2, ry + rdy / 2, lbl_txt,
                                ha="center", va="center", fontsize=_sfs,
                                color=_lum_txt(col), fontweight="bold",
                                zorder=3, clip_on=True), rx, ry, rdx, rdy)

        # ── Albero: icicle / partition chart ─────────────────────────────────
        elif treemap_style == "Albero":
            # Asse temporaneo su [0,1]×[0,1]
            # Prima riga (alto): categorie X, larghezza ∝ valore totale
            # Seconda riga (basso): sottocategorie G, larghezza ∝ valore interno

            if y and y != _FREQ_LABEL:
                root_s = df_sub.groupby(x)[y].sum()
            else:
                root_s = df_sub.groupby(x).size()
            root_s = root_s[root_s > 0].sort_values(ascending=False)
            if root_s.empty:
                raise ValueError("Nessun valore positivo.")
            root_total = float(root_s.sum())
            root_cats  = list(root_s.index)
            root_cc    = _cat_colors(root_cats, palette)

            has_sub = bool(g and not is_freq)
            row_h   = 0.38 if has_sub else 0.6   # altezza barra categoria
            row_y   = 0.62 if has_sub else 0.2   # y bassa della barra categoria

            cx = 0.0
            for cat in root_cats:
                v   = float(root_s[cat])
                cw  = v / root_total
                col = root_cc[cat]
                ax.add_patch(mpatches.FancyBboxPatch(
                    (cx + .003, row_y + .003), cw - .006, row_h - .006,
                    boxstyle="round,pad=0.005",
                    facecolor=col, edgecolor="white", linewidth=1.5, zorder=2,
                    transform=ax.transData))
                if cw > .04:
                    _afs = max(5, min(11, cw * 30))
                    _clip_to_box(ax.text(cx + cw / 2, row_y + row_h / 2,
                            _fit_text(str(cat), cw, _afs),
                            ha="center", va="center", fontsize=_afs,
                            color=_lum_txt(col), fontweight="bold",
                            zorder=3, clip_on=True), cx, row_y, cw, row_h)
                if has_sub:
                    sub = df_sub[df_sub[x] == cat]
                    sub_s = (sub.groupby(g)[y].sum()
                             if (y and y != _FREQ_LABEL) else sub.groupby(g).size())
                    sub_s = sub_s[sub_s > 0].sort_values(ascending=False)
                    sub_total = float(sub_s.sum()) if not sub_s.empty else 1
                    sx = cx
                    for sc, sv in sub_s.items():
                        sw  = (sv / sub_total) * cw
                        scl = _shade(col, 0.6 + 0.4 * list(sub_s.index).index(sc) /
                                     max(len(sub_s) - 1, 1))
                        ax.add_patch(mpatches.FancyBboxPatch(
                            (sx + .003, .05 + .003), sw - .006, .30 - .006,
                            boxstyle="round,pad=0.005",
                            facecolor=scl, edgecolor="white", linewidth=.8,
                            zorder=2, transform=ax.transData))
                        if sw > .03:
                            _bfs = max(4, min(8, sw * 28))
                            _clip_to_box(ax.text(sx + sw / 2, .05 + .15,
                                    _fit_text(str(sc), sw, _bfs),
                                    ha="center", va="center", fontsize=_bfs,
                                    color=_lum_txt(scl), fontweight="bold",
                                    zorder=3, clip_on=True), sx, .05, sw, .30)
                        # Linea di collegamento cat → subcat
                        ax.plot([sx + sw / 2, cx + cw / 2],
                                [.05 + .30 + .003, row_y - .003],
                                color=col, linewidth=.8, alpha=.5,
                                zorder=1, transform=ax.transData)
                        sx += sw
                cx += cw
            # Etichette assi informativi
            ax.text(0.5, 0.98, str(x),
                    ha="center", va="top", fontsize=9, color="#555",
                    transform=ax.transData)
            if has_sub:
                ax.text(0.5, 0.02, str(g),
                        ha="center", va="bottom", fontsize=9, color="#555",
                        transform=ax.transData)

        # ── Spirale: cerchi filogenetici phyllotaxis ──────────────────────────
        elif treemap_style == "Spirale":
            agg_df, vcol = _agg_by_x(df_sub, x, y)
            vals_num = pd.to_numeric(agg_df[vcol], errors="coerce").fillna(0)
            agg_df   = agg_df[vals_num > 0].copy().reset_index(drop=True)
            vals_num = pd.to_numeric(agg_df[vcol], errors="coerce").fillna(0)
            if agg_df.empty:
                raise ValueError("Nessun valore positivo.")
            n       = len(agg_df)
            v_max   = float(vals_num.max())
            colors  = _colors(n, palette)
            GOLDEN  = 2.399963  # angolo d'oro in radianti (≈ 137.51°)
            # Scala del raggio: l'ultimo cerchio deve stare entro r=0.48
            r_scale = 0.48 / (_math.sqrt(n) if n > 1 else 1)
            r_base  = r_scale * 0.85  # raggio massimo cerchio (per valore massimo)
            ax.set_aspect("equal")
            for i in range(n):
                vi   = float(vals_num.iloc[i])
                lbl  = str(agg_df.iloc[i][x])
                col  = colors[i]
                cr   = r_base * _math.sqrt(vi / v_max)  # raggio ∝ sqrt(valore)
                r_i  = r_scale * _math.sqrt(i + 0.5)    # distanza dal centro
                th_i = i * GOLDEN
                cx_i = 0.5 + r_i * _math.cos(th_i)
                cy_i = 0.5 + r_i * _math.sin(th_i)
                circ = mpatches.Circle((cx_i, cy_i), cr,
                                       facecolor=col, edgecolor="white",
                                       linewidth=.8, zorder=2,
                                       transform=ax.transData)
                ax.add_patch(circ)
                if cr > 0.025:
                    fsize = max(4, min(9, cr * 55))
                    _lbl_txt = ax.text(cx_i, cy_i, _fit_text(lbl, 2 * cr * 0.92, fsize),
                            ha="center", va="center", fontsize=fsize,
                            color=_lum_txt(col), fontweight="bold",
                            zorder=3, clip_on=True)
                    _lbl_txt.set_clip_path(mpatches.Circle((cx_i, cy_i), cr, transform=ax.transData))

    else:
        raise ValueError(f"Tipo grafico non supportato: {chart_key!r}")

    return ax


# ══════════════════════════════════════════════════════════════════════════════
# GRAFICI INTERATTIVI (Plotly) — Fase 1: tipi "standard" con legenda/titolo
# trascinabili col mouse (config editable=True in st.plotly_chart). Mirror
# funzionale di _render_chart_on_ax/_apply_style/_apply_axis_settings sopra:
# stessa preparazione dati (_agg/_agg_by_x/_sort_cats_desc/_colors sono pure
# funzioni pandas/colore, riusate identiche), cambia solo il motore di
# disegno. I tipi NON ancora qui (bar_stacked 3D, radar, funnel, podio,
# treemap, combo, heatmap, mappe) restano su matplotlib — vedi
# _PLOTLY_CHART_KEYS più sotto, che decide quale motore usare.
# ══════════════════════════════════════════════════════════════════════════════

_PLOTLY_CHART_KEYS = {"bar", "bar_stacked", "line", "scatter", "histogram",
                       "pie", "boxplot", "violin", "radar", "funnel"}

_LEGEND_POSITIONS_PLOTLY = {
    "Auto (migliore)":  {"x": 1.02, "y": 1.0,  "xanchor": "left",   "yanchor": "top"},
    "Alto a destra":    {"x": 0.99, "y": 0.99, "xanchor": "right",  "yanchor": "top"},
    "Alto a sinistra":  {"x": 0.01, "y": 0.99, "xanchor": "left",   "yanchor": "top"},
    "Basso a destra":   {"x": 0.99, "y": 0.01, "xanchor": "right",  "yanchor": "bottom"},
    "Basso a sinistra": {"x": 0.01, "y": 0.01, "xanchor": "left",   "yanchor": "bottom"},
    "Fuori a destra":   {"x": 1.02, "y": 0.5,  "xanchor": "left",   "yanchor": "middle"},
    "Fuori in alto":    {"x": 0.5,  "y": 1.12, "xanchor": "center", "yanchor": "bottom"},
    "Fuori in basso":   {"x": 0.5,  "y": -0.15,"xanchor": "center", "yanchor": "top"},
}


def _to_css_color(c):
    """Converte un colore matplotlib (tupla RGBA float 0-1, dai colormap) o
    una stringa hex/CSS già pronta (dalle palette brand) nel formato CSS che
    Plotly si aspetta — _colors()/_cat_colors() restano invariate e servono
    a entrambi i motori di disegno."""
    if isinstance(c, str):
        return c
    r, g, b = c[0], c[1], c[2]
    a = c[3] if len(c) > 3 else 1.0
    return f"rgba({int(r*255)},{int(g*255)},{int(b*255)},{a})"


def _apply_style_plotly(fig, title, xlabel, ylabel, show_legend, show_grid, opts=None):
    """Mirror di _apply_style per go.Figure (Plotly) — stessi 'opts' del
    motore matplotlib, riusati senza modifiche dal chiamante."""
    o = opts or {}
    title_sz   = o.get("title_sz", 13)
    title_bold = o.get("title_bold", True)
    title_ital = o.get("title_ital", False)
    label_sz   = o.get("label_sz", 10)

    title_html = None
    if title:
        weight = "bold" if title_bold else "normal"
        style  = "italic" if title_ital else "normal"
        _TITLE_X = {"Sinistra": (0.02, "left"), "Centro": (0.5, "center"), "Destra": (0.98, "right")}
        _tx, _txanchor = _TITLE_X.get(o.get("title_loc", "Centro"), _TITLE_X["Centro"])
        title_html = dict(text=f"<span style='font-weight:{weight};font-style:{style}'>{title}</span>",
                           font=dict(size=title_sz), x=_tx, xanchor=_txanchor,
                           y=min(0.99, max(0.7, 0.95 + float(o.get("title_offset_y", 0.0)))),
                           yanchor="top")

    pos_key = o.get("legend_loc", "Auto (migliore)")
    pos     = _LEGEND_POSITIONS_PLOTLY.get(pos_key, _LEGEND_POSITIONS_PLOTLY["Auto (migliore)"])
    legend_sz = int(o.get("legend_sz", 8))
    legend_cfg = dict(pos)
    legend_cfg["x"] = legend_cfg["x"] + float(o.get("legend_offset_x", 0.0))
    legend_cfg["y"] = legend_cfg["y"] + float(o.get("legend_offset_y", 0.0))
    legend_cfg["font"] = dict(size=legend_sz)
    # Il riquadro colore della legenda non si ingrandisce da solo col font
    # (sono due propretà separate in Plotly) — lo si scala qui in proporzione
    # perché "ingrandire la legenda" intuitivamente vuol dire tutto insieme,
    # testo e simbolo colore, non solo il testo.
    legend_cfg["itemwidth"] = max(30, int(legend_sz * 3.2))

    # Dimensioni: senza queste il grafico usa l'altezza di default di Plotly
    # (~450px) mentre use_container_width allarga solo la larghezza — risultato
    # un grafico molto largo e basso. Qui si replica il rapporto larghezza/
    # altezza configurato in "Esportazione" (stesso _fw/_fh usati prima da
    # matplotlib figsize, convertiti in pixel).
    fig_w = float(o.get("fig_w", 11.0)) * 96
    fig_h = float(o.get("fig_h", 6.0)) * 96
    # Margini generosi: l'export statico (kaleido) NON ritaglia il canvas per
    # far stare etichette lunghe/ruotate come faceva bbox_inches="tight" di
    # matplotlib — senza margine esplicito le etichette dell'asse X restano
    # tagliate fuori dal PNG/SVG scaricato.
    # Sfondo e palette-per-sfondo sono DUE scelte indipendenti (non un unico
    # selettore a 3 stati): "Aggiungi sfondo" decide SOLO se il pannello è
    # trasparente (default — MAI il grigio-azzurro di default del template
    # "plotly", il motivo principale per cui i grafici sembravano meno
    # "eleganti" di un tema minimale) o riempito con un colore a scelta
    # (color picker) — "Palette per sfondo" decide SOLO se testi/assi/
    # palette (quest'ultima schiarita già da _colors()) sono tarati per
    # stare su chiaro o su scuro. Restano indipendenti perché un grafico può
    # restare trasparente qui e finire composto su uno sfondo scuro altrove
    # (slide, video) — la palette va comunque schiarita anche senza un
    # riempimento scuro visibile in anteprima.
    dark_text = o.get("palette_mode") == "Scuro"
    if o.get("bg_on", False):
        bg_fill = o.get("bg_color") or "white"
    else:
        bg_fill = "rgba(0,0,0,0)"
    if dark_text:
        text_col, spine_col, tick_col, grid_col = (
            "#E8E6EF", "#4A4468", "#B8B2C8", "rgba(255,255,255,0.10)")
    else:
        text_col, spine_col, tick_col, grid_col = (
            "#2B2B33", "#D7D7DE", "#6B6B76", "rgba(0,0,0,0.08)")

    fig.update_layout(
        title=title_html,
        font=dict(size=label_sz, color=text_col),
        showlegend=bool(show_legend),
        legend=legend_cfg,
        plot_bgcolor=bg_fill,
        paper_bgcolor=bg_fill,
        width=int(fig_w), height=int(fig_h),
        margin=dict(l=90, r=40, t=60 if title else 30, b=110),
    )

    # Un grafico polare (radar) non ha assi cartesiani x/y — toccarli
    # (fig.update_xaxes/update_yaxes) creerebbe una coppia di assi vuoti e
    # inutilizzati sovrapposti al grafico polare. Stessa gerarchia di stile
    # (sfondo/griglia/testi), ma sugli equivalenti polari.
    is_polar = bool(fig.data) and fig.data[0].type == "scatterpolar"
    if is_polar:
        fig.update_layout(polar=dict(
            bgcolor=bg_fill,
            radialaxis=dict(gridcolor=grid_col if show_grid else "rgba(0,0,0,0)",
                            linecolor=spine_col, tickfont=dict(color=tick_col)),
            angularaxis=dict(gridcolor=grid_col if show_grid else "rgba(0,0,0,0)",
                             linecolor=spine_col, tickfont=dict(color=tick_col)),
        ))
        for _ann in (o.get("extra_texts") or []):
            fig.add_annotation(text=str(_ann.get("text", "")), x=float(_ann.get("x", 0.5)),
                                y=float(_ann.get("y", 0.9)), xref="paper", yref="paper",
                                showarrow=False, font=dict(size=int(_ann.get("size", 12))))
        return

    # Assi sottili e neutri invece del grigio più marcato di default di
    # Plotly, ed etichette dei tick in un grigio (o lavanda chiara, su sfondo
    # scuro) attenuato invece che nere: stessa gerarchia "titolo in risalto,
    # resto discreto" di un tema editoriale minimale — il dato deve
    # risaltare, non la cornice.
    fig.update_xaxes(showline=True, linecolor=spine_col, linewidth=1,
                      ticks="outside", tickcolor=spine_col, tickfont=dict(color=tick_col))
    fig.update_yaxes(showline=True, linecolor=spine_col, linewidth=1,
                      ticks="outside", tickcolor=spine_col, tickfont=dict(color=tick_col))
    # automargin: i margini fissi sopra sono solo un punto di partenza — per
    # etichette molto lunghe (categorie con nomi lunghi, ruotate) non
    # bastano mai in ogni caso. automargin fa espandere l'area del grafico
    # in automatico finché testo/titoli non ci stanno più dentro, anche
    # nell'export statico via kaleido — è l'equivalente reale di
    # bbox_inches="tight" di matplotlib, i margini fissi sopra no.
    fig.update_xaxes(automargin=True)
    fig.update_yaxes(automargin=True)
    # A differenza di xaxis_title/yaxis_title in update_layout (che
    # sovrascriverebbe SEMPRE, anche con None), qui si aggiorna il titolo
    # asse SOLO se l'utente ha scritto qualcosa — altrimenti resta il nome
    # colonna già impostato da _render_chart_plotly come default automatico.
    if xlabel:
        fig.update_xaxes(title_text=xlabel)
    if ylabel:
        fig.update_yaxes(title_text=ylabel)

    # Scostamento 2D delle etichette assi: Plotly non offre una posizione
    # libera per xaxis.title/yaxis.title (solo 'standoff', una distanza
    # perpendicolare, non un vero x/y) — a scostamento diverso da zero si
    # nasconde il titolo nativo e lo si ridisegna come annotazione libera in
    # coordinate "paper" (0-1 sull'intera figura), che invece supporta
    # qualunque x/y — stesso meccanismo già usato per i "Testi liberi".
    _xox, _xoy = float(o.get("xlabel_offset_x", 0.0)), float(o.get("xlabel_offset_y", 0.0))
    if (_xox or _xoy) and fig.layout.xaxis and fig.layout.xaxis.title and fig.layout.xaxis.title.text:
        _xtext = fig.layout.xaxis.title.text
        fig.update_xaxes(title_text="")
        fig.add_annotation(text=_xtext, x=0.5 + _xox, y=-0.14 + _xoy,
                           xref="paper", yref="paper", showarrow=False,
                           font=dict(size=label_sz, color=text_col))
    _yox, _yoy = float(o.get("ylabel_offset_x", 0.0)), float(o.get("ylabel_offset_y", 0.0))
    if (_yox or _yoy) and fig.layout.yaxis and fig.layout.yaxis.title and fig.layout.yaxis.title.text:
        _ytext = fig.layout.yaxis.title.text
        fig.update_yaxes(title_text="")
        fig.add_annotation(text=_ytext, x=-0.09 + _yox, y=0.5 + _yoy,
                           xref="paper", yref="paper", showarrow=False, textangle=-90,
                           font=dict(size=label_sz, color=text_col))

    # BUG corretto: la griglia veniva disattivata solo sull'asse Y — quella
    # verticale (asse X) restava sempre al comportamento di default di
    # Plotly, mai controllata dal toggle "Griglia", che l'utente si aspetta
    # controlli entrambe le direzioni insieme.
    fig.update_yaxes(showgrid=bool(show_grid), gridcolor=grid_col)
    fig.update_xaxes(showgrid=bool(show_grid), gridcolor=grid_col)

    y_min_s, y_max_s = o.get("y_min", ""), o.get("y_max", "")
    if y_min_s and y_max_s:
        try:
            fig.update_yaxes(range=[float(y_min_s), float(y_max_s)])
        except ValueError:
            pass

    if not o.get("show_axes_y", True):
        fig.update_yaxes(visible=False)
    if not o.get("show_axes_x", True):
        fig.update_xaxes(visible=False)

    # Testi liberi aggiunti dall'utente (tab Personalizzazione → "Testi
    # aggiuntivi") — posizionati in coordinate "paper" (0-1 sull'intera area
    # del grafico, indipendenti dai dati) così restano dove li ha messi
    # l'utente qualunque siano i dati. A differenza del trascinamento col
    # mouse, questi SONO salvati in PNG/SVG/export perché vivono in
    # st.session_state, non solo nel browser.
    for _ann in (o.get("extra_texts") or []):
        fig.add_annotation(text=str(_ann.get("text", "")), x=float(_ann.get("x", 0.5)),
                            y=float(_ann.get("y", 0.9)), xref="paper", yref="paper",
                            showarrow=False, font=dict(size=int(_ann.get("size", 12))))


def _apply_axis_settings_plotly(fig, opts, horiz=False):
    """Mirror parziale di _apply_axis_settings per go.Figure: scala log e
    minor tick di griglia. Nota di parità (Fase 1): la scala 'Sqrt' non ha
    equivalente nativo in Plotly (solo linear/log/date/category) — resta
    lineare invece di dare errore. Le opzioni di griglia avanzate (stile
    tratteggio, minor tick multipli) non sono ancora portate, solo
    griglia on/off (via _apply_style_plotly).

    horiz: come per il motore matplotlib (dove la scala si applica SOLO
    all'asse che non ha un FixedLocator, cioè quello categoriale), con le
    barre orizzontali l'asse dei VALORI diventa quello X — forzare "type=log"
    sull'asse Y (categorico in quel caso) produrrebbe tick numerici casuali
    al posto delle etichette, perché Plotly ha già inferito quell'asse come
    categoriale dai dati testuali."""
    o = opts or {}
    x_is_log = str(o.get("x_scale", "Normale")) in ("Log10", "Log₂")
    y_is_log = str(o.get("y_scale", "Normale")) in ("Log10", "Log₂")
    if horiz:
        x_is_log, y_is_log = y_is_log, False
    # BUG corretto: senza dtick esplicito, per un range dati che copre poco
    # più di una decade Plotly sceglie in automatico uno step "lineare"
    # (es. 30,40,50...130) invece del classico step per potenza di dieci
    # (1,10,100...) — sembra un asse lineare travestito, non un vero log10.
    # dtick=1 forza UN tick maggiore per decade; i minor tick (le sotto-
    # suddivisioni 2,3,4...9 per decade, mostrate di default da Plotly sugli
    # assi log) vengono disattivati esplicitamente invece di lasciarli al
    # comportamento automatico, che appariva "a scatti"/incoerente a seconda
    # del range dati.
    # Radar (polare): non ha assi cartesiani x/y, ma la scala si applica
    # comunque al RAGGIO (l'equivalente dell'asse Y qui) — stessa cosa che
    # faceva la versione matplotlib, dove il raggio DI FATTO è l'asse "y" di
    # un Axes polare (ax.set_yscale su un asse polare scala il raggio).
    # Perderla è un regresso reale quando i valori delle categorie coprono
    # un range ampio: senza log il poligono schiaccia i valori piccoli
    # vicino al centro.
    if bool(fig.data) and fig.data[0].type == "scatterpolar":
        if y_is_log:
            fig.update_layout(polar=dict(radialaxis=dict(type="log")))
        return
    if x_is_log:
        fig.update_xaxes(type="log", dtick=1, minor=dict(ticks=""))
    if y_is_log:
        fig.update_yaxes(type="log", dtick=1, minor=dict(ticks=""))


def _add_grouped_bars_plotly(fig, pivot_raw, cols, horiz, bar_kwargs, gap=0.15):
    """Barre raggruppate Plotly con la stessa logica di _grouped_bar (matplotlib):
    a ogni valore dell'asse la larghezza si divide solo tra le categorie
    EFFETTIVAMENTE presenti lì, non tra tutte quelle del dataset (con barmode
    'group' Plotly riserva uno slot vuoto anche alle combinazioni mancanti, e le
    barre restano strette). Posizione/larghezza esplicite per ogni barra
    (offset/width, in unità dell'asse a categorie), lette anche dall'animazione
    'acqua' per allineare il livello alla barra giusta.
    `pivot_raw` deve avere NaN (non 0) dove la combinazione non esiste."""
    present = pivot_raw.notna()
    idx_str = [str(v) for v in pivot_raw.index]
    columns = list(pivot_raw.columns)
    slot = 1.0 - gap
    geom = {c: ([0.0] * len(idx_str), [0.0] * len(idx_str)) for c in columns}
    for i in range(len(idx_str)):
        here = [c for c in columns if present.iloc[i][c]]
        w = slot / (len(here) or 1)
        for j, c in enumerate(here):
            center = (j - (len(here) - 1) / 2) * w
            geom[c][0][i] = center - w * 0.45
            geom[c][1][i] = w * 0.9
    for col_name, col_ in zip(columns, cols):
        vals = [None if pd.isna(v) else float(v) for v in pivot_raw[col_name]]
        offsets, widths = geom[col_name]
        kw = dict(name=str(col_name), marker_color=col_, offset=offsets, width=widths,
                  **bar_kwargs(vals))
        if horiz:
            fig.add_trace(go.Bar(y=idx_str, x=vals, orientation="h", **kw))
        else:
            fig.add_trace(go.Bar(x=idx_str, y=vals, **kw))
    fig.update_layout(barmode="overlay")


def _render_chart_plotly(df_sub, chart_key, x, y, g, palette, show_nums,
                          xtick_rot, data_lbl_sz, extra=None):
    """Mirror Plotly di _render_chart_on_ax per i tipi in _PLOTLY_CHART_KEYS.
    Stessa preparazione dati (_agg/_agg_by_x/_sort_cats_desc/_colors/
    _cat_colors/_xkey — invariate, indipendenti dal motore di disegno) e
    stesso significato dei parametri sort_y_desc/highlight_*/show_nums/
    xtick_rot già usati dal motore matplotlib."""
    ex = extra or {}
    horiz           = bool(ex.get("horiz", False))
    show_line       = bool(ex.get("show_line", False))
    sort_y_desc     = bool(ex.get("sort_y_desc", False))
    highlight_set   = {str(v) for v in (ex.get("highlight_cats") or [])}
    highlight_color = ex.get("highlight_color") or "#E63946"
    is_freq         = (y == _FREQ_LABEL or not y)

    fig = go.Figure()

    def _hl(cols_, labels_):
        if not highlight_set:
            return cols_
        return [highlight_color if str(lbl) in highlight_set else c
                for lbl, c in zip(labels_, cols_)]

    def _num_texts(vals):
        # Formattati qui come in matplotlib (%.4g → "3", non "3.000"/"3.0" del
        # formato d3 di Plotly); vuoto per le combinazioni assenti.
        return ["" if v is None or pd.isna(v) else f"{float(v):.4g}" for v in vals]

    def _bar_kwargs(vals):
        kw = {}
        if show_nums:
            kw["text"] = _num_texts(vals)
            kw["texttemplate"] = "%{text}"
            kw["textposition"] = "outside"
        return kw

    if chart_key == "bar":
        if not x:
            raise ValueError("Seleziona almeno Asse X.")
        if not is_freq and g:
            agg   = _agg(df_sub, x, y, g)
            pivot = agg.pivot_table(index=x, columns=g, values=y, aggfunc="sum")
            if sort_y_desc:
                pivot = pivot.loc[pivot.sum(axis=1).sort_values(ascending=False).index]
            cols = [_to_css_color(c) for c in _colors(len(pivot.columns), palette)]
            _add_grouped_bars_plotly(fig, pivot, cols, horiz, _bar_kwargs)
        elif not is_freq:
            agg_df, val_col = _agg_by_x(df_sub, x, y, sort_y_desc=sort_y_desc)
            vals   = pd.to_numeric(agg_df[val_col], errors="coerce").fillna(0)
            labels = agg_df[x].astype(str)
            cols   = _hl([_to_css_color(c) for c in _colors(len(agg_df), palette)], labels)
            kw = dict(marker_color=cols, **_bar_kwargs(vals))
            if horiz:
                fig.add_trace(go.Bar(y=labels, x=vals, orientation="h", **kw))
            else:
                fig.add_trace(go.Bar(x=labels, y=vals, **kw))
        else:
            if g:
                frq   = df_sub.groupby([x, g], sort=False).size().reset_index(name="Frequenza")
                pivot = frq.pivot_table(index=x, columns=g, values="Frequenza")
                if sort_y_desc:
                    pivot = pivot.loc[pivot.sum(axis=1).sort_values(ascending=False).index]
                cols = [_to_css_color(c) for c in _colors(len(pivot.columns), palette)]
                _add_grouped_bars_plotly(fig, pivot, cols, horiz, _bar_kwargs)
            else:
                agg_df, _ = _agg_by_x(df_sub, x, _FREQ_LABEL, sort_y_desc=sort_y_desc)
                vals   = pd.to_numeric(agg_df["Frequenza"], errors="coerce").fillna(0)
                labels = agg_df[x].astype(str)
                cols   = _hl([_to_css_color(c) for c in _colors(len(agg_df), palette)], labels)
                kw = dict(marker_color=cols, **_bar_kwargs(vals))
                if horiz:
                    fig.add_trace(go.Bar(y=labels, x=vals, orientation="h", **kw))
                else:
                    fig.add_trace(go.Bar(x=labels, y=vals, **kw))

    elif chart_key == "bar_stacked":
        if not x:
            raise ValueError("Barre impilate richiedono almeno Asse X.")
        if not (y and g):
            raise ValueError("Barre impilate richiedono Asse X, Asse Y e Raggruppa.")
        agg   = _agg(df_sub, x, y, g)
        pivot = agg.pivot_table(index=x, columns=g, values=y, aggfunc="sum").fillna(0)
        if sort_y_desc:
            pivot = pivot.loc[pivot.sum(axis=1).sort_values(ascending=False).index]
        cols = [_to_css_color(c) for c in _colors(len(pivot.columns), palette)]
        idx_str = [str(v) for v in pivot.index]
        for col_name, col_ in zip(pivot.columns, cols):
            vals = pivot[col_name]
            kw = dict(name=str(col_name), marker_color=col_)
            if show_nums:
                kw.update(text=_num_texts(vals), texttemplate="%{text}", textposition="inside")
            fig.add_trace(go.Bar(x=idx_str, y=vals, **kw))
        fig.update_layout(barmode="stack")

    elif chart_key == "line":
        if not x:
            raise ValueError("Seleziona almeno Asse X.")
        x_all = sorted(df_sub[x].dropna().unique(), key=_xkey)
        if sort_y_desc:
            x_all = _sort_cats_desc(df_sub, x, x_all, y)
        if g:
            groups = sorted(df_sub[g].dropna().unique(), key=_xkey)
            cols = [_to_css_color(c) for c in _colors(len(groups), palette)]
            for grp, col_ in zip(groups, cols):
                sub = df_sub[df_sub[g] == grp]
                agg_sub, vcol = _agg_by_x(sub, x, y)
                fig.add_trace(go.Scatter(
                    x=agg_sub[x].astype(str), y=pd.to_numeric(agg_sub[vcol], errors="coerce"),
                    mode="lines+markers", name=str(grp),
                    line=dict(color=col_), marker=dict(color=col_)))
        else:
            agg_df, vcol = _agg_by_x(df_sub, x, y)
            col_ = _to_css_color(_colors(3, palette)[1])
            fig.add_trace(go.Scatter(
                x=agg_df[x].astype(str), y=pd.to_numeric(agg_df[vcol], errors="coerce"),
                mode="lines+markers", line=dict(color=col_), marker=dict(color=col_)))
        # Plotly ordina l'asse categoriale secondo l'array indicato — TUTTE le
        # tracce lo rispettano automaticamente (a differenza di matplotlib,
        # dove riordinare l'asse senza riordinare anche i punti disegnati
        # aveva causato linee che si incrociavano nell'ordine sbagliato).
        fig.update_xaxes(type="category", categoryorder="array",
                          categoryarray=[str(v) for v in x_all])

    elif chart_key == "scatter":
        if not (x and y):
            raise ValueError("Seleziona Asse X e Asse Y numerici.")
        xv = pd.to_numeric(df_sub[x], errors="coerce")
        yv = pd.to_numeric(df_sub[y], errors="coerce")
        mode = "markers+lines" if show_line else "markers"
        if g:
            groups = sorted(df_sub[g].dropna().unique(), key=_xkey)
            cc = _cat_colors(groups, palette)
            for grp in groups:
                mask = df_sub[g] == grp
                col_ = _to_css_color(cc[grp])
                fig.add_trace(go.Scatter(x=xv[mask], y=yv[mask], mode=mode, name=str(grp),
                                          marker=dict(color=col_, size=8,
                                                      line=dict(color="white", width=0.5))))
        else:
            col_ = _to_css_color(_colors(3, palette)[1])
            fig.add_trace(go.Scatter(x=xv, y=yv, mode=mode,
                                      marker=dict(color=col_, size=8,
                                                  line=dict(color="white", width=0.5))))

    elif chart_key == "histogram":
        # go.Bar con bin PRE-CALCOLATI in Python (numpy), non go.Histogram
        # (dati grezzi, bin calcolati da Plotly.js nel browser): un
        # go.Histogram non è animabile con lo stesso motore delle barre
        # (chart_animation.py lavora su valori y già noti in Python, un
        # istogramma non li espone finché non lo renderizza il browser).
        # Bonus con più gruppi: bin CONDIVISI (calcolati sull'intera
        # colonna) invece che ognuno per conto suo — le barre dei gruppi
        # restano allineate sugli stessi intervalli, comparabili a vista.
        col_name = y or x
        if not col_name:
            raise ValueError("Seleziona una colonna numerica.")
        _NBINS = 20
        all_vals = pd.to_numeric(df_sub[col_name], errors="coerce").dropna()
        if all_vals.empty:
            raise ValueError("Nessun valore numerico valido nella colonna scelta.")
        _, edges = np.histogram(all_vals, bins=_NBINS)
        centers = (edges[:-1] + edges[1:]) / 2
        bin_w = edges[1] - edges[0]
        if g:
            groups = sorted(df_sub[g].dropna().unique(), key=_xkey)
            cols = [_to_css_color(c) for c in _colors(len(groups), palette)]
            for grp, col_ in zip(groups, cols):
                vals = pd.to_numeric(df_sub.loc[df_sub[g] == grp, col_name], errors="coerce").dropna()
                counts, _ = np.histogram(vals, bins=edges)
                fig.add_trace(go.Bar(x=centers, y=counts, name=str(grp),
                                      marker_color=col_, opacity=0.55, width=bin_w))
            fig.update_layout(barmode="overlay", bargap=0)
        else:
            counts, _ = np.histogram(all_vals, bins=edges)
            fig.add_trace(go.Bar(x=centers, y=counts,
                                  marker_color=_to_css_color(_colors(3, palette)[1]), width=bin_w))
            fig.update_layout(bargap=0)

    elif chart_key == "pie":
        if not x:
            raise ValueError("Seleziona Asse X (categorie).")
        if g:
            # Torta a doppio anello (stessa logica del motore matplotlib): anello
            # interno = totale per gruppo, anello esterno = categorie X dentro
            # ciascun gruppo, in sfumature del colore del gruppo. Due go.Pie
            # concentriche (non go.Sunburst) così l'animazione le tratta come torte.
            if is_freq:
                agg = df_sub.groupby([x, g], sort=False).size().reset_index(name="_v")
                vcol = "_v"
            else:
                agg = df_sub.groupby([x, g], sort=False)[y].sum().reset_index()
                vcol = y
            agg[vcol] = pd.to_numeric(agg[vcol], errors="coerce").fillna(0)
            group_tot = agg.groupby(g)[vcol].sum().sort_values(ascending=False)
            cc = _cat_colors(list(group_tot.index), palette)
            in_v, in_l, in_c, out_v, out_l, out_c = [], [], [], [], [], []
            for grp in group_tot.index:
                sub = agg[agg[g] == grp].sort_values(vcol, ascending=False).head(8)
                tot = float(sub[vcol].sum())
                if tot <= 0:
                    continue
                in_v.append(tot); in_l.append(str(grp)); in_c.append(_to_css_color(cc[grp]))
                base = mcolors.to_rgb(cc[grp])
                n_sub = len(sub)
                for i, (_, row) in enumerate(sub.iterrows()):
                    shade = 0.45 + 0.5 * (1 - i / max(n_sub - 1, 1))
                    out_c.append(mcolors.to_hex(tuple(c * shade + (1 - shade) for c in base)))
                    out_v.append(float(row[vcol])); out_l.append(f"{row[x]} ({grp})")
            if not in_v:
                raise ValueError("Nessun dato da mostrare per Asse X / Raggruppa selezionati.")
            fig.add_trace(go.Pie(labels=out_l, values=out_v, hole=0.62, sort=False,
                                  direction="clockwise", marker=dict(colors=out_c, line=dict(color="white", width=1)),
                                  textinfo="percent", textfont=dict(size=data_lbl_sz + 2),
                                  showlegend=False))
            fig.add_trace(go.Pie(labels=in_l, values=in_v, sort=False, direction="clockwise",
                                  domain=dict(x=[0.2, 0.8], y=[0.2, 0.8]),
                                  marker=dict(colors=in_c, line=dict(color="white", width=1)),
                                  textinfo="label+percent", textfont=dict(size=data_lbl_sz + 3)))
        else:
            agg_df, vcol = _agg_by_x(df_sub, x, y)
            agg_df = agg_df.sort_values(vcol, ascending=False)
            top  = agg_df.head(12)
            vals = pd.to_numeric(top[vcol], errors="coerce").fillna(0)
            cols = [_to_css_color(c) for c in _colors(len(top), palette)]
            fig.add_trace(go.Pie(labels=[str(l) for l in top[x]], values=vals,
                                  marker=dict(colors=cols), textinfo="percent",
                                  textfont=dict(size=data_lbl_sz + 4)))

    elif chart_key == "funnel":
        # go.Funnel nativo invece delle trapezoidi disegnate a mano in
        # matplotlib — motore Plotly = animabile con lo stesso meccanismo
        # già usato per bar/istogramma (list_cells/_trace_frame_update
        # trattano un funnel come una barra orizzontale: categorie su Y,
        # valori su X, un solo valore per livello che si accorcia/allarga).
        # Col Raggruppa: Plotly non ha un equivalente nativo delle
        # trapezoidi segmentate per sottogruppo della vecchia versione
        # matplotlib — usa invece il suo confronto nativo, più funnel
        # affiancati (funnelmode='group'), un colore per gruppo.
        if not x:
            raise ValueError("Seleziona Asse X (etichette) per l'imbuto.")
        if g and not is_freq:
            groups = sorted(df_sub[g].dropna().unique(), key=_xkey)
            cats = sorted(df_sub[x].dropna().unique(), key=_xkey)
            if y and y != _FREQ_LABEL:
                totals = {c: float(pd.to_numeric(df_sub.loc[df_sub[x] == c, y], errors="coerce").sum())
                          for c in cats}
            else:
                totals = {c: float((df_sub[x] == c).sum()) for c in cats}
            cats = sorted(cats, key=lambda c: totals.get(c, 0), reverse=True)
            if len(cats) < 2:
                raise ValueError("L'imbuto richiede almeno 2 categorie.")
            if y and y != _FREQ_LABEL:
                piv = df_sub.groupby([x, g])[y].sum().unstack(g).reindex(cats).fillna(0)
            else:
                piv = df_sub.groupby([x, g]).size().unstack(g).reindex(cats).fillna(0)
            cols = [_to_css_color(c) for c in _colors(len(groups), palette)]
            for grp, col_ in zip(groups, cols):
                vals = [float(piv.loc[c, grp]) if (c in piv.index and grp in piv.columns) else 0.0
                        for c in cats]
                fig.add_trace(go.Funnel(y=[str(c) for c in cats], x=vals, name=str(grp),
                                         marker=dict(color=col_),
                                         textinfo="value+percent initial"))
            fig.update_layout(funnelmode="group")
        else:
            agg_df, vcol = _agg_by_x(df_sub, x, y)
            agg_df = agg_df.sort_values(vcol, ascending=False).reset_index(drop=True)
            labels = [str(v) for v in agg_df[x]]
            vals = pd.to_numeric(agg_df[vcol], errors="coerce").fillna(0)
            if len(labels) < 2:
                raise ValueError("L'imbuto richiede almeno 2 categorie.")
            cols = [_to_css_color(c) for c in _colors(len(labels), palette)]
            fig.add_trace(go.Funnel(y=labels, x=vals, marker=dict(color=cols),
                                     textinfo="value+percent initial"))

    elif chart_key == "boxplot":
        if not y:
            raise ValueError("Seleziona Asse Y (colonna numerica).")
        if x:
            cats = sorted(df_sub[x].dropna().unique(), key=_xkey)
            if sort_y_desc:
                cats = _sort_cats_desc(df_sub, x, cats, y, agg="median")
            cc = _cat_colors(cats, palette)
            for c in cats:
                vals = pd.to_numeric(df_sub.loc[df_sub[x] == c, y], errors="coerce").dropna()
                fig.add_trace(go.Box(y=vals, name=str(c), marker_color=_to_css_color(cc[c])))
        else:
            vals = pd.to_numeric(df_sub[y], errors="coerce").dropna()
            fig.add_trace(go.Box(y=vals, marker_color=_to_css_color(_colors(3, palette)[1])))

    elif chart_key == "violin":
        if not y:
            raise ValueError("Seleziona Asse Y (valori numerici) per il violino.")
        def _add_violin(vals, name, color):
            if len(vals) >= 2:
                fig.add_trace(go.Violin(y=vals, name=name, line_color=color, fillcolor=color,
                                         opacity=0.65, box_visible=True, meanline_visible=True))
        if x:
            cats = sorted(df_sub[x].dropna().unique(), key=_xkey)
            if sort_y_desc:
                cats = _sort_cats_desc(df_sub, x, cats, y, agg="median")
            cc = _cat_colors(cats, palette)
            for c in cats:
                vals = pd.to_numeric(df_sub.loc[df_sub[x] == c, y], errors="coerce").dropna()
                _add_violin(vals, str(c), _to_css_color(cc[c]))
        elif g:
            groups = sorted(df_sub[g].dropna().unique(), key=_xkey)
            if sort_y_desc:
                groups = _sort_cats_desc(df_sub, g, groups, y, agg="median")
            cc = _cat_colors(groups, palette)
            for grp in groups:
                vals = pd.to_numeric(df_sub.loc[df_sub[g] == grp, y], errors="coerce").dropna()
                _add_violin(vals, str(grp), _to_css_color(cc[grp]))
        else:
            vals = pd.to_numeric(df_sub[y], errors="coerce").dropna()
            _add_violin(vals, str(y), _to_css_color(_colors(3, palette)[1]))

    elif chart_key == "radar":
        # Mirror di _render_chart_on_ax's chart_key=="radar" (matplotlib
        # polar) — stessa preparazione dati (medie per categoria, chiusura
        # del poligono ripetendo il primo punto), motore Plotly nativo
        # (go.Scatterpolar) invece dell'asse polare matplotlib.
        if not (x and y):
            raise ValueError("Radar richiede Asse X (categorie) e Asse Y (valori).")
        cats = sorted(df_sub[x].dropna().unique(), key=_xkey)
        if sort_y_desc:
            cats = _sort_cats_desc(df_sub, x, cats, y, agg="mean")
        if len(cats) < 3:
            raise ValueError("Il radar richiede almeno 3 categorie nell'Asse X.")
        cats_s = [str(c) for c in cats]
        n_cat = len(cats)
        # theta NUMERICO (gradi 0..360), non le stringhe delle categorie —
        # un angularaxis a CATEGORIE (theta=stringhe) impedisce a qualsiasi
        # traccia ausiliaria di animazione di posizionarsi tra due vertici
        # (serve per lo spicchio di 'cubetti'/'acqua': un valore theta
        # numerico su un asse a categorie non viene letto come grado, viene
        # registrato come categoria IN PIÙ, deformando l'asse — verificato
        # con uno screenshot che mostrava tick spuri "0.47"/"-0.47"). I nomi
        # restano leggibili uguali: tickvals/ticktext sotto li rimette al
        # posto giusto sull'asse ora numerico.
        theta_deg = [j * 360.0 / n_cat for j in range(n_cat)]
        theta_deg_c = theta_deg + [0.0]

        def _mean_per_cat(sub):
            # NaN (categoria senza dati per quel gruppo) → 0: "nan or 0" resta nan
            # in Python e spezzava il poligono in pezzetti.
            out = []
            for cat in cats:
                v = pd.to_numeric(sub.loc[sub[x] == cat, y], errors="coerce").mean()
                out.append(0.0 if pd.isna(v) else float(v))
            return out

        if g:
            groups = sorted(df_sub[g].dropna().unique(), key=_xkey)
            cc = _cat_colors(groups, palette)
            for grp in groups:
                vals_c = _mean_per_cat(df_sub[df_sub[g] == grp])
                vals_c = vals_c + [vals_c[0]]
                col_ = _to_css_color(cc[grp])
                fig.add_trace(go.Scatterpolar(r=vals_c, theta=theta_deg_c, name=str(grp),
                                              line=dict(color=col_, width=2),
                                              fill="toself", fillcolor=col_, opacity=0.75,
                                              mode="lines+markers", marker=dict(color=col_, size=5)))
        else:
            vals_c = _mean_per_cat(df_sub)
            vals_c = vals_c + [vals_c[0]]
            col_ = _to_css_color(_colors(3, palette)[1])
            fig.add_trace(go.Scatterpolar(r=vals_c, theta=theta_deg_c,
                                          line=dict(color=col_, width=2),
                                          fill="toself", fillcolor=col_, opacity=0.8,
                                          mode="lines+markers", marker=dict(color=col_, size=5)))
        fig.update_layout(polar=dict(
            # showline=False: quella era la riga "a raggio" (un righello
            # radiale pieno sovrapposto al poligono) segnalata come brutta —
            # bastano i numeri dei tick e i cerchi della griglia, senza una
            # linea dura che attraversa la forma.
            radialaxis=dict(visible=True, showline=False, rangemode="tozero"),
            angularaxis=dict(direction="clockwise", rotation=90,
                              tickmode="array", tickvals=theta_deg, ticktext=cats_s),
        ))

    else:
        raise ValueError(f"Tipo grafico non supportato (motore Plotly): {chart_key!r}")

    if chart_key in ("bar", "bar_stacked"):
        # Forza esplicitamente l'asse categoriale (invece di lasciarlo
        # all'inferenza automatica di Plotly) e uno spaziamento fisso tra le
        # barre — così ogni barra resta sempre centrata sul proprio tick,
        # indipendentemente da come sono fatti i valori della colonna X.
        cat_axis = fig.update_yaxes if (chart_key == "bar" and horiz) else fig.update_xaxes
        cat_axis(type="category")
        fig.update_layout(bargap=0.15, bargroupgap=0.1)

    # Etichette assi = nome colonna di default (come faceva il motore
    # matplotlib per alcuni tipi, es. bar/set_xlabel(x or "")) — l'utente può
    # poi sovrascriverle da "Etichetta Asse X/Y"; _apply_style_plotly le
    # tocca SOLO se non vuote, quindi questo default sopravvive finché non lo
    # si cambia esplicitamente.
    if chart_key not in ("pie", "radar", "funnel"):
        def_x, def_y = x, y
        if chart_key == "histogram":
            def_x, def_y = (y or x), "Frequenza"
        elif is_freq and chart_key in ("bar", "bar_stacked"):
            def_y = "Frequenza"
        if chart_key == "bar" and horiz:
            def_x, def_y = def_y, def_x
        if def_x:
            fig.update_xaxes(title_text=str(def_x))
        if def_y:
            fig.update_yaxes(title_text=str(def_y))

    if xtick_rot and xtick_rot > 0 and chart_key != "radar":
        fig.update_xaxes(tickangle=-int(xtick_rot))

    return fig


# ══════════════════════════════════════════════════════════════════════════════
# NAVIGAZIONE  (radio orizzontale — permette st.rerun() per cambiare pagina)
# ══════════════════════════════════════════════════════════════════════════════
if "active_tab" not in st.session_state:
    st.session_state.active_tab = 0

# Valori di default per le opzioni di personalizzazione
_OPT_DEFAULTS = {
    # Testi
    "opt_title":        "",
    "opt_title_sz":     13,
    "opt_title_bold":   True,
    "opt_title_ital":   False,
    "opt_xlabel":       "",
    "opt_ylabel":       "",
    "opt_label_sz":     10,
    # Stile
    "opt_palette_nm":   list(PALETTES.keys())[0],
    "opt_show_leg":     True,
    "opt_show_grid":    True,
    "opt_show_nums":    False,
    "opt_data_lbl_sz":  7,
    "opt_minimal":      False,
    "opt_show_axes_y":  True,
    "opt_show_axes_x":  True,
    "opt_bg_on":        False,
    "opt_bg_color":     "#FFFFFF",
    "opt_palette_mode": "Chiaro",
    "opt_legend_loc":    "Auto (migliore)",
    "opt_legend_off_x":  0.0,
    "opt_legend_off_y":  0.0,
    "opt_title_loc":     "Centro",
    "opt_title_off_y":   0.0,
    "opt_xlabel_off_x":  0.0,
    "opt_xlabel_off_y":  0.0,
    "opt_ylabel_off_x":  0.0,
    "opt_ylabel_off_y":  0.0,
    "opt_extra_texts":   [],
    # Font tick e legenda
    "opt_xtick_sz":     8,
    "opt_ytick_sz":     8,
    "opt_legend_sz":    8,
    # Assi base
    "opt_xtick_rot":    -1,
    "opt_y_min":        "",
    "opt_y_max":        "",
    # Scala
    "opt_y_scale":      "Normale",
    "opt_x_scale":      "Normale",
    "opt_y_step":       "",
    "opt_x_step":       "",
    # Minor tick
    "opt_y_minor":      False,
    "opt_y_minor_ndiv": 5,
    "opt_x_minor":      False,
    "opt_x_minor_ndiv": 5,
    # Griglia avanzata Y (major + minor)
    "opt_gy_maj":       False,
    "opt_gy_maj_sty":   "-- tratteggiata",
    "opt_gy_maj_alp":   0.4,
    "opt_gy_maj_lw":    0.8,
    "opt_gy_min":       False,
    "opt_gy_min_sty":   ":  punteggiata",
    "opt_gy_min_alp":   0.25,
    "opt_gy_min_lw":    0.5,
    "opt_gy_min_ndiv":  5,
    # Griglia avanzata X (major + minor)
    "opt_gx_maj":       False,
    "opt_gx_maj_sty":   "-- tratteggiata",
    "opt_gx_maj_alp":   0.4,
    "opt_gx_maj_lw":    0.8,
    "opt_gx_min":       False,
    "opt_gx_min_sty":   ":  punteggiata",
    "opt_gx_min_alp":   0.25,
    "opt_gx_min_lw":    0.5,
    "opt_gx_min_ndiv":  5,
    # Esportazione
    "opt_dpi":          200,
    "opt_fig_w":        11.0,
    "opt_fig_h":        6.0,
}
for _k, _v in _OPT_DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

_TABS = ["📁  Dati", "🗂  Variabili", "📊  Grafico", "⚙️  Personalizzazione", "🎬  Animazione"]
_sel = st.radio(
    "", _TABS, horizontal=True,
    index=st.session_state.active_tab,
    label_visibility="collapsed",
)
st.session_state.active_tab = _TABS.index(_sel)
st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# PAGINA 1 — DATI
# ══════════════════════════════════════════════════════════════════════════════
if _sel == _TABS[0]:

    st.subheader("Carica file dati")
    uploaded = st.file_uploader(
        "Trascina qui il file oppure clicca per cercarlo",
        type=["xlsx", "xls", "xlsm", "csv"],
        label_visibility="collapsed",
    )

    if uploaded:
        try:
            if uploaded.name.lower().endswith(".csv"):
                df_new = pd.read_csv(uploaded)
                label  = uploaded.name
            else:
                xf     = pd.ExcelFile(uploaded)
                sheet  = st.selectbox("Foglio", xf.sheet_names, key="sheet_sel")
                df_new = pd.read_excel(uploaded, sheet_name=sheet)
                label  = f"{uploaded.name} [{sheet}]"

            st.success(f"✓  {label}  —  {len(df_new):,} righe × {len(df_new.columns)} colonne")
            st.session_state.df = df_new
            st.session_state.col_roles = {c: "mostra" for c in df_new.columns}
            st.session_state.fig = None

        except Exception as e:
            st.error(f"Errore nel caricamento: {e}")

    if st.session_state.df is not None:
        st.dataframe(st.session_state.df.head(200), use_container_width=True)
        st.divider()
        if st.button("🗂  Vai alle Variabili →", use_container_width=True):
            st.session_state.active_tab = 1
            st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# PAGINA 2 — VARIABILI
# ══════════════════════════════════════════════════════════════════════════════
elif _sel == _TABS[1]:
    df = st.session_state.df
    if df is None:
        st.info("Carica prima un file nella sezione Dati.")
    else:
        st.subheader("Colonne visibili")
        st.caption("**Mostra** = colonna disponibile per assi e filtri  |  **Ignora** = esclusa")

        roles        = st.session_state.col_roles.copy()
        ROLE_OPTIONS = ["mostra", "ignora"]

        # ── Azioni rapide ──────────────────────────────────────────────────────
        qa, qb = st.columns(2)
        with qa:
            if st.button("✅  Mostra tutto", use_container_width=True):
                for col in df.columns:
                    roles[col] = "mostra"
                st.session_state.col_roles = roles
                st.rerun()
        with qb:
            if st.button("🚫  Ignora tutto", use_container_width=True):
                for col in df.columns:
                    roles[col] = "ignora"
                st.session_state.col_roles = roles
                st.rerun()

        st.divider()

        # ── Selettori per colonna (griglia 4 per riga) ────────────────────────
        N_COLS   = 4
        col_list = list(df.columns)
        for row_start in range(0, len(col_list), N_COLS):
            row_widgets = st.columns(N_COLS)
            for j, col_name in enumerate(col_list[row_start : row_start + N_COLS]):
                with row_widgets[j]:
                    cur  = roles.get(col_name, "mostra")
                    if cur not in ROLE_OPTIONS:
                        cur = "mostra"
                    icon = "🔢" if _is_num(df[col_name]) else "🔤"
                    new_role = st.selectbox(
                        f"{icon} {col_name}", ROLE_OPTIONS,
                        index=ROLE_OPTIONS.index(cur),
                        key=f"role_{col_name}",
                    )
                    roles[col_name] = new_role

        st.session_state.col_roles = roles

        show_cols = [c for c, r in roles.items() if r == "mostra"]
        if show_cols:
            st.divider()
            st.subheader("Anteprima dati")
            st.dataframe(df[show_cols].head(300), use_container_width=True)

        st.divider()
        if st.button("📊  Vai al Grafico →", use_container_width=True):
            st.session_state.active_tab = 2
            st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# PAGINA 3 — GRAFICO
# ══════════════════════════════════════════════════════════════════════════════
elif _sel == _TABS[2]:
    df = st.session_state.df
    if df is None:
        st.info("Carica prima un file nella sezione Dati.")
    else:
        roles   = st.session_state.col_roles
        visible = [""] + [c for c, r in roles.items() if r == "mostra"]
        indip   = visible
        dipend  = visible
        all_col = visible

        left, right = st.columns([1, 2])

        with left:
            ss = st.session_state

            # ── Stato persistente Tab Grafico (pattern: value= senza key=) ────
            _g3_init = {
                "g3_chart":     None,
                "g3_scope":     None,
                "g3_geo_col":   "",
                "g3_x_col":     "",
                "g3_y_col":     "",
                "g3_grp_col":   "",
                "g3_n_filt":    0,
                "g3_n_sep":     0,
                "g3_filters":   [{"col": "", "type": "Valori (lista)", "val": []}
                                  for _ in range(8)],
                "g3_seps":      [{"col": "", "mode": "cumulativo", "fval": ""}
                                  for _ in range(6)],
                # opzioni specifiche per tipo grafico
                "g3_bar_horiz":       False,
                "g3_bar_3d":          False,
                "g3_show_line":       False,
                "g3_line_col":        "",
                "g3_podio_n":         3,
                "g3_treemap_style":   "Classica",
                "g3_sort_y_desc":     False,
                "g3_highlight_cats":  [],
                "g3_highlight_color": "#E63946",
            }
            for _k, _v in _g3_init.items():
                if _k not in ss:
                    ss[_k] = _v

            # ── Tipo grafico ──────────────────────────────────────────────────
            st.subheader("Tipo di grafico")
            chart_labels = [t[0] for t in CHART_TYPES]
            chart_keys   = [t[1] for t in CHART_TYPES]
            _cl_idx      = chart_labels.index(ss.g3_chart) if ss.g3_chart in chart_labels else 0
            chart_label  = st.selectbox("", chart_labels, index=_cl_idx, label_visibility="collapsed")
            ss.g3_chart  = chart_label
            chart_key    = chart_keys[chart_labels.index(chart_label)]
            is_map       = chart_key in MAP_KEYS

            # ── Opzioni mappa ─────────────────────────────────────────────────
            if is_map:
                st.subheader("Impostazioni mappa")
                if not GEO_OK:
                    st.error("geopandas non installato — le mappe non sono disponibili.")
                _sc_opts    = [s[0] for s in MAP_SCOPES]
                _sc_idx     = _sc_opts.index(ss.g3_scope) if ss.g3_scope in _sc_opts else 0
                scope_label = st.selectbox("Ambito mappa", _sc_opts, index=_sc_idx)
                ss.g3_scope = scope_label
                scope_key   = MAP_SCOPE_KEYS[scope_label]
                _geo_opts   = [""] + list(df.columns)
                _geo_idx    = _geo_opts.index(ss.g3_geo_col) if ss.g3_geo_col in _geo_opts else 0
                geo_col     = st.selectbox(
                    "Colonna geo (nomi geografici nel tuo file)", _geo_opts, index=_geo_idx,
                    help="Seleziona la colonna che contiene i nomi delle aree geografiche",
                )
                ss.g3_geo_col = geo_col
                if chart_key == "band_map":
                    val_col_map     = st.selectbox("Valore (opzionale, per proporzionare bande)", dipend)
                    grp_col_map     = st.selectbox("Categoria colore (obbligatorio)", all_col)
                    band_orient     = st.radio(
                        "Direzione bande", ["Verticale (→)", "Orizzontale (↑)"], horizontal=True,
                    )
                    band_equal_area = st.checkbox(
                        "Area proporzionale al valore", value=False,
                        help=(
                            "Se attivo: l'area della banda nel poligono è proporzionale al valore "
                            "(ricerca binaria). Se disattivo: la larghezza/altezza del bounding box "
                            "è proporzionale al valore (più veloce, meno preciso per forme irregolari)."
                        ),
                    )
                elif chart_key in ("bubble_map", "pin_map"):
                    val_col_map = st.selectbox("Valore (dimensione/colore)", dipend)
                    grp_col_map = st.selectbox("Raggruppa/Colore (opzionale)", all_col)
                else:  # choropleth
                    val_col_map = st.selectbox("Valore numerico (Asse Y)", dipend)
                    grp_col_map = ""

            # ── Assi (solo grafici non-mappa) ─────────────────────────────────
            if not is_map:
                st.subheader("Assi")
                is_heatmap = (chart_key == "heatmap")
                is_podio   = (chart_key == "podio")
                is_funnel  = (chart_key == "funnel")
                is_radar   = (chart_key == "radar")
                is_pie     = (chart_key == "pie")

                # Etichette dinamiche per heatmap
                if is_heatmap:
                    lbl_x   = "Asse X1 — Colonne"
                    lbl_grp = "Asse X2 — Righe (obbligatorio)"
                elif is_podio:
                    lbl_x   = "Asse X — Nomi (etichette)"
                    lbl_grp = "Dettaglio aggiuntivo (opzionale)"
                elif is_funnel:
                    lbl_x   = "Asse X — Etichette livelli"
                    lbl_grp = "Raggruppa / Colore (opzionale)"
                elif is_radar:
                    lbl_x   = "Asse X — Categorie angoli"
                    lbl_grp = "Raggruppa / Colore (opzionale)"
                elif is_pie:
                    lbl_x   = "Asse X — Categorie (anello esterno)"
                    lbl_grp = "Raggruppa (anello interno, opzionale)"
                else:
                    lbl_x   = "Asse X"
                    lbl_grp = "Raggruppa / Colore (opzionale)"

                # Asse X
                _xi   = visible.index(ss.g3_x_col) if ss.g3_x_col in visible else 0
                x_col = st.selectbox(lbl_x, visible, index=_xi)
                ss.g3_x_col = x_col

                # Asse Y — include Frequenza tranne per grafici che richiedono colonna reale
                _y_no_freq = {"boxplot", "violin", "scatter", "histogram", "radar"}
                if chart_key in _y_no_freq:
                    y_opts  = visible
                    _yi_def = visible.index(ss.g3_y_col) if ss.g3_y_col in visible else 0
                else:
                    y_opts  = visible[:]  # copia
                    # inserisci Frequenza dopo il primo elemento (stringa vuota)
                    y_opts.insert(1, _FREQ_LABEL)
                    _y_ss   = ss.g3_y_col if ss.g3_y_col else ""
                    _yi_def = y_opts.index(_y_ss) if _y_ss in y_opts else 0

                if is_heatmap:
                    lbl_y = "Colore / Valori (opzionale — vuoto = Frequenza)"
                elif is_podio:
                    lbl_y = "Asse Y — Punteggio/Valore (opzionale)"
                elif is_funnel:
                    lbl_y = "Asse Y — Valori (opzionale — vuoto = Frequenza)"
                else:
                    lbl_y = "Asse Y"
                y_col = st.selectbox(lbl_y, y_opts, index=_yi_def)
                ss.g3_y_col = y_col

                # Raggruppa / Asse X2
                _gi     = visible.index(ss.g3_grp_col) if ss.g3_grp_col in visible else 0
                grp_col = st.selectbox(lbl_grp, visible, index=_gi)
                ss.g3_grp_col = grp_col

                # Ordinamento per valore Y: disponibile per ogni tipo dove ha senso
                # una singola categoria → un valore aggregato (non per heatmap/scatter/
                # histogram, che non hanno questo concetto; pie/funnel/podio/treemap
                # ordinano già SEMPRE per valore decrescente per come sono definiti).
                _SORTABLE_CHART_KEYS = {"bar", "bar_stacked", "line", "boxplot",
                                         "violin", "radar", "combo"}
                if chart_key in _SORTABLE_CHART_KEYS:
                    ss.g3_sort_y_desc = st.checkbox("Ordina categorie per valore Y (decrescente)",
                                                     value=bool(ss.g3_sort_y_desc))
                else:
                    ss.g3_sort_y_desc = False

                # ── Opzioni specifiche per tipo ────────────────────────────────
                if chart_key == "bar":
                    ss.g3_bar_horiz = st.checkbox("Barre orizzontali",
                                                   value=bool(ss.g3_bar_horiz))
                    if not (y_col == _FREQ_LABEL or not y_col):
                        ss.g3_show_line = st.checkbox("Aggiungi linea sovrapposta",
                                                       value=bool(ss.g3_show_line))
                    _hl_opts = sorted({str(v) for v in df[x_col].dropna().unique()}) if x_col else []
                    ss.g3_highlight_cats = st.multiselect(
                        "Evidenzia categorie (colore dedicato)", _hl_opts,
                        default=[v for v in ss.g3_highlight_cats if v in _hl_opts])
                    if ss.g3_highlight_cats:
                        ss.g3_highlight_color = st.color_picker(
                            "Colore evidenziato", value=ss.g3_highlight_color)

                elif chart_key == "bar_stacked":
                    ss.g3_bar_3d = st.checkbox("Variante 3D (Pila mattoncini)",
                                                value=bool(ss.g3_bar_3d))

                elif chart_key == "scatter":
                    ss.g3_show_line = st.checkbox("Mostra linee di collegamento",
                                                   value=bool(ss.g3_show_line))

                elif chart_key == "combo":
                    _lc_opts  = [""] + [c for c in list(df.columns) if c != x_col]
                    _lc_val   = ss.g3_line_col if ss.g3_line_col in _lc_opts else ""
                    _lc_i     = _lc_opts.index(_lc_val) if _lc_val in _lc_opts else 0
                    _sel_lc   = st.selectbox("Colonna per la linea (asse Y2)",
                                              _lc_opts, index=_lc_i)
                    ss.g3_line_col = _sel_lc

                elif chart_key == "podio":
                    ss.g3_podio_n = int(st.number_input(
                        "Numero di posizioni da mostrare", 1, 10,
                        value=int(ss.g3_podio_n), step=1))

                elif chart_key == "treemap":
                    _tm_opts = ["Classica", "Albero", "Spirale"]
                    _tm_idx  = (_tm_opts.index(ss.g3_treemap_style)
                                if ss.g3_treemap_style in _tm_opts else 0)
                    ss.g3_treemap_style = st.selectbox(
                        "Variante treemap", _tm_opts, index=_tm_idx)

            # ── Personalizzazione (legge da session_state, si imposta nella tab ⚙️) ──
            title      = ss.opt_title
            palette_nm = ss.opt_palette_nm
            if palette_nm in ss.get("custom_palettes", {}):
                palette = palette_nm  # _resolve_brand la trova nelle palette personalizzate
            else:
                palette = PALETTES.get(palette_nm, list(PALETTES.values())[0])
            show_leg   = ss.opt_show_leg
            if not is_map:
                xlabel    = ss.opt_xlabel
                ylabel    = ss.opt_ylabel
                show_grid = ss.opt_show_grid
                show_nums = ss.opt_show_nums
                xtick_rot = ss.opt_xtick_rot
            else:
                xlabel = ylabel = ""
                show_grid = show_nums = False
                xtick_rot = -1

            # ── Filtri ────────────────────────────────────────────────────────
            with st.expander("🔍  Filtri"):
                _nf       = int(st.number_input("Numero di filtri", 0, 8, value=int(ss.g3_n_filt), step=1))
                ss.g3_n_filt = _nf
                df_filt   = df.copy()
                _FC_TYPES = ["Valori (lista)", "Contiene", "Non contiene", "Num ≥", "Num ≤"]
                _fc_opts  = [""] + list(df.columns)

                for fi in range(_nf):
                    _fs = ss.g3_filters[fi]
                    st.markdown(f"**Filtro {fi + 1}**")
                    fc1, fc2 = st.columns(2)
                    with fc1:
                        _fci     = _fc_opts.index(_fs["col"]) if _fs["col"] in _fc_opts else 0
                        filt_col = st.selectbox("Colonna", _fc_opts, index=_fci, key=f"g3_filt_col_{fi}")
                        _fs["col"] = filt_col
                    with fc2:
                        _fti      = _FC_TYPES.index(_fs["type"]) if _fs["type"] in _FC_TYPES else 0
                        filt_type = st.selectbox("Tipo", _FC_TYPES, index=_fti, key=f"g3_filt_type_{fi}")
                        _fs["type"] = filt_type
                    if filt_col and filt_col in df.columns:
                        if filt_type == "Valori (lista)":
                            uniq  = sorted(df_filt[filt_col].dropna().unique(), key=_xkey)
                            _fdef = [v for v in (_fs.get("val") or []) if v in uniq]
                            sel   = st.multiselect("Valori da includere", uniq, default=_fdef, key=f"g3_filt_val_{fi}")
                            _fs["val"] = sel
                            if sel:
                                df_filt = df_filt[df_filt[filt_col].astype(str).isin([str(v) for v in sel])]
                        elif filt_type in ("Contiene", "Non contiene"):
                            _fv  = str(_fs.get("val") or "") if not isinstance(_fs.get("val"), list) else ""
                            txt  = st.text_input("Testo", value=_fv, key=f"g3_filt_txt_{fi}")
                            _fs["val"] = txt
                            if txt:
                                mask    = df_filt[filt_col].astype(str).str.contains(txt, case=False, na=False)
                                df_filt = df_filt[mask if filt_type == "Contiene" else ~mask]
                        elif filt_type == "Num ≥":
                            _fv  = float(_fs["val"]) if isinstance(_fs.get("val"), (int, float)) else 0.0
                            val  = st.number_input("Valore minimo", value=_fv, key=f"g3_filt_min_{fi}")
                            _fs["val"] = val
                            df_filt = df_filt[pd.to_numeric(df_filt[filt_col], errors="coerce") >= val]
                        elif filt_type == "Num ≤":
                            _fv  = float(_fs["val"]) if isinstance(_fs.get("val"), (int, float)) else 0.0
                            val  = st.number_input("Valore massimo", value=_fv, key=f"g3_filt_max_{fi}")
                            _fs["val"] = val
                            df_filt = df_filt[pd.to_numeric(df_filt[filt_col], errors="coerce") <= val]
                    st.caption(f"→ {len(df_filt):,} righe dopo questo filtro")
                    ss.g3_filters[fi] = _fs

            # ── Separa ────────────────────────────────────────────────────────
            sep_col   = ""
            _SEP_MODES = ["cumulativo", "separa", "valore fisso"]
            with st.expander("✂️  Separa"):
                _ns      = int(st.number_input("Numero di colonne", 0, 6, value=int(ss.g3_n_sep), step=1))
                ss.g3_n_sep = _ns
                _vis_only   = [c for c in visible if c]
                _sv_opts    = [""] + _vis_only
                for _si in range(_ns):
                    _sp = ss.g3_seps[_si]
                    _sca, _scb = st.columns(2)
                    with _sca:
                        _sci = _sv_opts.index(_sp["col"]) if _sp["col"] in _sv_opts else 0
                        _sc  = st.selectbox("Colonna", _sv_opts, index=_sci, key=f"g3_sep_col_{_si}")
                        _sp["col"] = _sc
                    with _scb:
                        _smi = _SEP_MODES.index(_sp["mode"]) if _sp["mode"] in _SEP_MODES else 0
                        _sm  = st.selectbox("Modalità", _SEP_MODES, index=_smi, key=f"g3_sep_mode_{_si}")
                        _sp["mode"] = _sm
                    if _sc and _sm == "separa" and not sep_col:
                        sep_col = _sc
                    elif _sc and _sm == "valore fisso" and _sc in df_filt.columns:
                        _fv_opts = [""] + sorted([str(v) for v in df_filt[_sc].dropna().unique()], key=_xkey)
                        _fvi     = _fv_opts.index(_sp["fval"]) if _sp["fval"] in _fv_opts else 0
                        _fv      = st.selectbox("Valore fisso", _fv_opts, index=_fvi, key=f"g3_sep_fval_{_si}")
                        _sp["fval"] = _fv
                        if _fv:
                            df_filt = df_filt[df_filt[_sc].astype(str) == _fv]
                    ss.g3_seps[_si] = _sp

            gen = st.button("▶  Genera grafico", type="primary", use_container_width=True)

        # ── Area grafico ──────────────────────────────────────────────────────
        # Raccoglie le opzioni di stile dalla session_state (impostati in tab ⚙️)
        _so = st.session_state
        style_opts = {
            "title_sz":      int(_so.opt_title_sz),
            "title_bold":    bool(_so.opt_title_bold),
            "title_ital":    bool(_so.opt_title_ital),
            "label_sz":      int(_so.opt_label_sz),
            "y_min":         str(_so.opt_y_min),
            "y_max":         str(_so.opt_y_max),
            "minimal":       bool(_so.opt_minimal),
            "show_axes_y":   bool(_so.opt_show_axes_y),
            "show_axes_x":   bool(_so.opt_show_axes_x),
            "bg_on":         bool(_so.opt_bg_on),
            "bg_color":      _so.opt_bg_color,
            "palette_mode":  _so.opt_palette_mode,
            "legend_sz":     int(_so.opt_legend_sz),
            "legend_loc":       str(_so.opt_legend_loc),
            "legend_offset_x":  float(_so.opt_legend_off_x),
            "legend_offset_y":  float(_so.opt_legend_off_y),
            "title_loc":        str(_so.opt_title_loc),
            "title_offset_y":   float(_so.opt_title_off_y),
            "xlabel_offset_x":  float(_so.opt_xlabel_off_x),
            "xlabel_offset_y":  float(_so.opt_xlabel_off_y),
            "ylabel_offset_x":  float(_so.opt_ylabel_off_x),
            "ylabel_offset_y":  float(_so.opt_ylabel_off_y),
            "extra_texts":      list(_so.opt_extra_texts),
        }
        axis_opts = {
            "y_scale":       str(_so.opt_y_scale),
            "x_scale":       str(_so.opt_x_scale),
            "y_step":        str(_so.opt_y_step),
            "x_step":        str(_so.opt_x_step),
            "y_minor":       bool(_so.opt_y_minor),
            "y_minor_ndiv":  int(_so.opt_y_minor_ndiv),
            "x_minor":       bool(_so.opt_x_minor),
            "x_minor_ndiv":  int(_so.opt_x_minor_ndiv),
            "gy_maj":        bool(_so.opt_gy_maj),
            "gy_maj_sty":    str(_so.opt_gy_maj_sty),
            "gy_maj_alp":    float(_so.opt_gy_maj_alp),
            "gy_maj_lw":     float(_so.opt_gy_maj_lw),
            "gy_min":        bool(_so.opt_gy_min),
            "gy_min_sty":    str(_so.opt_gy_min_sty),
            "gy_min_alp":    float(_so.opt_gy_min_alp),
            "gy_min_lw":     float(_so.opt_gy_min_lw),
            "gy_min_ndiv":   int(_so.opt_gy_min_ndiv),
            "gx_maj":        bool(_so.opt_gx_maj),
            "gx_maj_sty":    str(_so.opt_gx_maj_sty),
            "gx_maj_alp":    float(_so.opt_gx_maj_alp),
            "gx_maj_lw":     float(_so.opt_gx_maj_lw),
            "gx_min":        bool(_so.opt_gx_min),
            "gx_min_sty":    str(_so.opt_gx_min_sty),
            "gx_min_alp":    float(_so.opt_gx_min_alp),
            "gx_min_lw":     float(_so.opt_gx_min_lw),
            "gx_min_ndiv":   int(_so.opt_gx_min_ndiv),
            "xtick_sz":      int(_so.opt_xtick_sz),
            "ytick_sz":      int(_so.opt_ytick_sz),
        }

        with right:
            # ── Navigazione Separa ─────────────────────────────────────────────
            _df_chart    = df_filt
            _chart_title = title
            # BUG corretto: prima il grafico si ridisegnava SOLO al click di
            # "Genera grafico" — cambiare titolo/legenda/griglia nella tab
            # Personalizzazione (o qualunque altra opzione) non aggiornava
            # st.session_state.fig, quindi i bottoni di export sotto (che lo
            # leggono sempre, a prescindere da questo blocco) scaricavano il
            # vecchio grafico con le impostazioni precedenti — sembrava che
            # l'export "ignorasse" le modifiche, ma in realtà non era mai
            # stato rigenerato. Ora, una volta generato almeno una volta, si
            # ridisegna ad ogni rerun (ogni interazione con un controllo fa
            # già ripartire lo script in Streamlit, come per ogni altro
            # widget di questa pagina) — il bottone resta necessario solo per
            # la primissima generazione.
            do_render    = gen or (st.session_state.get("fig") is not None)

            if sep_col and sep_col in df_filt.columns:
                _sv  = sorted(df_filt[sep_col].dropna().unique(), key=_xkey)
                _nsv = len(_sv)
                # Resetta indice quando la colonna separa cambia
                if st.session_state.get("_sep_col_last") != sep_col:
                    st.session_state["_sep_idx"]      = 0
                    st.session_state["_sep_col_last"] = sep_col
                _idx = max(0, min(int(st.session_state.get("_sep_idx", 0)), _nsv - 1))
                st.session_state["_sep_idx"] = _idx

                _nc1, _nc2, _nc3 = st.columns([1, 6, 1])
                with _nc1:
                    if st.button("◀", disabled=(_idx == 0), key="sep_btn_prev"):
                        st.session_state["_sep_idx"] = max(0, _idx - 1)
                        do_render = True
                with _nc2:
                    _cur_v = _sv[st.session_state["_sep_idx"]]
                    st.markdown(
                        f"**{st.session_state['_sep_idx'] + 1} / {_nsv}**"
                        f" — *{sep_col}*: **{_cur_v}**"
                    )
                with _nc3:
                    if st.button("▶", disabled=(_idx >= _nsv - 1), key="sep_btn_next"):
                        st.session_state["_sep_idx"] = min(_nsv - 1, _idx + 1)
                        do_render = True

                _cur_v       = _sv[st.session_state["_sep_idx"]]
                _df_chart    = df_filt[df_filt[sep_col].astype(str) == str(_cur_v)]
                _chart_title = title or f"{sep_col}: {_cur_v}"

            if do_render:
                try:
                    _fw = float(st.session_state.opt_fig_w)
                    _fh = float(st.session_state.opt_fig_h)

                    if not is_map and palette:
                        plt.rcParams["axes.prop_cycle"] = plt.cycler(
                            color=_colors(10, palette)
                        )

                    # ── MAPPE ─────────────────────────────────────────────────
                    if is_map:
                        if not GEO_OK:
                            raise ImportError("Installa geopandas per usare le mappe.")
                        if not geo_col:
                            raise ValueError("Seleziona la colonna geo.")

                        fig, ax = plt.subplots(figsize=(_fw, _fh))
                        gdf = _load_geo(scope_key)
                        if gdf is None:
                            raise ValueError(
                                f"Impossibile caricare le geometrie per '{scope_label}'."
                            )

                        if chart_key == "choropleth":
                            if not val_col_map:
                                raise ValueError(
                                    "Seleziona il Valore numerico per la mappa coropletica."
                                )
                            _plot_choropleth(fig, ax, _df_chart, geo_col, val_col_map,
                                             gdf, palette, _chart_title)

                        elif chart_key == "bubble_map":
                            _plot_bubble_map(fig, ax, _df_chart, geo_col,
                                             val_col_map or None,
                                             grp_col_map or None,
                                             gdf, palette, _chart_title, show_leg)

                        elif chart_key == "band_map":
                            _orient = "Verticale" if band_orient.startswith("V") else "Orizzontale"
                            _plot_band_map(fig, ax, _df_chart, geo_col,
                                           grp_col_map or None,
                                           val_col_map or None,
                                           gdf, palette, _chart_title, show_leg,
                                           orientation=_orient,
                                           equal_area=band_equal_area)

                        elif chart_key == "pin_map":
                            _plot_pin_map(fig, ax, _df_chart, geo_col,
                                          val_col_map or None,
                                          grp_col_map or None,
                                          gdf, palette, _chart_title, show_leg)

                        fig.tight_layout(pad=1.5)

                    # ── GRAFICI STANDARD ──────────────────────────────────────
                    else:
                        # y_col può essere _FREQ_LABEL (stringa speciale) o colonna reale
                        x    = x_col or None
                        y    = y_col if y_col else None
                        g    = grp_col or None
                        _dlz = int(_so.opt_data_lbl_sz)
                        _is_3d = (chart_key == "bar_stacked" and bool(ss.g3_bar_3d))

                        _extra = {
                            "horiz":          bool(ss.g3_bar_horiz),
                            "stacked_3d":     bool(ss.g3_bar_3d),
                            "show_line":      bool(ss.g3_show_line),
                            "line_col":       ss.g3_line_col or None,
                            "podio_n":        int(ss.g3_podio_n),
                            "treemap_style":  str(ss.g3_treemap_style),
                            "show_legend":    bool(show_leg),
                            "show_xlabels":   bool(_so.opt_show_axes_x),
                            "y_scale":        str(axis_opts.get("y_scale", "Normale")),
                            "sort_y_desc":    bool(ss.g3_sort_y_desc),
                            "highlight_cats": list(ss.g3_highlight_cats),
                            "highlight_color": str(ss.g3_highlight_color),
                            "legend_loc":       str(_so.opt_legend_loc),
                            "legend_offset_x":  float(_so.opt_legend_off_x),
                            "legend_offset_y":  float(_so.opt_legend_off_y),
                            "legend_sz":        int(_so.opt_legend_sz),
                            "fig_w": _fw, "fig_h": _fh,
                        }

                        if chart_key in _PLOTLY_CHART_KEYS and not _is_3d:
                            # Motore Plotly (Fase 1): grafico interattivo, legenda/
                            # titolo trascinabili col mouse a schermo.
                            fig = _render_chart_plotly(
                                _df_chart, chart_key, x, y, g,
                                palette, show_nums, xtick_rot, _dlz, extra=_extra)
                            _apply_style_plotly(fig, _chart_title, xlabel, ylabel,
                                                 show_leg, show_grid, opts={**style_opts, **_extra})
                            _apply_axis_settings_plotly(
                                fig, axis_opts,
                                horiz=(bool(ss.g3_bar_horiz) and chart_key == "bar") or chart_key == "funnel")
                        else:
                            # Motore matplotlib (tipi non ancora migrati a Plotly)
                            mpl_fig, ax = plt.subplots(figsize=(_fw, _fh))
                            ax = _render_chart_on_ax(
                                ax, _df_chart, chart_key, x, y, g,
                                palette, show_nums, xtick_rot, _dlz, extra=_extra)
                            # non applicare impostazioni asse su grafici con layout custom
                            if chart_key not in _NO_AXIS_KEYS and not _is_3d:
                                _apply_style(ax, _chart_title, xlabel, ylabel,
                                             show_leg, show_grid, opts=style_opts)
                                _apply_axis_settings(ax, axis_opts)
                            else:
                                if _chart_title:
                                    ax.set_title(_chart_title, fontsize=13, fontweight="bold")
                            mpl_fig.tight_layout(pad=1.5)
                            fig = mpl_fig

                    st.session_state.fig = fig
                    if is_map:
                        _pkg_roles = {"geo": geo_col, "value": val_col_map}
                    else:
                        _pkg_roles = {"x": x_col, "y": y_col, "group": grp_col}
                    st.session_state.last_chart_pkg = {
                        "chart_type": chart_key,
                        "data": _df_chart.to_dict(orient="records"),
                        "column_roles": _pkg_roles,
                        "style": {
                            "palette": palette_nm, "title": _chart_title,
                            "xlabel": xlabel, "ylabel": ylabel,
                            "show_legend": bool(show_leg), "show_grid": bool(show_grid),
                        },
                    }

                except Exception as e:
                    st.error(f"Errore grafico: {e}")
                    st.session_state.fig = None
                    st.session_state.last_chart_pkg = None

            # ── Mostra grafico + download ──────────────────────────────────────
            if st.session_state.fig is not None:
                _fig_obj    = st.session_state.fig
                _is_plotly  = isinstance(_fig_obj, go.Figure)

                if _is_plotly:
                    # editable=True + edits: legenda e titolo si trascinano col
                    # mouse direttamente nel grafico. Nota (comunicata e accettata
                    # dall'utente): questo spostamento è SOLO lato browser — non
                    # torna a Python, quindi NON si riflette nei bottoni "Scarica
                    # PNG"/"Scarica SVG"/"Esporta per Pipeline Video" qui sotto
                    # (che rigenerano lato server dalla posizione di default/
                    # manuale). Per un'immagine che includa lo spostamento fatto
                    # col mouse, usa l'icona fotocamera nella toolbar del grafico.
                    st.plotly_chart(_fig_obj, use_container_width=True,
                                     config={"editable": True,
                                             "edits": {"legendPosition": True,
                                                       "titleText": True,
                                                       "annotationPosition": True}})
                    st.caption("🖱️ Legenda e titolo si trascinano col mouse direttamente sul grafico. "
                              "Per un'immagine che includa lo spostamento, usa l'icona fotocamera "
                              "nella barra degli strumenti del grafico (non i bottoni sotto).")
                else:
                    # Anteprima come immagine PNG (anziché st.pyplot): si adatta sempre
                    # alla larghezza della colonna, altrimenti con figure grandi/DPI alti
                    # il grafico appare enorme e bisogna scorrere per vederlo tutto.
                    _prev_buf = io.BytesIO()
                    _fig_obj.savefig(_prev_buf, format="png", dpi=110, bbox_inches="tight")
                    st.image(_prev_buf.getvalue(), use_container_width=True)

                def _fig_bytes(fmt: str) -> bytes:
                    if _is_plotly:
                        return _fig_obj.to_image(format=fmt)
                    buf = io.BytesIO()
                    kw = {"dpi": int(st.session_state.opt_dpi)} if fmt in ("png", "jpg") else {}
                    if fmt == "jpg":
                        kw["facecolor"] = "white"  # il JPG non supporta la trasparenza
                    _fig_obj.savefig(buf, format=fmt, bbox_inches="tight", **kw)
                    return buf.getvalue()

                _DL_FMTS = [("png", "PNG", "image/png"),
                            ("jpg", "JPG", "image/jpeg"),
                            ("svg", "SVG", "image/svg+xml")]
                # Per i grafici Plotly l'export passa da Kaleido, che richiede Chromium
                # sul server (su Streamlit Cloud: 'chromium' in packages.txt). Se manca,
                # il grafico resta visibile e si scarica dall'icona fotocamera.
                try:
                    _dl_data = {ext: _fig_bytes(ext) for ext, _, _ in _DL_FMTS}
                except Exception as _exp_err:
                    _dl_data = None
                    st.warning("Export immagine non disponibile su questo server "
                               f"({type(_exp_err).__name__}). Usa l'icona 📷 nella barra "
                               "del grafico per scaricarlo in PNG.")
                if _dl_data:
                    for _dl_col, (_ext, _lbl, _mime) in zip(st.columns(len(_DL_FMTS)), _DL_FMTS):
                        with _dl_col:
                            st.download_button(
                                label=f"💾  Scarica {_lbl}",
                                data=_dl_data[_ext],
                                file_name=f"grafico.{_ext}",
                                mime=_mime,
                                use_container_width=True,
                                key=f"dl_fig_{_ext}",
                            )

                if _dl_data:
                    with st.expander("🎬 Esporta per Pipeline Video"):
                        st.caption("Pacchetto con dati e parametri di questo grafico, da importare in "
                                  "Pipeline Video (sequenza animazione → sorgente 'Da GUI Grafici "
                                  "semplificata') per farne un'animazione.")
                        _export_pkg = {
                            **(st.session_state.last_chart_pkg or {}),
                            "svg": _dl_data["svg"].decode("utf-8"),
                        }
                        st.download_button(
                            label="📤  Esporta per Pipeline Video (.json)",
                            data=json.dumps(_export_pkg, ensure_ascii=False, default=str),
                            file_name="grafico_per_pipeline_video.json",
                            mime="application/json",
                            use_container_width=True,
                        )
            else:
                st.info("Imposta le opzioni a sinistra e clicca **▶ Genera grafico**.")

        st.divider()
        if st.button("⚙️  Vai alle Impostazioni →", use_container_width=True):
            st.session_state.active_tab = 3
            st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# PAGINA 4 — PERSONALIZZAZIONE
# Usa value= + writeback immediato (NO key=) per evitare che Streamlit
# cancelli i valori quando si torna su un'altra tab.
# ══════════════════════════════════════════════════════════════════════════════
elif _sel == _TABS[3]:
    ss = st.session_state
    st.subheader("⚙️  Personalizzazione grafico")
    st.caption("Le impostazioni si applicano al prossimo **▶ Genera grafico**.")

    # ── 🎨 Stile ──────────────────────────────────────────────────────────────
    with st.expander("🎨  Stile e colori", expanded=True):
        _custom_names = list(ss.get("custom_palettes", {}).keys())
        _pal_keys = _custom_names + list(PALETTES.keys())
        _pal_idx  = _pal_keys.index(ss.opt_palette_nm) if ss.opt_palette_nm in _pal_keys else 0
        ss.opt_palette_nm = st.selectbox(
            "Palette colori", _pal_keys, index=_pal_idx,
            help="Le palette create da te (🖌️ qui sotto) compaiono in cima alla lista.")

        sc1, sc2, sc3 = st.columns(3)
        with sc1:
            ss.opt_show_leg    = st.checkbox("Mostra legenda",        value=ss.opt_show_leg)
        with sc2:
            ss.opt_show_grid   = st.checkbox("Griglia",               value=ss.opt_show_grid)
        with sc3:
            ss.opt_show_nums   = st.checkbox("Numeri sulle barre",    value=ss.opt_show_nums)

        sc4, sc5, sc6 = st.columns(3)
        with sc4:
            ss.opt_minimal     = st.checkbox("Stile minimal",         value=ss.opt_minimal,
                                             help="Rimuove le spine superiore e destra del grafico")
        with sc5:
            ss.opt_show_axes_y = st.checkbox("Mostra asse Y",         value=ss.opt_show_axes_y)
        with sc6:
            _pm_opts = ["Chiaro", "Scuro"]
            _pm_idx  = _pm_opts.index(ss.opt_palette_mode) if ss.opt_palette_mode in _pm_opts else 0
            ss.opt_palette_mode = st.selectbox(
                "Palette per sfondo", _pm_opts, index=_pm_idx,
                help="Indipendente dal colore di sfondo qui sotto: 'Scuro' schiarisce "
                     "automaticamente la palette scelta sopra e i testi (assi, titolo, "
                     "legenda), qualunque palette sia — per un grafico che finirà su uno "
                     "sfondo scuro anche se qui lo tieni trasparente per comporlo altrove "
                     "(es. su una slide).")

        sc7, sc8 = st.columns(2)
        with sc7:
            ss.opt_show_axes_x = st.checkbox("Mostra etichette asse X", value=ss.opt_show_axes_x)
        with sc8:
            ss.opt_bg_on = st.checkbox("Aggiungi sfondo", value=ss.opt_bg_on,
                                       help="Senza spunta il grafico resta trasparente "
                                            "(comportamento di default, adatto a comporlo "
                                            "su un'altra slide/sfondo).")
            if ss.opt_bg_on:
                ss.opt_bg_color = st.color_picker("Colore sfondo", value=ss.opt_bg_color)

        st.caption("Posizione legenda (nessun trascinamento col mouse su Streamlit — "
                  "posizione + scostamento manuale, vale per tutti i tipi di grafico)")
        lc1, lc2, lc3 = st.columns([2, 1, 1])
        with lc1:
            _leg_opts = list(_LEGEND_POSITIONS.keys())
            ss.opt_legend_loc = st.selectbox("Posizione legenda", _leg_opts,
                index=_leg_opts.index(ss.opt_legend_loc) if ss.opt_legend_loc in _leg_opts else 0)
        with lc2:
            ss.opt_legend_off_x = st.slider("Scost. X", -0.4, 0.4, float(ss.opt_legend_off_x), 0.02)
        with lc3:
            ss.opt_legend_off_y = st.slider("Scost. Y", -0.4, 0.4, float(ss.opt_legend_off_y), 0.02)

    # ── 🖌️ Palette personalizzata ───────────────────────────────────────────
    with st.expander("🖌️  Crea la tua palette colori", expanded=False):
        st.caption(
            "Parti da una palette esistente, poi modifica i colori e il loro ordine "
            "(l'ordine conta: il 1° colore è usato per la prima categoria, e così via)."
        )
        if "custom_palette_edit" not in ss:
            ss.custom_palette_edit = [mcolors.to_hex(c) for c in _colors(6, "Tab10")]

        _base_opts = list(PALETTES.keys())
        _cpb1, _cpb2 = st.columns([3, 1])
        with _cpb1:
            _base_pick = st.selectbox("Parti da questa palette", _base_opts, key="cp_base_pick")
        with _cpb2:
            st.markdown("")
            if st.button("⬇️  Carica", key="cp_load_base", use_container_width=True):
                ss.custom_palette_edit = [mcolors.to_hex(c) for c in _colors(6, PALETTES[_base_pick])]
                st.rerun()

        st.markdown("**I tuoi colori** (dall'alto verso il basso = ordine di utilizzo)")
        _edit = ss.custom_palette_edit
        for _i in range(len(_edit)):
            _cc1, _cc2, _cc3, _cc4 = st.columns([4, 1, 1, 1])
            with _cc1:
                _edit[_i] = st.color_picker(f"Colore {_i + 1}", value=_edit[_i], key=f"cp_color_{_i}")
            with _cc2:
                if st.button("⬆️", key=f"cp_up_{_i}", disabled=(_i == 0), use_container_width=True,
                              help="Sposta su"):
                    _edit[_i - 1], _edit[_i] = _edit[_i], _edit[_i - 1]
                    st.rerun()
            with _cc3:
                if st.button("⬇️", key=f"cp_dn_{_i}", disabled=(_i == len(_edit) - 1),
                              use_container_width=True, help="Sposta giù"):
                    _edit[_i + 1], _edit[_i] = _edit[_i], _edit[_i + 1]
                    st.rerun()
            with _cc4:
                if st.button("🗑️", key=f"cp_del_{_i}", disabled=(len(_edit) <= 2),
                              use_container_width=True, help="Elimina questo colore"):
                    _edit.pop(_i)
                    st.rerun()

        if st.button("➕  Aggiungi un colore"):
            _edit.append("#888888")
            st.rerun()

        st.markdown(
            "Anteprima: " + "".join(
                f"<span style='display:inline-block;width:26px;height:26px;"
                f"background:{c};margin-right:4px;border:1px solid #999;"
                f"border-radius:4px;vertical-align:middle;'></span>" for c in _edit
            ),
            unsafe_allow_html=True,
        )

        _cpn1, _cpn2 = st.columns([3, 1])
        with _cpn1:
            _cp_name = st.text_input("Nome della tua palette", value=ss.get("cp_name", "La mia palette"),
                                      key="cp_name")
        with _cpn2:
            st.markdown("")
            if st.button("💾  Salva", key="cp_save", use_container_width=True):
                if not _cp_name.strip():
                    st.warning("Dai un nome alla palette prima di salvarla.")
                else:
                    ss.setdefault("custom_palettes", {})[_cp_name.strip()] = list(_edit)
                    ss.opt_palette_nm = _cp_name.strip()
                    st.success(f"Palette '{_cp_name.strip()}' salvata e selezionata! "
                               f"La trovi in cima al menu 'Palette colori' qui sopra.")
                    st.rerun()
        st.caption(
            "⚠️ Le palette personalizzate durano per questa sessione del browser: "
            "se chiudi e riapri l'app dovrai ricrearle."
        )

    # ── 📝 Testi ──────────────────────────────────────────────────────────────
    with st.expander("📝  Titolo ed etichette assi", expanded=True):
        ss.opt_title = st.text_input("Titolo grafico", value=ss.opt_title)
        tc1, tc2, tc3 = st.columns([2, 1, 1])
        with tc1:
            ss.opt_title_sz   = int(st.number_input("Dim. titolo", min_value=6, max_value=36,
                                                     value=int(ss.opt_title_sz), step=1))
        with tc2:
            ss.opt_title_bold = st.checkbox("Grassetto", value=ss.opt_title_bold, key="_tb")
        with tc3:
            ss.opt_title_ital = st.checkbox("Corsivo",   value=ss.opt_title_ital, key="_ti")

        tp1, tp2 = st.columns([1, 2])
        with tp1:
            _tloc_opts = ["Sinistra", "Centro", "Destra"]
            ss.opt_title_loc = st.selectbox("Posizione titolo", _tloc_opts,
                index=_tloc_opts.index(ss.opt_title_loc) if ss.opt_title_loc in _tloc_opts else 1)
        with tp2:
            ss.opt_title_off_y = st.slider("Scost. verticale titolo", -0.15, 0.15,
                                            float(ss.opt_title_off_y), 0.01,
                                            help="Sposta il titolo su/giù rispetto alla posizione di default.")

        st.markdown("---")
        xa1, xa2 = st.columns(2)
        with xa1:
            ss.opt_xlabel = st.text_input("Etichetta Asse X", value=ss.opt_xlabel)
        with xa2:
            ss.opt_ylabel = st.text_input("Etichetta Asse Y", value=ss.opt_ylabel)

        st.caption("Scostamento etichette assi (sposta il testo rispetto alla posizione di default)")
        xo1, xo2, xo3, xo4 = st.columns(4)
        with xo1:
            ss.opt_xlabel_off_x = st.slider("Asse X · Scost. X", -0.3, 0.3,
                                            float(ss.opt_xlabel_off_x), 0.02, key="_xlox")
        with xo2:
            ss.opt_xlabel_off_y = st.slider("Asse X · Scost. Y", -0.3, 0.3,
                                            float(ss.opt_xlabel_off_y), 0.02, key="_xloy")
        with xo3:
            ss.opt_ylabel_off_x = st.slider("Asse Y · Scost. X", -0.3, 0.3,
                                            float(ss.opt_ylabel_off_x), 0.02, key="_ylox")
        with xo4:
            ss.opt_ylabel_off_y = st.slider("Asse Y · Scost. Y", -0.3, 0.3,
                                            float(ss.opt_ylabel_off_y), 0.02, key="_yloy")

        fsz1, fsz2, fsz3, fsz4 = st.columns(4)
        with fsz1:
            ss.opt_label_sz  = int(st.number_input("Dim. etich. assi", min_value=6, max_value=24,
                                                    value=int(ss.opt_label_sz), step=1))
        with fsz2:
            ss.opt_xtick_sz  = int(st.number_input("Dim. tick X", min_value=4, max_value=20,
                                                    value=int(ss.opt_xtick_sz), step=1))
        with fsz3:
            ss.opt_ytick_sz  = int(st.number_input("Dim. tick Y", min_value=4, max_value=20,
                                                    value=int(ss.opt_ytick_sz), step=1))
        with fsz4:
            ss.opt_legend_sz = int(st.number_input("Dim. legenda", min_value=4, max_value=32,
                                                    value=int(ss.opt_legend_sz), step=1,
                                                    help="Nei grafici interattivi (Plotly) anche il "
                                                         "riquadro colore della legenda si ingrandisce "
                                                         "in proporzione al testo."))

    # ── 📝 Testi aggiuntivi (annotazioni libere) ───────────────────────────────
    with st.expander("✏️  Testi aggiuntivi", expanded=False):
        st.caption("Aggiungi scritte libere in qualunque punto del grafico (es. note, frecce testuali, "
                  "etichette custom) — a differenza del trascinamento col mouse, questi testi sono "
                  "salvati e compaiono anche in PNG/SVG scaricati.")
        for _i, _ann in enumerate(list(ss.opt_extra_texts)):
            ac1, ac2 = st.columns([5, 1])
            with ac1:
                st.caption(f"“{_ann['text']}” — x={_ann['x']:.2f}, y={_ann['y']:.2f}, dim={_ann['size']}")
            with ac2:
                if st.button("🗑", key=f"rm_ann_{_i}"):
                    ss.opt_extra_texts.pop(_i)
                    st.rerun()
        with st.form("add_annotation_form", clear_on_submit=True):
            new_text = st.text_input("Testo da aggiungere")
            nc1, nc2, nc3 = st.columns(3)
            with nc1:
                new_x = st.slider("Posizione X (0=sinistra, 1=destra)", 0.0, 1.0, 0.5, 0.02)
            with nc2:
                new_y = st.slider("Posizione Y (0=basso, 1=alto)", 0.0, 1.0, 0.9, 0.02)
            with nc3:
                new_sz = st.number_input("Dim. testo", min_value=6, max_value=40, value=12, step=1)
            if st.form_submit_button("➕ Aggiungi testo"):
                if new_text.strip():
                    ss.opt_extra_texts = list(ss.opt_extra_texts) + [
                        {"text": new_text.strip(), "x": float(new_x), "y": float(new_y), "size": int(new_sz)}]
                    st.rerun()

    # ── 📐 Assi e scala ───────────────────────────────────────────────────────
    with st.expander("📐  Assi e scala", expanded=False):
        st.caption("Per chi vuole di più: scala logaritmica, limiti min/max, rotazione etichette. "
                   "Per la maggior parte dei grafici non serve toccare nulla qui.")
        # Scala
        _sc_keys = list(_SCALE_TYPES)
        sca1, sca2 = st.columns(2)
        with sca1:
            _yi = _sc_keys.index(ss.opt_y_scale) if ss.opt_y_scale in _sc_keys else 0
            ss.opt_y_scale = st.selectbox("Scala Asse Y", _sc_keys, index=_yi)
        with sca2:
            _xi = _sc_keys.index(ss.opt_x_scale) if ss.opt_x_scale in _sc_keys else 0
            ss.opt_x_scale = st.selectbox("Scala Asse X", _sc_keys, index=_xi)

        # Limiti e passo
        lc1, lc2, lc3, lc4 = st.columns(4)
        with lc1:
            ss.opt_y_min  = st.text_input("Y min", value=ss.opt_y_min)
        with lc2:
            ss.opt_y_max  = st.text_input("Y max", value=ss.opt_y_max)
        with lc3:
            ss.opt_y_step = st.text_input("Y passo tick", value=ss.opt_y_step,
                                           help="Lascia vuoto per automatico")
        with lc4:
            ss.opt_x_step = st.text_input("X passo tick", value=ss.opt_x_step,
                                           help="Lascia vuoto per automatico")

        # Minor tick
        st.markdown("**Minor tick**")
        mt1, mt2, mt3, mt4 = st.columns(4)
        with mt1:
            ss.opt_y_minor = st.checkbox("Minor tick Y", value=ss.opt_y_minor)
        with mt2:
            ss.opt_y_minor_ndiv = int(st.number_input("Subdiv Y", min_value=2, max_value=20,
                                                        value=int(ss.opt_y_minor_ndiv), step=1))
        with mt3:
            ss.opt_x_minor = st.checkbox("Minor tick X", value=ss.opt_x_minor)
        with mt4:
            ss.opt_x_minor_ndiv = int(st.number_input("Subdiv X", min_value=2, max_value=20,
                                                        value=int(ss.opt_x_minor_ndiv), step=1))

        # Rotazione etichette X
        ss.opt_xtick_rot = int(st.slider(
            "Rotazione etichette X  (−1 = auto)",
            min_value=-1, max_value=90, value=int(ss.opt_xtick_rot), step=5,
        ))

    # ── 🔲 Griglia avanzata ────────────────────────────────────────────────────
    with st.expander("🔲  Griglia avanzata", expanded=False):
        st.caption("Se abilitata, sovrascrive l'opzione 'Griglia' dello stile.")
        _gs_opts = list(_GRID_STYLE_MAP.keys())

        st.markdown("**Asse Y — major**")
        gy1, gy2, gy3, gy4 = st.columns([1, 2, 1, 1])
        with gy1:
            ss.opt_gy_maj = st.checkbox("Attiva##gym", value=ss.opt_gy_maj)
        with gy2:
            _gi = _gs_opts.index(ss.opt_gy_maj_sty) if ss.opt_gy_maj_sty in _gs_opts else 1
            ss.opt_gy_maj_sty = st.selectbox("Stile##gym", _gs_opts, index=_gi)
        with gy3:
            ss.opt_gy_maj_alp = float(st.number_input("Alpha##gym", 0.05, 1.0,
                                                        value=float(ss.opt_gy_maj_alp), step=0.05))
        with gy4:
            ss.opt_gy_maj_lw  = float(st.number_input("Spess.##gym", 0.2, 3.0,
                                                        value=float(ss.opt_gy_maj_lw), step=0.2))

        st.markdown("**Asse Y — minor**")
        gy5, gy6, gy7, gy8, gy9 = st.columns([1, 2, 1, 1, 1])
        with gy5:
            ss.opt_gy_min = st.checkbox("Attiva##gymi", value=ss.opt_gy_min)
        with gy6:
            _gi2 = _gs_opts.index(ss.opt_gy_min_sty) if ss.opt_gy_min_sty in _gs_opts else 2
            ss.opt_gy_min_sty = st.selectbox("Stile##gymi", _gs_opts, index=_gi2)
        with gy7:
            ss.opt_gy_min_alp = float(st.number_input("Alpha##gymi", 0.05, 1.0,
                                                        value=float(ss.opt_gy_min_alp), step=0.05))
        with gy8:
            ss.opt_gy_min_lw  = float(st.number_input("Spess.##gymi", 0.2, 3.0,
                                                        value=float(ss.opt_gy_min_lw), step=0.2))
        with gy9:
            ss.opt_gy_min_ndiv = int(st.number_input("Subdiv##gymi", 2, 20,
                                                       value=int(ss.opt_gy_min_ndiv), step=1))

        st.markdown("**Asse X — major**")
        gx1, gx2, gx3, gx4 = st.columns([1, 2, 1, 1])
        with gx1:
            ss.opt_gx_maj = st.checkbox("Attiva##gxm", value=ss.opt_gx_maj)
        with gx2:
            _gi3 = _gs_opts.index(ss.opt_gx_maj_sty) if ss.opt_gx_maj_sty in _gs_opts else 1
            ss.opt_gx_maj_sty = st.selectbox("Stile##gxm", _gs_opts, index=_gi3)
        with gx3:
            ss.opt_gx_maj_alp = float(st.number_input("Alpha##gxm", 0.05, 1.0,
                                                        value=float(ss.opt_gx_maj_alp), step=0.05))
        with gx4:
            ss.opt_gx_maj_lw  = float(st.number_input("Spess.##gxm", 0.2, 3.0,
                                                        value=float(ss.opt_gx_maj_lw), step=0.2))

        st.markdown("**Asse X — minor**")
        gx5, gx6, gx7, gx8, gx9 = st.columns([1, 2, 1, 1, 1])
        with gx5:
            ss.opt_gx_min = st.checkbox("Attiva##gxmi", value=ss.opt_gx_min)
        with gx6:
            _gi4 = _gs_opts.index(ss.opt_gx_min_sty) if ss.opt_gx_min_sty in _gs_opts else 2
            ss.opt_gx_min_sty = st.selectbox("Stile##gxmi", _gs_opts, index=_gi4)
        with gx7:
            ss.opt_gx_min_alp = float(st.number_input("Alpha##gxmi", 0.05, 1.0,
                                                        value=float(ss.opt_gx_min_alp), step=0.05))
        with gx8:
            ss.opt_gx_min_lw  = float(st.number_input("Spess.##gxmi", 0.2, 3.0,
                                                        value=float(ss.opt_gx_min_lw), step=0.2))
        with gx9:
            ss.opt_gx_min_ndiv = int(st.number_input("Subdiv##gxmi", 2, 20,
                                                       value=int(ss.opt_gx_min_ndiv), step=1))

    # ── 🔤 Etichette dati ────────────────────────────────────────────────────
    with st.expander("🔤  Etichette dati (numeri su barre)", expanded=False):
        ss.opt_data_lbl_sz = int(st.number_input(
            "Grandezza font etichette", min_value=4, max_value=20,
            value=int(ss.opt_data_lbl_sz), step=1,
        ))

    # ── 💾 Esportazione ───────────────────────────────────────────────────────
    with st.expander("💾  Esportazione immagine", expanded=False):
        st.caption("I bottoni 'Scarica' sotto al grafico salvano in PNG, JPG o SVG "
                   "(SVG è vettoriale: qualità perfetta a ogni zoom/stampa).")
        ec1, ec2, ec3 = st.columns(3)
        with ec1:
            ss.opt_fig_w = float(st.number_input("Larghezza (pollici)", min_value=4.0,
                                                   max_value=24.0, value=float(ss.opt_fig_w), step=0.5))
        with ec2:
            ss.opt_fig_h = float(st.number_input("Altezza (pollici)", min_value=3.0,
                                                   max_value=20.0, value=float(ss.opt_fig_h), step=0.5))
        with ec3:
            ss.opt_dpi   = int(st.number_input("DPI", min_value=72, max_value=600,
                                                value=int(ss.opt_dpi), step=50,
                                                help="Ignorato per il formato SVG (vettoriale)."))

    st.divider()
    rc1, rc2 = st.columns(2)
    with rc1:
        if st.button("🔄  Ripristina default", use_container_width=True):
            for _k, _v in _OPT_DEFAULTS.items():
                st.session_state[_k] = _v
            st.rerun()
    with rc2:
        if st.button("📊  Vai al Grafico →", use_container_width=True):
            st.session_state.active_tab = 2
            st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# PAGINA 5 — ANIMAZIONE
# Scompone l'ultimo grafico generato in CELLE (il singolo dato più fine: un
# punto/una barra/una fetta, mai una serie intera in blocco) e le rivela in
# sequenza secondo un ordine scelto (per serie, per posizione X, a cascata
# per valore...), con controllo separato su COSA compare insieme al dato
# (etichetta asse, valore numerico, voce di legenda) — vedi chart_animation.py.
# ══════════════════════════════════════════════════════════════════════════════
elif _sel == _TABS[4]:
    st.subheader("🎬  Animazione")
    if st.session_state.fig is None or not isinstance(st.session_state.fig, go.Figure):
        st.info("Genera prima un grafico nella tab '📊 Grafico'. Solo i tipi sul motore "
                "Plotly sono animabili per ora: barre, barre impilate (2D), linee, "
                "dispersione, istogramma, torta, boxplot, violino.")
    else:
        _fig0 = st.session_state.fig
        _cells = CA.list_cells(_fig0)
        if not _cells:
            st.warning("Nessun elemento animabile trovato in questo grafico.")
        else:
            _ckey = tuple(c["id"] for c in _cells)
            if st.session_state.get("anim_ckey") != _ckey:
                st.session_state.anim_ckey = _ckey
                st.session_state.anim_excl_series = []
                st.session_state.anim_order = "dati"

            _has_series = CA.has_series(_cells)
            _series = CA.series_names(_cells)

            st.caption("Scegli come, in che ordine e cosa compare insieme a ciascun dato — "
                      "l'ultimo fotogramma coincide sempre con il grafico che vedi nella tab "
                      "'Grafico', qualunque opzione scegli qui sotto.")

            # Solo gli stili che hanno davvero senso per QUESTO tipo di
            # grafico (vedi CA.STYLE_COMPAT) — es. 'cubetti'/'acqua'/
            # 'sipario' non sono offerti per un grafico a linee (la
            # 'colonna fantasma' che usano per barre/dispersione ci
            # comparirebbe come una barra estranea, segnalato dall'utente).
            _chart_key_now = (st.session_state.last_chart_pkg or {}).get("chart_type")
            _style_opts = CA.styles_for_chart(_chart_key_now)
            if st.session_state.get("anim_style") not in _style_opts:
                st.session_state.anim_style = _style_opts[0]
            anim_style = st.selectbox(
                "🎞  Tipo di animazione", _style_opts,
                format_func=lambda k: CA.ANIMATION_STYLES[k], key="anim_style")

            c1, c2 = st.columns([2, 1])
            with c1:
                _order_opts = [k for k in CA.ORDER_MODES if k != "categoria" or _has_series]
                order_mode = st.selectbox(
                    "Ordine di comparsa", _order_opts,
                    format_func=lambda k: CA.ORDER_MODES[k], key="anim_order")
            with c2:
                _spot_opts = list(CA.SPOTLIGHT_MODES.keys())
                spotlight_mode = st.selectbox(
                    "Evidenzia il dato corrente", _spot_opts,
                    format_func=lambda k: CA.SPOTLIGHT_MODES[k], key="anim_spotlight_mode")

            direction_deg = 0.0
            if order_mode == "direzione":
                _dir_opts = list(CA.DIRECTIONS.keys())
                _dir_choice = st.selectbox(
                    "Direzione del fronte (16 punti cardinali)", _dir_opts,
                    key="anim_direction",
                    help="Le celle si rivelano in ordine di posizione lungo questa direzione — "
                         "'da sinistra' rivela prima quelle con X più piccola, 'dall'alto' "
                         "prima quelle con Y più grande, e così per le altre 12.")
                direction_deg = CA.DIRECTIONS[_dir_choice]

            st.markdown("**Cosa compare insieme al dato**")
            wcols = st.columns(3)
            reveal_x = wcols[0].checkbox("La sua etichetta (asse)", value=False, key="anim_reveal_x")
            reveal_v = wcols[1].checkbox("Il suo valore numerico", value=False, key="anim_reveal_v")
            reveal_l = wcols[2].checkbox("La voce di legenda della sua serie", value=_has_series,
                                          key="anim_reveal_l", disabled=not _has_series)

            excl: list[str] = []
            if _has_series:
                excl = st.multiselect(
                    "Serie da escludere dall'animazione (restano comunque visibili nel "
                    "grafico finale, ma senza un loro momento dedicato)",
                    _series, key="anim_excl_series")

            st.divider()
            duration_s = st.slider(
                "⏱  Durata totale dell'animazione (secondi)", 3, 40,
                value=int(st.session_state.get("anim_duration_s", 10)),
                key="anim_duration_s",
                help="Quanto dura un giro completo, dall'inizio alla fine. Il numero di "
                     "fotogrammi si calcola da qui — nessun altro comando indiretto da "
                     "combinare a mano.")

            # Ricalcolata a ogni rerun (economico, puro Python) da quello che è
            # selezionato ORA nei controlli sopra — cambiare stile/ordine/spunte
            # si vede subito, senza dover premere un bottone per "aggiornare".
            included_cells = [c for c in _cells if c["series"] not in excl]
            steps = CA.build_steps(included_cells, order_mode, direction_deg=direction_deg)
            n_steps = max(1, len(steps))
            # Un solo comando in secondi invece di due sliders indiretti
            # (conteggio fotogrammi + ms-per-fotogramma) che, combinati male,
            # potevano produrre un'animazione che finiva quasi all'istante —
            # qui la durata TOTALE è l'unica cosa scelta dall'utente; quanti
            # fotogrammi servano per starci dentro con un passo fluido
            # (~80ms/fotogramma) è un dettaglio calcolato, non da impostare.
            _TARGET_FRAME_MS = 80
            subframes = max(2, round((duration_s * 1000) / (n_steps * _TARGET_FRAME_MS)))
            frame_ms = max(30, round((duration_s * 1000) / (n_steps * subframes)))
            frames, extra_traces = CA.build_frames(
                _fig0, steps, style=anim_style, reveal_x_label=reveal_x,
                reveal_value=reveal_v, reveal_legend=reveal_l, spotlight_mode=spotlight_mode,
                subframes_per_step=subframes)

            dur = CA.suggested_duration_s(frames, frame_ms)
            st.caption(f"▶ Play la avvia, ⏸ Pausa la ferma dove si trova, ⏹ Stop la riporta "
                      f"all'inizio ferma (utile dopo aver cambiato un'opzione sopra, per "
                      f"rivederla da capo con i parametri nuovi) — un giro completo dura "
                      f"~{dur}s. In Pipeline Video invece l'HTML scaricato parte da solo e fa "
                      "un solo passaggio, restando ferma sul grafico completo alla fine — lì "
                      "non serve nessun bottone.")

            preview_fig = CA.prepare_animated_figure(_fig0, frames, extra_traces)
            preview_fig.update_layout(updatemenus=[dict(
                # showactive=True: Plotly evidenzia da solo il bottone
                # premuto per ultimo (bordo/sfondo diversi) — così si vede
                # sempre in che stato è l'animazione senza doverlo dedurre.
                type="buttons", direction="left", showactive=True,
                x=0.0, y=1.12, xanchor="left", yanchor="top",
                bgcolor="#F3F1FA", bordercolor="#D7D7DE", borderwidth=1,
                active=-1,
                font=dict(size=13, color="#312A74"),
                buttons=[
                    dict(label="▶  Play", method="animate",
                         args=[None, {"frame": {"duration": frame_ms, "redraw": True},
                                       "transition": {"duration": 0},
                                       "fromcurrent": True, "mode": "immediate"}]),
                    dict(label="⏸  Pausa", method="animate",
                         args=[[None], {"frame": {"duration": 0, "redraw": False},
                                        "mode": "immediate", "transition": {"duration": 0}}]),
                    dict(label="⏹  Stop", method="animate",
                         args=[["0"], {"frame": {"duration": 0, "redraw": True},
                                       "mode": "immediate", "transition": {"duration": 0}}]),
                ])])
            # key= che cambia con ogni parametro che rifà i frame: forza
            # Streamlit a rimontare da zero il componente Plotly invece di
            # riconciliarlo (Plotly.react) sopra quello esistente — SENZA
            # questo, cambiare un'opzione lasciava lo stato dell'animazione
            # precedente "incollato" al nuovo grafico (es. la prima barra
            # partiva già a 3/4 invece che da 0, ereditato da un play/pausa
            # precedente). A parità di parametri la key resta la stessa, quindi
            # un rerun "innocente" (un widget non correlato altrove in
            # pagina) continua a riusare lo stesso componente e non la fa
            # ripartire da capo. 'duration_s' (non subframes/frame_ms, che ne
            # sono solo la traduzione interna) è la variabile che conta qui.
            _preview_key = "animprev_" + "_".join(str(v) for v in (
                _ckey, order_mode, direction_deg, anim_style, reveal_x, reveal_v, reveal_l,
                spotlight_mode, duration_s, tuple(sorted(excl))))
            st.plotly_chart(preview_fig, use_container_width=True, key=_preview_key)

            _download_html = CA.build_autoplay_html(_fig0, frames, frame_ms, loop=False,
                                                      extra_traces=extra_traces)
            st.download_button(
                label="📤  Scarica animazione HTML per Pipeline Video",
                data=_download_html,
                file_name="animazione_grafico.html",
                mime="text/html",
                use_container_width=True,
            )
