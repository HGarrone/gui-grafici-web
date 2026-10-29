"""Motore di animazione per la tab "Animazione" di GUI Grafici semplificata.

Non ricostruisce dati/stile: opera direttamente sull'oggetto go.Figure già
generato dalla Pagina "Grafico" (st.session_state.fig). Scompone il grafico
in CELLE — il singolo dato più fine possibile (un punto/una barra/una fetta,
mai una serie intera in blocco) — così l'utente può scegliere SEPARATAMENTE:
1) COSA compare insieme al dato (solo il dato? anche la sua etichetta
   sull'asse "categorico"? il suo valore numerico? la voce di legenda della
   sua serie, al primo dato che la introduce?);
2) in CHE ORDINE le celle si rivelano (per serie: prima l'intera serie A poi
   la B; per posizione: tutte le serie alla stessa categoria insieme, poi la
   categoria successiva; a cascata per valore, decrescente o crescente;
   oppure l'ordine grezzo dei dati).

Il risultato è una sequenza di "passi" (ogni passo = una o più celle rivelate
insieme, es. tutte le serie alla stessa categoria X nell'ordine "posizione")
trasformata in fig.frames Plotly: i passi già passati restano pieni, quello
corrente si rivela con una piccola easing, i successivi restano nello stato
iniziale — fino all'ULTIMO frame, che coincide esattamente col grafico
statico di partenza.

L'output è un HTML autoplay auto-contenuto (Plotly.js inline, nessun CDN)
pensato per essere caricato in Pipeline Video come "File pronto" e catturato
in video con la pipeline Playwright già esistente lì (core/assemble.py:
capture_html) — nessuna assunzione oltre "l'animazione parte da sola al
caricamento della pagina".
"""
from __future__ import annotations

import math
from typing import Any

import plotly.graph_objects as go

ORDER_MODES = {
    "categoria": "Per serie: prima tutta la serie A (con la sua legenda), poi la B...",
    "posizione": "Per posizione: tutte le serie alla stessa categoria X insieme, poi la prossima",
    "valore_desc": "A cascata: dal valore più alto al più basso",
    "valore_asc": "A cascata: dal valore più basso al più alto",
    "dati": "Ordine dati (come nel grafico, nessun riordino)",
    "insieme": "Tutti insieme: nessuna sequenza, un solo passo simultaneo",
    "direzione": "Direzione: spazza da un lato all'altro (scegli da dove, qui sotto)",
}

# 16 punti cardinali — l'angolo è la direzione VERSO CUI avanza il fronte
# (0°=verso destra quindi parte da sinistra/min-x, 90°=verso l'alto quindi
# parte dal basso/min-y, e così via). Copre "da min x/min y/max x/max y" +
# le altre 12 direzioni richieste con un solo meccanismo generico invece di
# 16 casi speciali.
DIRECTIONS = {
    "→  Da sinistra (min X)":      0.0,
    "↗  Da basso-sinistra":       45.0,
    "↑  Dal basso (min Y)":       90.0,
    "↖  Da basso-destra":        135.0,
    "←  Da destra (max X)":      180.0,
    "↙  Da alto-destra":         225.0,
    "↓  Dall'alto (max Y)":      270.0,
    "↘  Da alto-sinistra":       315.0,
    "→  Da sinistra (fine)":      22.5,
    "↗  ENE":                     67.5,
    "↖  NNO":                    112.5,
    "←  OSO":                    157.5,
    "↙  SSO":                    202.5,
    "↓  SSE (fine)":             247.5,
    "↘  ESE":                    292.5,
    "→  ENE (fine)":             337.5,
}

ANIMATION_STYLES = {
    "crescita": "Crescita — le barre crescono da 0, le fette si aprono, i punti si ingrandiscono",
    "rimbalzo": "Rimbalzo — cresce oltre il valore finale e torna indietro con un piccolo balzo",
    "dissolvenza": "Dissolvenza — il dato compare già alla sua dimensione, in trasparenza crescente",
    "istantanea": "Comparsa istantanea — il dato scatta di colpo alla sua dimensione piena",
    "fiorisce": "Fiorisce — crescita organica con un piccolo respiro elastico, più morbida del rimbalzo",
    "stampa": "Stampa — scatta pieno con un lampo di luce nell'istante in cui compare",
    "particelle": "Particelle — compare con una piccola esplosione di scintille nel suo colore",
    "verniciata": "Verniciata — il colore si stende come una mano di vernice: lucida appena data, si opacizza mentre 'asciuga'",
    "cubetti": "Cubetti — cresce a scatti, come tanti cubetti che si impilano uno sull'altro fino al valore finale",
    "acqua": "Acqua — la barra è un contenitore trasparente che si riempie di liquido colorato fino all'orlo",
    "linea": "Linea che cammina — solo per grafici a linee: il tratto si allunga con continuità da un punto al successivo",
    "sipario": "Sipario — il dato è già lì a piena dimensione, un sipario si apre da centro per rivelarlo",
}

_GENERIC_STYLES = ("crescita", "rimbalzo", "dissolvenza", "istantanea", "fiorisce",
                    "stampa", "particelle", "verniciata")

# Quali stili offrire per ciascun tipo di grafico ("chart_key" della pagina
# Grafico, non il tipo di traccia Plotly) — non tutte le combinazioni
# style×tipo hanno un risultato genuinamente diverso o corretto: meglio
# offrire di meno ma che funzioni davvero, invece di mostrare stili che
# sull'uno o sull'altro tipo non fanno nulla di distinguibile o si
# comportano in modo scorretto (segnalato dall'utente per barra fantasma
# su un grafico a linee — 'cubetti'/'acqua' presumono un punto come una
# barra indipendente, sensato per la dispersione ma non per una linea
# continua, dove la 'colonna fantasma' compariva come un rettangolo
# estraneo in fondo al grafico). 'sipario' invece è un'unica dissolvenza
# sull'INTERO grafico a coordinate di pagina (non per singolo dato), quindi
# funziona identico su ogni tipo — disponibile ovunque.
STYLE_COMPAT: dict[str, tuple[str, ...]] = {
    "bar":          _GENERIC_STYLES + ("cubetti", "acqua", "sipario"),
    "bar_stacked":  _GENERIC_STYLES + ("cubetti", "acqua", "sipario"),
    "histogram":    _GENERIC_STYLES + ("cubetti", "acqua", "sipario"),
    "line":         _GENERIC_STYLES + ("linea", "sipario"),
    "scatter":      _GENERIC_STYLES + ("cubetti", "acqua", "sipario"),
    "pie":          _GENERIC_STYLES + ("sipario",),
    # cubetti/acqua non offerti: userebbero la stessa 'colonna fantasma
    # rettangolare' delle barre, che stonerebbe contro i lati OBLIQUI di un
    # trapezio funnel — non verificato visivamente, meglio non offrirlo
    # finché non lo è (stessa cautela già usata per boxplot/violino).
    "funnel":       _GENERIC_STYLES + ("sipario",),
    # 'dissolvenza' esclusa qui: il poligono radar non ha un'opacità per
    # vertice (solo di trace intera), quindi si comporta come 'istantanea'
    # senza dare un risultato visivamente distinto — offrirla creerebbe
    # confusione (due opzioni, stesso identico risultato).
    "radar":        tuple(s for s in _GENERIC_STYLES if s != "dissolvenza") + ("cubetti", "acqua", "sipario"),
    # boxplot/violino supportano SOLO comparsa/sparizione dell'intera forma
    # (non un valore singolo da far crescere/schiarire/lampeggiare — vedi
    # _trace_frame_update) — offrire gli 8 stili generici sarebbe
    # fuorviante, sembrerebbero tutti diversi ma darebbero lo STESSO
    # risultato identico. Solo le opzioni che riflettono onestamente cosa
    # succede davvero — 'sipario' incluso, non ha bisogno di un valore per
    # elemento.
    "boxplot":      ("istantanea", "sipario"),
    "violin":       ("istantanea", "sipario"),
}


def styles_for_chart(chart_key: str | None) -> list[str]:
    """Elenco stili sensati per questo tipo di grafico, nell'ordine di
    ANIMATION_STYLES — fallback ai soli generici per un chart_key ignoto
    (più sicuro che mostrare tutto)."""
    allowed = STYLE_COMPAT.get(chart_key or "", _GENERIC_STYLES)
    return [k for k in ANIMATION_STYLES if k in allowed]


SPOTLIGHT_MODES = {
    "bagliore": "Bagliore — il dato attivo si schiarisce, fisso finché resta il suo turno",
    "lampeggio": "Lampeggio — il colore alterna acceso/normale un paio di volte mentre è il turno del dato",
    "impulso": "Impulso — bordo e dimensione pulsano ritmicamente, colore invariato",
    "doppio": "Doppio — lampeggio di colore + impulso di dimensione insieme",
    "nessuna": "Nessuna evidenziazione",
}


def _arr(v) -> list:
    """Converte in list un attributo Plotly (può essere una tupla, un
    ndarray numpy o None) — MAI valutarlo con `or` (un array numpy con più
    elementi rompe la truthiness: 'the truth value of an array... is
    ambiguous')."""
    return list(v) if v is not None else []


def _fmt_value(v) -> str:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    if f == int(f):
        return str(int(f))
    return f"{f:.1f}"


def _value_text(value, reveal_value: bool) -> str:
    """Testo del valore numerico che compare accanto al dato (bottone 'Il
    suo valore numerico'). L'etichetta di categoria ('La sua etichetta
    (asse)') invece NON passa più da qui: tocca il vero asse Plotly (vedi
    _reveal_axes/_axis_layout_update) — un'etichetta di categoria appartiene
    all'asse, non a un testo fluttuante sopra il dato."""
    return _fmt_value(value) if reveal_value else ""


def _lighten(hex_color, amount: float = 0.65) -> str:
    """Schiarisce un colore hex verso il bianco — usato per 'illuminare' il
    dato attivo nel SUO STESSO colore (effetto fluo/glow), invece di un
    bordo rosso fisso scollegato dalla palette del grafico. Se il colore di
    partenza è già chiaro (es. un giallo/oro pallido di palette), lo stesso
    schiarimento lo spinge fin quasi al bianco — un punto attivo diventava
    invisibile a schermo (bug segnalato dall'utente: 'non si vede nulla' era
    davvero solo il primissimo dato, minuscolo E quasi bianco insieme).
    L'intensità effettiva si riduce quando la luminanza di partenza è già
    alta, così un colore chiaro resta comunque distinguibile dal bianco."""
    s = str(hex_color or "#4C72B0").lstrip("#")
    if len(s) != 6:
        return "#FFFFFF"
    try:
        r, g, b = int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16)
    except ValueError:
        return "#FFFFFF"
    lum = 0.299 * r + 0.587 * g + 0.114 * b
    if lum > 140:
        amount = amount * max(0.15, 1 - (lum - 140) / 140)
    r = int(r + (255 - r) * amount)
    g = int(g + (255 - g) * amount)
    b = int(b + (255 - b) * amount)
    return f"#{r:02X}{g:02X}{b:02X}"


def _bar_base_colors(tr, n: int) -> list[str]:
    mc = tr.marker.color if tr.marker else None
    if isinstance(mc, (list, tuple)):
        out = [str(c) if c else "#4C72B0" for c in mc]
        return (out + ["#4C72B0"] * n)[:n]
    return [str(mc) if mc else "#4C72B0"] * n


def _pie_base_colors(tr, n: int) -> list[str]:
    mc = tr.marker.colors if tr.marker else None
    if isinstance(mc, (list, tuple)):
        out = [str(c) if c else "#4C72B0" for c in mc]
        return (out + ["#4C72B0"] * n)[:n]
    return ["#4C72B0"] * n


def _scatter_base_color(tr) -> str:
    if tr.marker and tr.marker.color and not isinstance(tr.marker.color, (list, tuple)):
        return str(tr.marker.color)
    if tr.line and tr.line.color:
        return str(tr.line.color)
    return "#4C72B0"


def _is_pure_scatter(tr) -> bool:
    """True solo se la traccia è punti SENZA linea che li collega (mode
    'markers', non 'lines+markers'/'lines') — 'cubetti'/'acqua'/'sipario'
    trattano ogni punto come una barra fantasma dalla baseline 0 al suo
    valore, cosa che ha senso per punti sparsi ma non per un grafico a
    linee: lì il segmento continuo è già la rappresentazione corretta, e la
    'colonna fantasma' comparirebbe come una barra estranea in fondo al
    grafico (bug segnalato dall'utente) — su quelli restano disponibili
    solo gli stili generici e 'linea'."""
    mode = str(getattr(tr, "mode", None) or "")
    return "lines" not in mode


# ── Scomposizione in celle ──────────────────────────────────────────────────

def _try_float(v, fallback: float) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return fallback


def list_cells(fig: go.Figure) -> list[dict]:
    """Una cella per ogni dato singolo (mai una serie intera in blocco).
    Ogni cella porta: id univoco, trace_idx/elem_idx (dove scrivere
    l'aggiornamento), series (nome serie/legenda di appartenenza — None se
    il grafico non ha serie multiple), x_key (la sua categoria/ascissa, come
    stringa — usata per l'ordine "posizione" e per rivelare i tick asse),
    value (numero — usato per l'ordine a cascata e per il valore rivelabile),
    pos_x/pos_y (posizione numerica sul grafico — usata SOLO dall'ordine
    "direzione", una proiezione geometrica: barra/punto = indice e valore,
    torta/radar = coordinate sul cerchio secondo l'angolo della fetta/vertice,
    così "da sinistra"/"dall'alto"/ecc. ha senso anche lì), legend_label
    (testo della voce di legenda associata, se applicabile)."""
    traces = fig.data
    multi = len(traces) > 1
    cells: list[dict] = []

    for ti, tr in enumerate(traces):
        if tr.type == "pie":
            labels = _arr(tr.labels)
            values = _arr(tr.values)
            total = sum(_try_float(v, 0.0) for v in values) or 1.0
            cum = 0.0
            for j, lbl in enumerate(labels):
                v = values[j] if j < len(values) else 0
                vf = _try_float(v, 0.0)
                angle = 2 * math.pi * (cum + vf / 2) / total
                cum += vf
                cells.append({
                    "id": f"t{ti}_{j}", "trace_idx": ti, "elem_idx": j,
                    "series": None, "x_key": str(lbl),
                    "value": float(v) if v is not None else 0.0,
                    "pos_x": math.cos(angle), "pos_y": math.sin(angle),
                    "legend_label": str(lbl),
                })

        elif tr.type == "box":
            ys = _arr(tr.y)
            med = sorted(float(v) for v in ys if v is not None)
            val = med[len(med) // 2] if med else 0.0
            name = str(tr.name) if tr.name else f"Box {ti + 1}"
            cells.append({
                "id": f"t{ti}", "trace_idx": ti, "elem_idx": None,
                "series": None, "x_key": name, "value": val,
                "pos_x": float(ti), "pos_y": val,
                "legend_label": name if multi else None,
            })

        elif tr.type == "violin":
            # Come 'box': un violino è definito da una stima di densità
            # (KDE) che Plotly calcola da sé nel browser dai dati grezzi —
            # niente valore singolo da far crescere, solo comparsa/sparizione
            # dell'intera forma come unità (vedi branch 'violin' in
            # _trace_frame_update).
            ys = _arr(tr.y)
            med = sorted(float(v) for v in ys if v is not None)
            val = med[len(med) // 2] if med else 0.0
            name = str(tr.name) if tr.name else f"Violino {ti + 1}"
            cells.append({
                "id": f"t{ti}", "trace_idx": ti, "elem_idx": None,
                "series": None, "x_key": name, "value": val,
                "pos_x": float(ti), "pos_y": val,
                "legend_label": name if multi else None,
            })

        elif tr.type == "bar":
            horiz = getattr(tr, "orientation", None) == "h"
            cats = _arr(tr.y if horiz else tr.x)
            vals = _arr(tr.x if horiz else tr.y)
            series_name = str(tr.name) if (multi and tr.name) else None
            for j, c in enumerate(cats):
                v = vals[j] if j < len(vals) else 0
                if v is None:
                    continue  # combinazione assente nelle barre raggruppate: nessun passo vuoto
                vf = float(v)
                px, py = (vf, float(j)) if horiz else (float(j), vf)
                cells.append({
                    "id": f"t{ti}_{j}", "trace_idx": ti, "elem_idx": j,
                    "series": series_name, "x_key": str(c),
                    "value": vf, "pos_x": px, "pos_y": py,
                    "legend_label": series_name,
                })

        elif tr.type == "funnel":
            # Sempre 'orizzontale' per come Plotly disegna un funnel:
            # categorie (livelli) su y, valori su x — stessa geometria di
            # una barra orizzontale, un solo valore per livello.
            cats = _arr(tr.y)
            vals = _arr(tr.x)
            series_name = str(tr.name) if (multi and tr.name) else None
            for j, c in enumerate(cats):
                v = vals[j] if j < len(vals) else 0
                vf = float(v) if v is not None else 0.0
                cells.append({
                    "id": f"t{ti}_{j}", "trace_idx": ti, "elem_idx": j,
                    "series": series_name, "x_key": str(c),
                    "value": vf, "pos_x": vf, "pos_y": float(j),
                    "legend_label": series_name,
                })

        elif tr.type == "scatter":
            xs = _arr(tr.x)
            ys = _arr(tr.y)
            series_name = str(tr.name) if (multi and tr.name) else None
            for j, xv in enumerate(xs):
                v = ys[j] if j < len(ys) else 0
                vf = float(v) if v is not None else 0.0
                cells.append({
                    "id": f"t{ti}_{j}", "trace_idx": ti, "elem_idx": j,
                    "series": series_name, "x_key": str(xv),
                    "value": vf, "pos_x": _try_float(xv, float(j)), "pos_y": vf,
                    "legend_label": series_name,
                })

        elif tr.type == "scatterpolar":
            # Radar: theta/r sono lunghi N+1 (l'ultimo punto ripete il primo
            # per richiudere il poligono) — quel punto di chiusura non è una
            # cella propria, segue sempre il vertice 0. pos_x/pos_y = punto
            # sul cerchio secondo la posizione angolare del vertice (stessa
            # logica delle fette di torta).
            thetas = _arr(tr.theta)
            rs = _arr(tr.r)
            n_real = max(0, len(thetas) - 1)
            series_name = str(tr.name) if (multi and tr.name) else None
            for j in range(n_real):
                v = rs[j] if j < len(rs) else 0
                angle = 2 * math.pi * j / max(n_real, 1)
                cells.append({
                    "id": f"t{ti}_{j}", "trace_idx": ti, "elem_idx": j,
                    "series": series_name, "x_key": str(thetas[j]),
                    "value": float(v) if v is not None else 0.0,
                    "pos_x": math.cos(angle), "pos_y": math.sin(angle),
                    "legend_label": series_name,
                })

    return cells


def has_series(cells: list[dict]) -> bool:
    return any(c["series"] is not None for c in cells)


def series_names(cells: list[dict]) -> list[str]:
    out = []
    for c in cells:
        if c["series"] is not None and c["series"] not in out:
            out.append(c["series"])
    return out


# ── Ordinamento in passi ────────────────────────────────────────────────────

def build_steps(cells: list[dict], mode: str, direction_deg: float = 0.0) -> list[list[dict]]:
    """Ritorna una lista di PASSI (ogni passo = una o più celle che si
    rivelano insieme nello stesso momento). "posizione", "insieme" e
    "direzione" raggruppano più celle nello stesso passo; gli altri modi
    rivelano una cella alla volta.

    "direzione": ordina per proiezione geometrica (pos_x, pos_y) lungo
    l'angolo scelto (0°=verso destra → parte da sinistra/min-x, 90°=verso
    l'alto → parte dal basso/min-y, ecc. — vedi DIRECTIONS) invece che per
    categoria o valore: un fronte che spazza il grafico in una direzione
    scelta liberamente, non solo le 4 cardinali."""
    if not cells:
        return []
    if mode == "insieme":
        return [list(cells)]
    if mode == "direzione":
        rad = math.radians(direction_deg)
        cx, sx = math.cos(rad), math.sin(rad)
        ordered = sorted(cells, key=lambda c: c.get("pos_x", 0.0) * cx + c.get("pos_y", 0.0) * sx)
        return [[c] for c in ordered]
    if mode == "valore_desc":
        return [[c] for c in sorted(cells, key=lambda c: -c["value"])]
    if mode == "valore_asc":
        return [[c] for c in sorted(cells, key=lambda c: c["value"])]
    if mode == "categoria":
        order = []
        for c in cells:
            if c["series"] not in order:
                order.append(c["series"])
        steps = []
        for s in order:
            steps += [[c] for c in cells if c["series"] == s]
        return steps
    if mode == "posizione":
        order = []
        for c in cells:
            if c["x_key"] not in order:
                order.append(c["x_key"])
        return [[c for c in cells if c["x_key"] == xk] for xk in order]
    return [[c] for c in cells]  # "dati"


# ── Costruzione dei frame ───────────────────────────────────────────────────

def _reveal_axes(fig: go.Figure) -> list[str]:
    """Asse/i Plotly su cui vivono le 'categorie' (x_key) delle celle — una
    funzione GENERICA che restituisce quale asse è quello "indipendente"
    invece di assumere sempre xaxis: xaxis normalmente, yaxis per le barre
    orizzontali (l'asse col nome delle categorie si scambia con
    l'orientamento), entrambi per un domani un tipo con due assi categorici
    (es. heatmap). Un solo punto da estendere quando arriva un nuovo tipo,
    invece di casi speciali sparsi in tutto il modulo."""
    axes: set[str] = set()
    for tr in fig.data:
        if tr.type == "bar":
            axes.add("yaxis" if getattr(tr, "orientation", None) == "h" else "xaxis")
        elif tr.type == "scatter":
            axes.add("xaxis")
    return sorted(axes)


def _ease_for(style: str, t: float) -> float:
    """Curva di easing sul progresso grezzo t in [0,1] — 'rimbalzo' supera
    temporaneamente 1.0 (il dato cresce oltre il valore finale) e torna
    indietro (easeOutBack); gli altri stili usano un ease-out puro (parte a
    velocità piena, rallenta solo alla fine). NON uno smoothstep: uno
    smoothstep ha velocità ZERO sia alla fine di un passo sia all'inizio del
    successivo — incatenando i passi, quei due "rallentamenti a velocità
    zero" si sommano proprio al cambio dato, e si percepiscono come una
    lunga pausa prima che il prossimo dato inizi a comparire."""
    t = min(max(t, 0.0), 1.0)
    if style == "rimbalzo":
        c1, c3 = 1.70158, 2.70158
        return 1 + c3 * (t - 1) ** 3 + c1 * (t - 1) ** 2
    if style == "fiorisce":
        # Come 'rimbalzo' ma con un superamento molto più piccolo — un
        # "respiro" organico invece di uno scatto elastico, più adatto a un
        # petalo/fetta che si apre che a un rimbalzo netto.
        c1 = 0.55
        c3 = c1 + 1
        return 1 + c3 * (t - 1) ** 3 + c1 * (t - 1) ** 2
    if style in ("stampa", "particelle"):
        # Non è la curva del VALORE (che scatta pieno come 'istantanea',
        # vedi _styled_value) ma quella del LAMPO: parte alta (impatto) e
        # decade rapidamente a 1.0 — 'stampa' decade in fretta (colpo secco),
        # 'particelle' decade più lentamente (scintille che si spengono).
        decay = 0.12 if style == "stampa" else 0.35
        peak = 3.2 if style == "stampa" else 2.4
        if t >= decay:
            return 1.0
        return 1.0 + (peak - 1.0) * (1 - t / decay)
    if style == "cubetti":
        # Non una curva continua: N scalini discreti — il valore salta di
        # colpo a ogni 1/N invece di crescere con continuità, come tanti
        # cubetti impilati uno sopra l'altro (vedi _styled_value, che usa
        # questo stesso p come frazione del valore finale, e il blocco
        # 'cubetti' in build_frames, che disegna un vero quadratino ad ogni
        # scatto sulla stessa quantizzazione).
        return math.ceil(t * _CUBETTI_N_BLOCKS) / _CUBETTI_N_BLOCKS
    return 1 - (1 - t) ** 2  # ease-out quadratico


def _styled_value(final: float, p: float, style: str) -> float:
    """Come il VALORE del dato risponde al progresso p (già passato per
    `_ease_for`), secondo lo stile scelto: 'crescita'/'rimbalzo'/'fiorisce'
    crescono con continuità (barre che crescono, fette che si aprono —
    'rimbalzo'/'fiorisce' superano temporaneamente il valore finale);
    tutti gli altri (compresi 'stampa'/'particelle', dove p può superare 1
    per il lampo — vedi _flash_amount) mostrano subito il valore pieno
    appena il dato inizia a comparire — la differenza sta nell'OPACITÀ
    (vedi _styled_opacity)."""
    if style in ("crescita", "rimbalzo", "fiorisce", "cubetti"):
        return max(0.0, final * p) if final >= 0 else min(0.0, final * p)
    return final if p > 0.001 else 0.0


def _styled_opacity(p: float, style: str) -> float:
    if style in ("dissolvenza", "verniciata"):
        return min(1.0, p)
    return 1.0 if p > 0.001 else 0.0


def _flash_amount(p: float, style: str) -> float:
    """Quanto di 'lampo'/schiarimento extra aggiungere al colore in questo
    istante. 'stampa'/'particelle' hanno p che supera 1.0 (vedi _ease_for),
    il surplus (p-1) è l'intensità del lampo, già decadente da sola.
    'verniciata' invece resta in [0,1] come 'dissolvenza' (vedi
    _styled_opacity): qui la 'lucentezza' è massima appena il dato compare
    (vernice fresca) e scompare mentre p sale verso 1 (vernice che
    'asciuga' e torna al colore normale)."""
    if style in ("stampa", "particelle"):
        return max(0.0, p - 1.0)
    if style == "verniciata":
        return max(0.0, 1.0 - min(p, 1.0)) if p > 0.001 else 0.0
    return 0.0


_SPOTLIGHT_BLINKS = 2.5  # cicli di lampeggio completi nella finestra attiva di un dato
_CUBETTI_N_BLOCKS = 6  # numero di "cubetti" (scatti) in cui si scompone la crescita di un dato


def _hl_amount(p: float, local_t: float, active: bool, spotlight_mode: str,
               style: str, glow_amount: float) -> float:
    """Quanto schiarire il colore in questo frame — somma di due effetti
    indipendenti: l'evidenziazione 'spotlight' (dato attualmente attivo, se
    richiesta — 'bagliore'/'doppio' danno uno schiarimento fisso, 'lampeggio'/
    'doppio' lo fanno invece ALTERNARE acceso/spento in base a local_t, la
    posizione lineare 0..1 nella finestra attiva del dato — 'impulso' da solo
    non tocca il colore, vedi _pulse_mult per quello) e il lampo/lucentezza
    di 'stampa'/'particelle'/'verniciata' (che si applica da solo,
    indipendentemente dallo spotlight, così l'effetto si vede anche a
    spotlight disattivato)."""
    base_amt = 0.0
    if active and spotlight_mode in ("bagliore", "lampeggio", "doppio"):
        if spotlight_mode == "bagliore":
            base_amt = glow_amount
        else:
            base_amt = glow_amount if math.sin(local_t * 2 * math.pi * _SPOTLIGHT_BLINKS) > 0 else 0.0
    if style == "particelle":
        flash_mult = 0.85
    elif style == "verniciata":
        flash_mult = 0.6
    else:
        flash_mult = 0.7
    flash_amt = _flash_amount(p, style) * flash_mult
    return min(0.97, base_amt + flash_amt)


def _pulse_mult(local_t: float, active: bool, spotlight_mode: str) -> float:
    """Moltiplicatore di dimensione/spessore bordo per 'impulso'/'doppio'
    (1.0 = nessun impulso) — un respiro ritmico sincronizzato allo stesso
    local_t/ciclo del lampeggio, indipendente dal colore."""
    if active and spotlight_mode in ("impulso", "doppio"):
        return 1.0 + 0.32 * (0.5 + 0.5 * math.sin(local_t * 2 * math.pi * _SPOTLIGHT_BLINKS))
    return 1.0


def _trace_frame_update(tr, trace_cells: list[dict], progress: dict[str, float],
                         local_progress: dict[str, float], active_ids: set[str], *,
                         reveal_value: bool, spotlight_mode: str, style: str,
                         glow_amount: float) -> dict:
    by_idx = {c["elem_idx"]: c for c in trace_cells}
    show_text = reveal_value

    if tr.type == "box":
        c = trace_cells[0]
        return {"type": "box", "visible": bool(progress.get(c["id"], 1.0) > 0.001)}

    if tr.type == "violin":
        c = trace_cells[0]
        return {"type": "violin", "visible": bool(progress.get(c["id"], 1.0) > 0.001)}

    if tr.type == "pie":
        n = len(_arr(tr.labels))
        orig_vals = _arr(tr.values)
        base_colors = _pie_base_colors(tr, n)
        vals, colors = [], []
        for j in range(n):
            c = by_idx.get(j)
            p = progress.get(c["id"], 1.0) if c else 1.0
            lt = local_progress.get(c["id"], 1.0) if c else 1.0
            ov = orig_vals[j] if j < len(orig_vals) else 0
            vals.append(_styled_value(float(ov), p, style) if ov is not None else 0.0)
            active = bool(c and c["id"] in active_ids)
            amt = _hl_amount(p, lt, active, spotlight_mode, style, glow_amount)
            colors.append(_lighten(base_colors[j], amt) if amt > 0.001 else base_colors[j])
        return {"type": "pie", "values": vals, "marker": {"colors": colors}}

    if tr.type == "scatterpolar":
        # Radar: il poligono "si riempie dal centro" — ogni vertice cresce
        # da 0 al suo valore (stessa logica di crescita di una barra), il
        # punto di chiusura (ultimo elemento, duplicato del primo) segue
        # sempre il valore CORRENTE del vertice 0 così il poligono resta
        # sempre chiuso anche a metà animazione.
        base_color = _scatter_base_color(tr)
        orig_r = _arr(tr.r)
        n_real = max(0, len(orig_r) - 1)
        vals = list(orig_r)
        sizes, texts, marker_colors = [], [], []
        for j in range(n_real):
            c = by_idx.get(j)
            p = progress.get(c["id"], 1.0) if c else 1.0
            lt = local_progress.get(c["id"], 1.0) if c else 1.0
            ov = orig_r[j]
            if style in ("cubetti", "acqua") and c and 0.0 < lt < 1.0:
                # Il vertice VERO resta a 0 (un 'incavo' verso il centro)
                # per tutta la sua finestra di crescita — la sua forma in
                # quella finestra è interamente lo SPICCHIO ausiliario (vedi
                # build_frames), un'AREA che si riempie, non un marker o una
                # linea che viaggia lungo il raggio. Stesso principio già
                # applicato alle barre: mai due rappresentazioni diverse
                # (poligono che cresce a scatti PIÙ uno spicchio) attive sullo
                # stesso pezzo contemporaneamente.
                v = 0.0
            else:
                v = _styled_value(float(ov), p, style) if ov is not None else 0.0
            vals[j] = v
            active = bool(c and c["id"] in active_ids)
            amt = _hl_amount(p, lt, active, spotlight_mode, style, glow_amount)
            marker_colors.append(_lighten(base_color, amt) if amt > 0.001 else base_color)
            size_boost = (1 + _flash_amount(p, style) * (1.2 if style == "particelle" else 0.5)) \
                * _pulse_mult(lt, active, spotlight_mode)
            sizes.append((9 if active else 5) * size_boost)
            texts.append(_value_text(v, reveal_value) if ov is not None else "")
        if n_real:
            vals[n_real] = vals[0]
            sizes.append(sizes[0])
            marker_colors.append(marker_colors[0])
            texts.append("")
        upd: dict[str, Any] = {"type": "scatterpolar", "r": vals,
                                "marker": {"size": sizes, "color": marker_colors}}
        if show_text:
            upd.update(text=texts, texttemplate="%{text}", textposition="top center",
                       mode="lines+markers+text")
        return upd

    if tr.type == "funnel":
        # Un solo valore per livello, sempre lungo x (vedi list_cells) — la
        # crescita/dissolvenza/lampo ecc. sono le stesse curve generiche
        # già usate per le barre, 'textinfo' del trace base ("value+percent
        # initial") ricalcola da sé l'etichetta a ogni frame dal valore
        # corrente, non serve gestire il testo qui.
        base_colors = _bar_base_colors(tr, n_cells := len(trace_cells))
        vals, colors, opacities = [], [], []
        for j in range(n_cells):
            c = by_idx.get(j)
            p = progress.get(c["id"], 1.0) if c else 1.0
            lt = local_progress.get(c["id"], 1.0) if c else 1.0
            ov = c["value"] if c else 0.0
            v = _styled_value(float(ov), p, style)
            vals.append(v)
            opacities.append(_styled_opacity(p, style))
            active = bool(c and c["id"] in active_ids)
            amt = _hl_amount(p, lt, active, spotlight_mode, style, glow_amount)
            base_col = base_colors[j] if j < len(base_colors) else "#4C72B0"
            colors.append(_lighten(base_col, amt) if amt > 0.001 else base_col)
        return {"type": "funnel", "x": vals, "marker": {"color": colors, "opacity": opacities}}

    horiz = tr.type == "bar" and getattr(tr, "orientation", None) == "h"
    val_attr = "x" if horiz else "y"
    xs = _arr(tr.x)
    ys = _arr(tr.y)
    orig_vals = xs if horiz else ys
    n = len(orig_vals)

    if tr.type == "scatter":
        base_color = _scatter_base_color(tr)
        active_x, active_y, sizes, opacities, colors, texts = [], [], [], [], [], []
        for j in range(len(xs)):
            c = by_idx.get(j)
            p = progress.get(c["id"], 1.0) if c else 1.0
            lt = local_progress.get(c["id"], 1.0) if c else 1.0
            if p > 0.001:
                px, py = xs[j], ys[j]
                if style == "linea" and c and 0.0 < lt < 1.0 and j > 0:
                    # 'Cammina': il punto ATTIVO non salta di colpo alla sua
                    # posizione finale — scivola con continuità da quella
                    # del punto precedente (interpolazione lineare su lt),
                    # così il segmento si vede allungare invece di comparire
                    # di colpo appena entrambi gli estremi esistono. Solo se
                    # entrambe le coordinate sono numeriche (un asse X a
                    # categorie non è interpolabile) — altrimenti si
                    # comporta come lo stile di default.
                    try:
                        x0v, x1v = float(xs[j - 1]), float(xs[j])
                        y0v, y1v = float(ys[j - 1]), float(ys[j])
                        px = x0v + (x1v - x0v) * lt
                        py = y0v + (y1v - y0v) * lt
                    except (TypeError, ValueError):
                        pass
                active_x.append(px)
                active_y.append(py)
                active = bool(c and c["id"] in active_ids)
                amt = _hl_amount(p, lt, active, spotlight_mode, style, glow_amount)
                base = (8 * min(p, 1.0)) if style == "crescita" else 8.0
                size_boost = (1 + _flash_amount(p, style) * (1.3 if style == "particelle" else 0.4)) \
                    * _pulse_mult(lt, active, spotlight_mode)
                sizes.append(max(3, (base * 1.6 if active else base)) * size_boost)
                opacities.append(_styled_opacity(p, style))
                colors.append(_lighten(base_color, amt) if amt > 0.001 else base_color)
                texts.append(_value_text(ys[j], reveal_value) if ys[j] is not None else "")
        upd: dict[str, Any] = {"type": "scatter", "x": active_x, "y": active_y,
                                "marker": {"size": sizes, "opacity": opacities, "color": colors}}
        if show_text:
            upd.update(text=texts, texttemplate="%{text}", textposition="top center",
                       mode="lines+markers+text")
        return upd

    # bar
    base_colors = _bar_base_colors(tr, n)
    vals, colors, line_colors, opacities, texts, line_w = [], [], [], [], [], []
    for j in range(n):
        c = by_idx.get(j)
        p = progress.get(c["id"], 1.0) if c else 1.0
        lt = local_progress.get(c["id"], 1.0) if c else 1.0
        ov = orig_vals[j]
        v = _styled_value(float(ov), p, style) if ov is not None else 0.0
        if style == "acqua" and ov is not None:
            # Il 'contenitore' (questa stessa barra) è già alla sua
            # dimensione finale non appena inizia il suo turno — è un
            # vetro/bicchiere di dimensione FISSA, non qualcosa che cresce;
            # a salire con continuità è solo il LIVELLO DELL'ACQUA dentro
            # (traccia separata, vedi build_frames). Il contenitore diventa
            # opaco/pieno-colore solo a riempimento completato.
            v = 0.0 if lt <= 0.0 else float(ov)
        vals.append(v)
        active = bool(c and c["id"] in active_ids)
        amt = _hl_amount(p, lt, active, spotlight_mode, style, glow_amount)
        base_col = _lighten(base_colors[j], amt) if amt > 0.001 else base_colors[j]
        # per 'particelle'/'stampa': un bordo che si accende col lampo (una
        # scintilla sul bordo che cresce) e si spegne insieme al colore;
        # 'impulso'/'doppio' aggiungono un respiro ritmico allo stesso bordo.
        pulse = _pulse_mult(lt, active, spotlight_mode)
        base_line_w = _flash_amount(p, style) * (5 if style == "particelle" else 3)
        extra_pulse_w = (pulse - 1.0) * 6.0
        lw = min(7, base_line_w + extra_pulse_w)

        if style == "cubetti" and 0.0 < lt < 1.0:
            # la barra VERA resta invisibile finché è in crescita: la sua
            # forma durante questa finestra è interamente quella della pila
            # di cubetti (vedi build_frames) — mostrarla ANCHE lei
            # (crescita normale "sotto" ai cubetti) è quello che rendeva
            # l'effetto "bislacco" segnalato dall'utente (un cubetto che
            # compare mentre la barra intera cresce comunque come al
            # solito). Appena il dato è completo la barra torna piena e
            # opaca nello stesso identico frame in cui la pila sparisce —
            # serve confrontare lt (posizione LINEARE nella finestra), non p
            # (che per 'cubetti' è una scalinata: arriva già a 1.0 PRIMA che
            # lt tocchi 1.0, perché l'ultimo scatto satura subito col
            # ceiling — usare p qui riaccendeva la barra un frame troppo
            # presto, sovrapposta all'ultimo cubetto ancora in scena.
            opacities.append(0.0)
            colors.append(base_col)
            line_colors.append(base_col)
            line_w.append(lw)
        elif style == "acqua" and 0.0 < lt < 1.0:
            # 'Vetro' vero: riempimento TRASPARENTE (rgba alpha 0, non la
            # proprietà opacity — quella scurirebbe anche il contorno),
            # solo il bordo laterale resta visibile nel colore del dato.
            # L'interno si riempie con l'acqua, la traccia separata sotto
            # (vedi build_frames) — mai i due sovrapposti sullo stesso
            # riquadro pieno, altrimenti si ricade nel bug già corretto per
            # 'cubetti' (due rappresentazioni in conflitto nello stesso
            # istante).
            opacities.append(1.0)
            colors.append("rgba(0,0,0,0)")
            line_colors.append(base_col)
            line_w.append(max(2.5, lw))
        elif style == "acqua" and lt <= 0.0:
            opacities.append(0.0)
            colors.append(base_col)
            line_colors.append(base_col)
            line_w.append(lw)
        else:
            opacities.append(_styled_opacity(p, style))
            colors.append(base_col)
            line_colors.append(base_col)
            line_w.append(lw)
        texts.append(_value_text(v, reveal_value) if (p > 0.001 and ov is not None) else "")
    upd = {"type": "bar", val_attr: vals,
           "marker": {"color": colors, "opacity": opacities,
                      "line": {"color": line_colors, "width": line_w}}}
    if show_text:
        upd.update(text=texts, texttemplate="%{text}", textposition="outside")
    return upd


def build_frames(fig: go.Figure, steps: list[list[dict]], *,
                  style: str = "crescita",
                  reveal_x_label: bool = False,
                  reveal_value: bool = False,
                  reveal_legend: bool = False,
                  spotlight_mode: str = "bagliore",
                  subframes_per_step: int = 10,
                  glow_amount: float = 0.65) -> tuple[list[go.Frame], list[go.Scatter]]:
    """`steps`: risultato di `build_steps` (celle incluse — quelle escluse
    dall'utente vanno già filtrate PRIMA di chiamare questa funzione, e
    restano semplicemente sempre piene, senza un loro momento dedicato).
    Ritorna (frames, extra_traces): frames avanzano a stati — l'ULTIMO
    coincide ESATTAMENTE col grafico originale; extra_traces sono tracce
    AUSILIARIE (es. i quadratini di 'cubetti', vedi sotto) che il chiamante
    DEVE aggiungere alla figura PRIMA di assegnarle questi frames (l'indice
    di ogni traccia in fig.data deve combaciare con la posizione dei suoi
    aggiornamenti in ogni frame — vedi prepare_animated_figure/
    build_autoplay_html, che lo fanno già).

    'reveal_x_label' tocca il vero asse categorico (xaxis, o yaxis per le
    barre orizzontali — vedi _reveal_axes): parte VUOTO e rivela ogni
    etichetta quando il suo dato compare, nell'ordine naturale dell'asse
    (non nell'ordine di rivelazione, che può essere stato riordinato) così i
    tick restano nella loro posizione corretta invece di spostarsi."""
    n = len(steps)
    if n == 0:
        return [go.Frame(data=[{"type": tr.type} for tr in fig.data], layout={}, name="0")], []

    cells_by_trace: dict[int, list[dict]] = {}
    for step in steps:
        for c in step:
            cells_by_trace.setdefault(c["trace_idx"], []).append(c)

    reveal_axes = _reveal_axes(fig) if reveal_x_label else []
    all_cells = [c for step in steps for c in step]
    x_order: list[str] = []
    for c in all_cells:
        if c["x_key"] not in x_order:
            x_order.append(c["x_key"])

    # Barre orizzontali: la categoria vive sull'asse Y (a sinistra) — nascondere
    # e poi rivelare il TESTO del tick per farlo "comparire" cambia la sua
    # LARGHEZZA (una stringa vuota è molto più stretta del nome vero), e con
    # automargin attivo (impostazione di default dell'app, vedi
    # fig.update_yaxes(automargin=True) in streamlit_app.py) questo fa
    # espandere/contrarre il margine sinistro a ogni fotogramma: l'intera area
    # del grafico "salta" a destra e sinistra mentre le barre compaiono
    # (bug segnalato dall'utente). Le barre verticali non hanno lo stesso
    # problema: l'etichetta sotto occupa un'ALTEZZA quasi costante
    # indipendentemente dalla lunghezza del testo (una riga sola, non
    # ruotata), quindi lì automargin resta invariato e attivo. Fix: un
    # margine sinistro FISSO per tutta l'animazione (automargin spento SOLO
    # su questo asse, solo qui) — stima approssimativa via conteggio
    # caratteri (nessun renderer disponibile in puro Python per misurare il
    # testo esatto), ma costante frame per frame è l'unica cosa che conta
    # per eliminare il salto.
    _margin_l_lock = None
    if "yaxis" in reveal_axes and x_order:
        _max_len = max(len(xk) for xk in x_order)
        _margin_l_lock = max(90, int(28 + _max_len * 6.3))

    # Legenda "inchiodata" alla posizione impostata dall'utente (statica sul
    # grafico finale): senza questo, quando reveal_legend attiva/disattiva
    # showlegend per traccia frame per frame, Plotly.js a volte ricalcola da
    # solo la posizione automatica della legenda (spazio disponibile diverso
    # a ogni fotogramma) e la fa "saltare" durante la riproduzione invece di
    # restare ferma dove l'utente l'ha messa dalle impostazioni.
    _legend_pin = fig.layout.legend.to_plotly_json() if fig.layout.legend else None

    # 'cubetti': un vero quadratino "atterra" in cima alla barra a ogni
    # scatto (vedi _CUBETTI_N_BLOCKS in _ease_for) — go.Bar non può disegnare
    # forme proprie dentro/sopra di sé, serve una traccia SEPARATA (marker a
    # quadrato) sovrapposta, che appare solo per l'istante dello scatto e
    # sfuma prima del successivo. Il chiamante deve aggiungerla alla figura
    # (vedi extra_traces nel valore di ritorno) — qui prepariamo solo colori
    # e orientamento una volta sola, fuori dal loop dei frame.
    extra_traces: list[go.Scatter] = []
    cubetti_cells: list[dict] = []
    cubetti_horiz: dict[int, bool] = {}
    cubetti_colors: dict[str, str] = {}
    # Radar: qui i 'cubetti' sono un vero SPICCHIO (area, fill='toself'),
    # centrato sul vertice, che cresce a SCATTI in raggio — non triangolini
    # o marker che viaggiano lungo il raggio (bocciati: sembravano ancora
    # "solo crescita a scatti" perché il poligono sotto continuava a
    # crescere anche lui). Il vertice VERO ora resta a 0 mentre lo spicchio
    # è nella sua finestra (vedi branch scatterpolar sopra in
    # _trace_frame_update) — stesso principio di "una sola rappresentazione
    # alla volta" già applicato alle barre.
    cubetti_radar_cells: list[dict] = []
    cubetti_radar_colors: dict[str, str] = {}
    # L'angularaxis del radar è ora NUMERICA in gradi (0..360, con
    # tickvals/ticktext per mostrare comunque i nomi delle categorie — vedi
    # streamlit_app.py, chart_key=='radar'): prima era a CATEGORIE (theta
    # del poligono = stringhe come "Nord"), e su un asse così un valore
    # theta numerico passato a una traccia ausiliaria NON veniva letto come
    # grado (solo marker.angle, una proprietà diversa, lo fa) — veniva
    # registrato come categoria IN PIÙ, deformando l'asse (verificato con
    # uno screenshot: comparivano tick spuri "0.47"/"-0.47"). Con l'asse
    # numerico lo spicchio può tornare a usare gradi veri: centro sul theta
    # del vertice ± questa frazione di mezzo passo tra vertici (con un
    # margine perché spicchi adiacenti non si tocchino).
    _RADAR_WEDGE_HALF_FRAC = 0.94
    # Dispersione/Linee: un punto non ha una 'quantità che si riempie' di
    # suo (è solo una posizione x,y) — trattato come una BARRA FANTASMA
    # invisibile dalla baseline 0 fino al suo valore y, stessa identica
    # logica di 'cubetti' su barra (quadratini impilati lungo quella
    # colonna), solo che qui non c'è una vera barra sotto da nascondere
    # (il punto/la linea restano quello che sono, il valore y non è mai
    # stato una barra riempita — vedi anche 'acqua'/'sipario' più sotto,
    # stessa idea di 'colonna fantasma').
    cubetti_scatter_cells: list[dict] = []
    cubetti_scatter_colors: dict[str, str] = {}
    if style == "cubetti":
        base_colors_cache: dict[int, list[str]] = {}
        for c in all_cells:
            ti = c["trace_idx"]
            tr = fig.data[ti]
            if tr.type == "bar":
                cubetti_horiz.setdefault(ti, getattr(tr, "orientation", None) == "h")
                if ti not in base_colors_cache:
                    n_cat = len(_arr(tr.y if cubetti_horiz[ti] else tr.x))
                    base_colors_cache[ti] = _bar_base_colors(tr, n_cat)
                colors_for_ti = base_colors_cache[ti]
                base_color = colors_for_ti[c["elem_idx"]] if c["elem_idx"] < len(colors_for_ti) else "#4C72B0"
                cubetti_colors[c["id"]] = _lighten(base_color, 0.5)
                cubetti_cells.append(c)
            elif tr.type == "scatterpolar":
                cubetti_radar_colors[c["id"]] = _scatter_base_color(tr)
                cubetti_radar_cells.append(c)
            elif tr.type == "scatter" and _is_pure_scatter(tr):
                cubetti_scatter_colors[c["id"]] = _lighten(_scatter_base_color(tr), 0.5)
                cubetti_scatter_cells.append(c)
        if cubetti_cells or cubetti_scatter_cells:
            extra_traces.append(go.Scatter(
                x=[], y=[], mode="markers",
                marker=dict(symbol="square", size=14, line=dict(width=0)),
                showlegend=False, hoverinfo="skip", name=""))
        if cubetti_radar_cells:
            extra_traces.append(go.Scatterpolar(
                r=[], theta=[], mode="lines", fill="toself", connectgaps=False,
                line=dict(width=1),
                showlegend=False, hoverinfo="skip", name=""))

    # 'acqua': un vero LIVELLO DI LIQUIDO che sale dentro al 'contenitore'
    # (la barra stessa, resa pallida/trasparente — vedi il branch 'acqua' più
    # sopra in _trace_frame_update). Anche qui serve una traccia SEPARATA
    # (un'altra go.Bar, non uno scatter: solo go.Bar calcola da sé la
    # larghezza giusta per stare esattamente sopra la barra/contenitore,
    # senza dover indovinare offset numerici su un asse a categorie — lo
    # stesso problema geometrico incontrato con 'cubetti'). Le due barre
    # (contenitore + acqua) devono sovrapporsi invece di affiancarsi: serve
    # barmode='overlay', impostato qui sotto in ogni layout_upd SOLO quando
    # 'acqua' è davvero in uso.
    acqua_cells: list[dict] = []
    acqua_horiz: dict[int, bool] = {}
    acqua_colors: dict[str, str] = {}
    # Radar: stessa geometria a SPICCHIO di 'cubetti' sopra (area,
    # fill='toself', vertice vero a 0 mentre lo spicchio è nella sua
    # finestra, indice del vertice ± _RADAR_WEDGE_HALF come coordinata theta
    # — vedi nota lì sul perché non sono gradi) — cambia solo l'easing del
    # raggio: qui cresce con continuità (curva ease-out di default, non a
    # scatti) invece che a blocchi, coerente con "acqua che sale" invece di
    # "cubetti che si impilano".
    acqua_radar_cells: list[dict] = []
    acqua_radar_colors: dict[str, str] = {}
    acqua_scatter_cells: list[dict] = []
    acqua_scatter_colors: dict[str, str] = {}
    if style == "acqua":
        base_colors_cache2: dict[int, list[str]] = {}
        for c in all_cells:
            ti = c["trace_idx"]
            tr = fig.data[ti]
            if tr.type == "bar":
                acqua_horiz.setdefault(ti, getattr(tr, "orientation", None) == "h")
                if ti not in base_colors_cache2:
                    n_cat = len(_arr(tr.y if acqua_horiz[ti] else tr.x))
                    base_colors_cache2[ti] = _bar_base_colors(tr, n_cat)
                colors_for_ti = base_colors_cache2[ti]
                base_color = colors_for_ti[c["elem_idx"]] if c["elem_idx"] < len(colors_for_ti) else "#4C72B0"
                acqua_colors[c["id"]] = base_color
                acqua_cells.append(c)
            elif tr.type == "scatterpolar":
                acqua_radar_colors[c["id"]] = _scatter_base_color(tr)
                acqua_radar_cells.append(c)
            elif tr.type == "scatter" and _is_pure_scatter(tr):
                # Dispersione (non Linee, vedi _is_pure_scatter): nessun
                # 'vetro' da rendere trasparente (un punto non è un
                # contenitore) — il punto stesso compare normale (già
                # gestito dal branch scatter generico), l'unica cosa in più
                # è lo STELO che sale verso di lui dalla baseline 0, come un
                # fiore/lecca-lecca che cresce — stessa traccia ausiliaria
                # go.Bar dell'acqua su barra.
                acqua_scatter_colors[c["id"]] = _scatter_base_color(tr)
                acqua_scatter_cells.append(c)
        if acqua_cells or acqua_scatter_cells:
            extra_traces.append(go.Bar(
                x=[], y=[], marker=dict(color=[], line=dict(width=0)),
                showlegend=False, hoverinfo="skip", name="", width=None))
        if acqua_radar_cells:
            extra_traces.append(go.Scatterpolar(
                r=[], theta=[], mode="lines", fill="toself", connectgaps=False,
                line=dict(width=1),
                showlegend=False, hoverinfo="skip", name=""))

    # 'linea': NON è un overlay — cambia direttamente come il punto attivo
    # si muove nel branch 'scatter' di _trace_frame_update (interpolazione
    # dalla posizione del punto precedente), così il segmento si vede
    # allungare con continuità invece di comparire di colpo. Solo per
    # 'scatter' (grafici a linee/dispersione): sugli altri tipi si comporta
    # come lo stile di default, nessun effetto — vedi _styled_value.

    # 'sipario': un sipario sul GRAFICO INTERO, non sui singoli dati
    # (correzione esplicita dell'utente rispetto alla prima versione, che
    # apriva un'antina per ogni barra) — il grafico è già completo, fermo,
    # fin dal primo frame; a "rivelarlo" sono due ante rettangolari OPACHE
    # a coordinate di PAGINA (xref/yref='paper', non 'x'/'y': coprono
    # l'intera area del grafico indipendentemente da assi/tipo di dato, e
    # per questo funzionano identiche su QUALSIASI tipo di grafico — barre,
    # linee, dispersione, torta, radar, box, violino — senza bisogno di
    # nessuna geometria per-elemento) che si aprono verso i due lati in
    # un'unica dissolvenza globale, sincronizzata sulla durata TOTALE
    # dell'animazione (non sull'ordine/i passi dei singoli dati, che per
    # questo stile non contano: vedi is_sipario più sotto, che forza subito
    # tutti i dati al loro stato finale).
    _SIPARIO_COLOR = "#7A1F3D"
    is_sipario = style == "sipario"

    total_steps = n * subframes_per_step
    frames: list[go.Frame] = []
    for step_i in range(total_steps + 1):
        is_last = step_i >= total_steps
        cur = n - 1 if is_last else min(step_i // subframes_per_step, n - 1)
        progress: dict[str, float] = {}
        local_progress: dict[str, float] = {}
        active_ids: set[str] = set()
        for i, step in enumerate(steps):
            if is_last or i < cur:
                p, lt = 1.0, 1.0
            elif i > cur:
                p, lt = 0.0, 0.0
            else:
                local = (step_i - i * subframes_per_step + 1) / subframes_per_step
                p = _ease_for(style, local)
                lt = min(max(local, 0.0), 1.0)
            for c in step:
                progress[c["id"]] = p
                local_progress[c["id"]] = lt
                if i == cur and not is_last:
                    active_ids.add(c["id"])

        if is_sipario:
            # Il grafico sotto al sipario è già completo fin dal primo
            # frame — nessun dato cresce/appare in sequenza, l'unica cosa
            # che anima è il sipario stesso (vedi sotto, su t_global).
            for cid in progress:
                progress[cid] = 1.0
                local_progress[cid] = 1.0
            active_ids = set()

        frame_data = []
        for ti, tr in enumerate(fig.data):
            tcells = cells_by_trace.get(ti)
            if not tcells:
                frame_data.append({"type": tr.type})
                continue
            upd = _trace_frame_update(tr, tcells, progress, local_progress, active_ids,
                                       reveal_value=reveal_value,
                                       spotlight_mode=spotlight_mode, style=style, glow_amount=glow_amount)
            if reveal_legend and tcells[0]["series"] is not None:
                shown = max((progress.get(c["id"], 1.0) for c in tcells), default=1.0)
                upd["showlegend"] = bool(shown > 0.001)
            frame_data.append(upd)

        if cubetti_cells or cubetti_scatter_cells:
            # La pila INTERA di cubetti già atterrati resta visibile (non
            # sparisce dopo il primo) — è lei stessa, per tutta la finestra
            # di crescita, l'unica cosa che rappresenta la barra (che nel
            # frattempo è invisibile, vedi sopra): un cubetto alla volta si
            # aggiunge in cima, quello appena atterrato ha un piccolo
            # "rimbalzo" di dimensione che si assesta prima del prossimo.
            ox, oy, osize, oop, ocol = [], [], [], [], []
            for c in cubetti_cells:
                lt = local_progress.get(c["id"], 1.0)
                if 0.0 < lt < 1.0:
                    frac = (lt * _CUBETTI_N_BLOCKS) % 1.0
                    k = math.ceil(lt * _CUBETTI_N_BLOCKS)
                    row_h = c["value"] / _CUBETTI_N_BLOCKS
                    horiz = cubetti_horiz.get(c["trace_idx"], False)
                    color = cubetti_colors.get(c["id"], "#FFFFFF")
                    for row in range(k):
                        is_newest = row == k - 1
                        center = row_h * (row + 0.5)
                        pop = (1.5 - 0.5 * frac) if is_newest else 1.0
                        op = max(0.55, 1.0 - frac * 0.3) if is_newest else 1.0
                        if horiz:
                            ox.append(center); oy.append(c["x_key"])
                        else:
                            ox.append(c["x_key"]); oy.append(center)
                        osize.append(15 * pop)
                        oop.append(op)
                        ocol.append(color)
            for c in cubetti_scatter_cells:
                lt = local_progress.get(c["id"], 1.0)
                if 0.0 < lt < 1.0:
                    frac = (lt * _CUBETTI_N_BLOCKS) % 1.0
                    k = math.ceil(lt * _CUBETTI_N_BLOCKS)
                    row_h = c["value"] / _CUBETTI_N_BLOCKS
                    color = cubetti_scatter_colors.get(c["id"], "#FFFFFF")
                    for row in range(k):
                        is_newest = row == k - 1
                        center = row_h * (row + 0.5)
                        pop = (1.5 - 0.5 * frac) if is_newest else 1.0
                        op = max(0.55, 1.0 - frac * 0.3) if is_newest else 1.0
                        ox.append(c["pos_x"]); oy.append(center)
                        osize.append(15 * pop); oop.append(op); ocol.append(color)
            frame_data.append({"type": "scatter", "x": ox, "y": oy,
                                "marker": {"size": osize, "opacity": oop, "color": ocol}})

        if cubetti_radar_cells:
            # Un vero SPICCHIO (area) per ogni vertice attivo, centrato sul
            # suo theta, largo _radar_half_deg su entrambi i lati — il
            # raggio salta a scatti (stessa quantizzazione delle barre) tra
            # i 6 traguardi, con un breve schiarimento appena dopo ogni
            # scatto che si affievolisce prima del prossimo (l'equivalente
            # del "rimbalzo" di dimensione dei cubetti su barra, qui sul
            # colore visto che l'oggetto è un'area, non un marker da poter
            # ingrandire). Più spicchi attivi insieme: segmenti separati da
            # None nella stessa traccia (fillcolor è unico per frame — vedi
            # nota acqua sotto).
            wr, wth, wcol = [], [], None
            for c in cubetti_radar_cells:
                lt = local_progress.get(c["id"], 1.0)
                if 0.0 < lt < 1.0:
                    frac = (lt * _CUBETTI_N_BLOCKS) % 1.0
                    k = math.ceil(lt * _CUBETTI_N_BLOCKS)
                    r_target = c["value"] * (k / _CUBETTI_N_BLOCKS)
                    n_real_ = max(0, len(_arr(fig.data[c["trace_idx"]].theta)) - 1)
                    ang = c["elem_idx"] * 360.0 / max(n_real_, 1)
                    half = (360.0 / max(n_real_, 1)) / 2 * _RADAR_WEDGE_HALF_FRAC
                    if wr:
                        wr.append(None); wth.append(None)
                    wr += [0.0, r_target, r_target]
                    wth += [ang, ang - half, ang + half]
                    if wcol is None:
                        base = cubetti_radar_colors.get(c["id"], "#4C72B0")
                        pop_amt = max(0.0, 1.0 - frac * 2.5) * 0.4
                        wcol = _lighten(base, pop_amt) if pop_amt > 0.001 else base
            frame_data.append({"type": "scatterpolar", "r": wr, "theta": wth,
                                "fillcolor": wcol or "#4C72B0", "line": {"color": wcol or "#4C72B0"}})

        if acqua_cells or acqua_scatter_cells:
            # Il livello dell'acqua sale con continuità (stessa curva
            # ease-out di 'crescita', qui via il fallback generico di
            # _ease_for) SOLO per i contenitori attualmente nel loro turno
            # (0<lt<1) — prima non è ancora il suo turno (contenitore
            # invisibile, niente acqua da mostrare), dopo il contenitore
            # stesso torna colore pieno (l'acqua "diventa" la barra finale,
            # la traccia ausiliaria si svuota nello stesso frame).
            wx, wy, wcol, woff, ww = [], [], [], [], []
            for c in acqua_cells:
                lt = local_progress.get(c["id"], 1.0)
                if 0.0 < lt < 1.0:
                    p_water = progress.get(c["id"], 0.0)
                    level = c["value"] * p_water
                    if acqua_horiz.get(c["trace_idx"], False):
                        wx.append(level); wy.append(c["x_key"])
                    else:
                        wx.append(c["x_key"]); wy.append(level)
                    wcol.append(acqua_colors.get(c["id"], "#4C72B0"))
                    # Barre raggruppate con posizione/larghezza esplicite (vedi
                    # _add_grouped_bars_plotly): l'acqua segue la SUA barra,
                    # non il centro della categoria.
                    src = fig.data[c["trace_idx"]]
                    offs, wids = _arr(src.offset) if isinstance(src.offset, (list, tuple)) else [], \
                                 _arr(src.width) if isinstance(src.width, (list, tuple)) else []
                    j = c["elem_idx"]
                    woff.append(offs[j] if j < len(offs) else None)
                    ww.append(wids[j] if j < len(wids) else None)
            for c in acqua_scatter_cells:
                lt = local_progress.get(c["id"], 1.0)
                if 0.0 < lt < 1.0:
                    p_water = progress.get(c["id"], 0.0)
                    level = c["value"] * p_water
                    wx.append(c["pos_x"]); wy.append(level)
                    wcol.append(acqua_scatter_colors.get(c["id"], "#4C72B0"))
            water = {"type": "bar", "x": wx, "y": wy, "marker": {"color": wcol}}
            if any(o is not None for o in woff):
                # una barra senza geometria esplicita resta centrata (offset 0 → -w/2)
                water["width"] = [w if w is not None else 0.8 for w in ww]
                water["offset"] = [o if o is not None else -0.4 for o in woff]
            frame_data.append(water)

        if acqua_radar_cells:
            # Stesso spicchio (area) di 'cubetti' sopra, ma il raggio cresce
            # con continuità (curva ease-out) invece che a scatti — vedi
            # nota lì per la semplificazione sul colore condiviso quando più
            # spicchi sono attivi insieme.
            rr, rth, rcol = [], [], None
            for c in acqua_radar_cells:
                lt = local_progress.get(c["id"], 1.0)
                if 0.0 < lt < 1.0:
                    p_water = progress.get(c["id"], 0.0)
                    r_target = c["value"] * p_water
                    n_real_ = max(0, len(_arr(fig.data[c["trace_idx"]].theta)) - 1)
                    ang = c["elem_idx"] * 360.0 / max(n_real_, 1)
                    half = (360.0 / max(n_real_, 1)) / 2 * _RADAR_WEDGE_HALF_FRAC
                    if rr:
                        rr.append(None); rth.append(None)
                    rr += [0.0, r_target, r_target]
                    rth += [ang, ang - half, ang + half]
                    if rcol is None:
                        rcol = acqua_radar_colors.get(c["id"], "#4C72B0")
            frame_data.append({"type": "scatterpolar", "r": rr, "theta": rth,
                                "fillcolor": rcol or "#4C72B0", "line": {"color": rcol or "#4C72B0"}})

        sipario_shapes: list[dict] = []
        if is_sipario:
            # Due ante opache a coordinate di PAGINA (0..1 su tutta la
            # figura, non sui dati) che coprono l'intero grafico all'inizio
            # e si aprono verso i due lati fino a sparire — t_global copre
            # l'intera durata dell'animazione, non i singoli passi/dati
            # (per questo stile non hanno alcun ruolo, vedi sopra).
            t_global = 1.0 if is_last else min(max(step_i / max(total_steps, 1), 0.0), 1.0)
            gap = 0.5 * t_global
            sipario_shapes.append(dict(type="rect", xref="paper", yref="paper",
                                        x0=0.0, x1=0.5 - gap, y0=0.0, y1=1.0,
                                        fillcolor=_SIPARIO_COLOR, line=dict(width=0), layer="above"))
            sipario_shapes.append(dict(type="rect", xref="paper", yref="paper",
                                        x0=0.5 + gap, x1=1.0, y0=0.0, y1=1.0,
                                        fillcolor=_SIPARIO_COLOR, line=dict(width=0), layer="above"))

        layout_upd: dict[str, Any] = {}
        if is_sipario:
            layout_upd["shapes"] = sipario_shapes
        if _legend_pin is not None:
            layout_upd["legend"] = _legend_pin
        if acqua_cells or acqua_scatter_cells:
            layout_upd["barmode"] = "overlay"
        if reveal_axes:
            # tickvals resta SEMPRE l'elenco completo (mai una lista che
            # cambia lunghezza tra un frame e l'altro): solo il TESTO delle
            # categorie non ancora raggiunte passa a "" e poi torna al nome
            # vero. Con un array che cambiava lunghezza ogni frame, Plotly
            # in alcuni casi disegnava etichette di categorie future prima
            # del loro turno (un redraw dell'asse con array di lunghezza
            # diversa è più soggetto a bug di questo tipo che una modifica
            # del solo contenuto testuale a lunghezza costante).
            ticktext = [xk if max((progress.get(c["id"], 1.0) for c in all_cells if c["x_key"] == xk),
                                   default=0.0) > 0.001 else ""
                        for xk in x_order]
            for ax_name in reveal_axes:
                layout_upd[ax_name] = {"tickmode": "array", "tickvals": x_order, "ticktext": ticktext}
                if ax_name == "yaxis" and _margin_l_lock is not None:
                    layout_upd[ax_name]["automargin"] = False
                    layout_upd.setdefault("margin", {})["l"] = _margin_l_lock

        frames.append(go.Frame(data=frame_data, layout=layout_upd, name=str(step_i)))
    return frames, extra_traces


# ── Export HTML autoplay ────────────────────────────────────────────────────

def suggested_duration_s(frames: list[go.Frame], frame_duration_ms: int) -> float:
    """Durata consigliata per la cattura in Pipeline Video: durata intrinseca
    dell'animazione + margine di sicurezza (avvio pagina + il piccolo ritardo
    prima che Plotly.animate parta) — verificato con una cattura Playwright
    reale che una durata 'esatta' arriva quasi sempre tagliata poco prima
    della fine."""
    return round(len(frames) * frame_duration_ms / 1000.0 + 1.0, 1)


def _nums(vals) -> list[float]:
    out = []
    for v in vals:
        try:
            out.append(float(v))
        except (TypeError, ValueError):
            pass  # non numerico (es. categoria testuale) — ignorato
    return out


def _axis_range(vals: list[float], log: bool) -> list[float] | None:
    """Range con margine per un asse numerico. Su asse log10, Plotly
    interpreta `range` come ESPONENTI in base 10 (non i valori grezzi) —
    calcolarlo in scala lineare romperebbe l'asse (range assurdo tipo
    10^-24..10^300). Richiede solo valori positivi in scala log (i non
    positivi vengono scartati, come fa Plotly stesso)."""
    if log:
        import math
        pos = [v for v in vals if v > 0]
        if not pos:
            return None
        lo, hi = math.log10(min(pos)), math.log10(max(pos))
        pad = (hi - lo) * 0.08 or 0.3
        return [lo - pad * 0.3, hi + pad]
    lo, hi = min(0, min(vals)), max(vals)
    pad = (hi - lo) * 0.08 or 1
    return [lo - pad * 0.2, hi + pad]


def _fix_axis_ranges(work: go.Figure) -> None:
    """Fissa il range degli assi numerici al valore FINALE (dati completi),
    invece di lasciarlo all'autorange di Plotly — che si blocca sul range del
    primo frame (quasi tutto a zero, a inizio animazione) e non lo
    ricalcola sui frame successivi: senza questo, tutte le barre/punti
    finiscono clippati alla stessa altezza invece di crescere ai loro valori
    reali."""
    xs, ys = [], []
    x_cat_vals, y_cat_vals = [], []
    for tr in work.data:
        if tr.type == "pie":
            continue
        xs += _nums(_arr(getattr(tr, "x", None)))
        ys += _nums(_arr(getattr(tr, "y", None)))
        x_cat_vals += [str(v) for v in _arr(getattr(tr, "x", None)) if v is not None]
        y_cat_vals += [str(v) for v in _arr(getattr(tr, "y", None)) if v is not None]

    x_log = bool(work.layout.xaxis and work.layout.xaxis.type == "log")
    y_log = bool(work.layout.yaxis and work.layout.yaxis.type == "log")
    x_cat = bool(work.layout.xaxis and work.layout.xaxis.type == "category")
    y_cat = bool(work.layout.yaxis and work.layout.yaxis.type == "category")
    # Un asse a CATEGORIE (es. le Linee, che impostano esplicitamente
    # xaxis.type='category') non va MAI fissato con un range NUMERICO
    # calcolato dai VALORI: se le etichette sembrano numeri (anni, ID...),
    # _nums() le converte comunque in float, e il range verrebbe calcolato
    # sul loro VALORE (es. 2020-2024) — ma un asse a categorie posiziona le
    # categorie a INDICE 0,1,2..., non ai loro valori numerici: il
    # risultato erano solo 1-2 punti visibili, schiacciati in un angolo
    # (bug segnalato dall'utente). Serve comunque FISSARE il range (in
    # unità di INDICE, non di valore) perché altrimenti l'autorange di
    # Plotly per un asse a categorie si adatta al SOTTOINSIEME di categorie
    # attualmente presenti nei dati del frame corrente (i punti non ancora
    # rivelati sono del tutto assenti dall'array, non solo invisibili) e il
    # grafico si "restringe" progressivamente man mano che i dati
    # compaiono, invece di mostrare fin da subito lo spazio per tutti.
    if ys and not y_cat:
        r = _axis_range(ys, y_log)
        if r:
            work.update_yaxes(range=r, autorange=False)
    elif y_cat and y_cat_vals:
        n_cat = len(dict.fromkeys(y_cat_vals))
        work.update_yaxes(range=[-0.5, n_cat - 0.5], autorange=False)
    if xs and not x_cat:
        r = _axis_range(xs, x_log)
        if r:
            work.update_xaxes(range=r, autorange=False)
    elif x_cat and x_cat_vals:
        n_cat = len(dict.fromkeys(x_cat_vals))
        work.update_xaxes(range=[-0.5, n_cat - 0.5], autorange=False)


def prepare_animated_figure(fig: go.Figure, frames: list[go.Frame],
                             extra_traces: list[go.Scatter] | None = None) -> go.Figure:
    """Copia di `fig` pronta per l'animazione: range assi fissato (vedi
    `_fix_axis_ranges`), frames assegnati, stato iniziale = primo frame
    (così la figura "a riposo" mostra il punto di partenza dell'animazione,
    non il grafico già completo). Usata sia dall'anteprima in-app sia
    dall'export HTML, così restano identiche.

    `extra_traces` (secondo valore di ritorno di `build_frames`): tracce
    ausiliarie (es. i quadratini di 'cubetti') che i frame si aspettano di
    trovare — vanno aggiunte QUI, prima di leggere il frame 0, così gli
    indici combaciano fin da subito."""
    work = go.Figure(fig)
    for tr in (extra_traces or []):
        work.add_trace(tr)
    _fix_axis_ranges(work)
    work.frames = frames
    if frames:
        f0 = frames[0]
        for i, d in enumerate(f0.data):
            if d:
                work.data[i].update(d)
        if f0.layout:
            work.layout.update(f0.layout)
    return work


def build_autoplay_html(fig: go.Figure, frames: list[go.Frame], frame_duration_ms: int,
                         loop: bool = False,
                         extra_traces: list[go.Scatter] | None = None) -> str:
    """HTML autonomo come STRINGA (Plotly.js inline, nessun CDN — cattura
    Playwright di Pipeline Video deve poter funzionare offline) che parte in
    animazione DA SOLO al caricamento della pagina, senza bisogno di
    cliccare alcun bottone. Usata sia per l'anteprima incorporata nella tab
    (st.components.v1.html, con loop=True — un rerun di Streamlit reincorpora
    lo stesso HTML e lo fa ripartire da capo: il loop evita che questo sembri
    un lampeggio "appare/sparisce") sia per il download (loop=False — in
    Pipeline Video l'animazione deve fare UN passaggio e poi restare ferma
    sul grafico completo, non ripartire dentro al video). `extra_traces`:
    vedi prepare_animated_figure."""
    work = prepare_animated_figure(fig, frames, extra_traces)
    animate_call = (
        "Plotly.animate(gd, null, {"
        f"frame: {{duration: {int(frame_duration_ms)}, redraw: true}}, "
        "transition: {duration: 0}, fromcurrent: false, mode: 'immediate'})"
    )
    if loop:
        total_ms = max(1, len(frames)) * int(frame_duration_ms)
        post_script = (
            "var gd = document.getElementsByClassName('plotly-graph-div')[0];"
            f"function _chartAnimPlay(){{ {animate_call}; }}"
            "setTimeout(_chartAnimPlay, 150);"
            f"setInterval(_chartAnimPlay, {total_ms + 600});"
        )
    else:
        post_script = (
            "var gd = document.getElementsByClassName('plotly-graph-div')[0];"
            f"setTimeout(function(){{ {animate_call}; }}, 150);"
        )
    return work.to_html(include_plotlyjs="inline", full_html=True,
                         post_script=post_script, auto_play=False)


def to_autoplay_html(fig: go.Figure, frames: list[go.Frame], frame_duration_ms: int, out_path,
                      extra_traces: list[go.Scatter] | None = None) -> None:
    """Scrive su file lo stesso HTML di `build_autoplay_html` (loop=False:
    per Pipeline Video, un solo passaggio)."""
    from pathlib import Path
    html = build_autoplay_html(fig, frames, frame_duration_ms, loop=False, extra_traces=extra_traces)
    Path(out_path).write_text(html, encoding="utf-8")
