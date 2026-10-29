# GUI Grafici (web)

App Streamlit per creare grafici e mappe da file Excel/CSV: barre, linee, torte (anche a doppio anello), radar, heatmap, treemap, imbuto e altri, con motore Plotly interattivo e animazioni esportabili.

- Palette predefinite, palette personalizzate (editor nella tab Personalizzazione) e gradienti per mappe e matrici
- Mappe coropletiche, a bolle, a bande e a puntine (Piemonte, Val d'Aosta, Italia, Europa)
- Download in PNG, JPG e SVG

## Avvio locale

```
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Streamlit Cloud

File principale: `streamlit_app.py`. I pacchetti di sistema (GEOS per le mappe, Chromium per l'export delle immagini Plotly) sono in `packages.txt`.

Le mappe per CAP di tutta Italia richiedono `xGUI/cap_ita.geojson` (oltre 200 MB, non incluso): le mappe CAP di Piemonte/Torino usano i file pre-filtrati già presenti, generabili con `prep_geo.py`.
