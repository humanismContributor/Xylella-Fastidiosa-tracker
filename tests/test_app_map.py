"""Map rendering rules for app.py, checked on the source: importing app.py runs the
Streamlit app, and the test setup has no folium or Streamlit."""
import re
from pathlib import Path

APP = (Path(__file__).resolve().parents[1] / "app.py").read_text(encoding="utf-8")

# Leaflet plugin shapes that draw only on SVG; on a canvas map they throw
# "_setPath is not a function" and the whole map stays blank
SVG_ONLY = ["RegularPolygonMarker", "BoatMarker", "SemiCircle", "PolyLineTextPath"]


def test_canvas_map_uses_no_svg_only_shapes():
    assert "prefer_canvas=True" in APP, "the overview map is expected to render on canvas"
    used = [name for name in SVG_ONLY if re.search(rf"\b{name}\(", APP)]
    assert not used, f"{used} cannot draw on a canvas map; use CircleMarker or a DivIcon marker"


def test_unreviewed_records_are_marked_and_get_no_spread_rings():
    projection = APP[APP.index("def add_projection"):APP.index("def add_countries")]
    assert "unreviewed(r)" in projection, "rings around an unchecked automatic record would overstate it"
    assert "UNREVIEWED_NOTE" in APP[APP.index("def record_popup"):APP.index("def add_records")]
