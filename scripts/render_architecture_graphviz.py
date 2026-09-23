"""
scripts/render_architecture_graphviz.py — architecture figure in a plain technical style.

    pip install graphviz          # plus the Graphviz system package, so that `dot` is on PATH
    python scripts/render_architecture_graphviz.py --out docs/assets/architecture --format png

Renders the fast loop (per-frame continuation) against the slow loop (event-driven relocalization).
Bracketed figures are Level 2 ablation deltas: the RMSE change on GeoTest1 when that component is
removed (see docs/level2_evaluation.md and out/ablations/summary.md).

Deliberately unstyled: right-angle edges, square corners, no fills, one typeface, black on white.
The .gv source is written next to the image, so the figure can also be rendered with plain
`dot -Tpdf docs/assets/architecture.gv -o architecture.pdf` (LaTeX-friendly vector output).

The matplotlib version of the same structure is scripts/render_architecture.py; the Mermaid source
is docs/architecture.md.
"""

import argparse
import sys

try:
    from graphviz import Digraph
except ImportError:                                            # pragma: no cover
    sys.exit("graphviz is not installed. Run: pip install graphviz   (and install the Graphviz binaries)")

FONT = "Helvetica"          # swap for "Times-Roman" to match a LaTeX document
MONO = "Courier"


def node_label(title, detail, ablation=None):
    """Title, a detail line, and an optional bracketed ablation delta."""
    rows = [f'<TR><TD ALIGN="CENTER"><B>{title}</B></TD></TR>',
            f'<TR><TD ALIGN="CENTER"><FONT POINT-SIZE="9">{detail}</FONT></TD></TR>']
    if ablation:
        rows.append(f'<TR><TD ALIGN="CENTER"><FONT POINT-SIZE="9" FACE="{MONO}">[{ablation}]</FONT></TD></TR>')
    return "<<TABLE BORDER=\"0\" CELLBORDER=\"0\" CELLSPACING=\"1\">" + "".join(rows) + "</TABLE>>"


def build():
    g = Digraph("uav_visloc", format="png")
    g.attr(rankdir="LR", splines="ortho", nodesep="0.35", ranksep="0.55", bgcolor="white",
           fontname=FONT, fontsize="11", labelloc="b", labeljust="l")
    g.attr("node", shape="box", style="", fontname=FONT, fontsize="11", color="black",
           penwidth="1.0", margin="0.12,0.08")
    g.attr("edge", fontname=FONT, fontsize="9", color="black", arrowsize="0.7", penwidth="1.0")

    with g.subgraph(name="cluster_in") as c:
        c.attr(label="INPUTS", fontsize="11", fontname=FONT, style="solid", color="black", penwidth="1.0")
        c.node("cam", node_label("Camera frame", "nadir, north-up, 1920&#215;1080"))
        c.node("map", node_label("Satellite map", "landmark catalogue (SAHI YOLO, offline)"))
        c.node("nav", node_label("NavSource", "leg heading (FC-log replay);<BR/>scale prior 0.5, locked once at boot"))

    with g.subgraph(name="cluster_fast") as c:
        c.attr(label="FAST LOOP — every frame (386 of 401; ~58 ms median, RTX 5060)",
               fontsize="11", fontname=FONT, style="solid", color="black", penwidth="1.0")
        c.node("yolo", node_label("YOLO landmarks", "building / tree / infrastructure"))
        c.node("trk", node_label("Pixel tracker", "persistent IDs, prediction, coasting", "A_coast: +40%"))
        c.node("pairs", node_label("Bound pairs", "track &#8596; map landmark (set at last commit)"))
        c.node("ransac", node_label("Deterministic RANSAC", "translation consensus at locked scale", "A1: +68%"))
        c.node("clamp", node_label("Clamp + estimate", "offset absorption; 15/8/4 px per frame<BR/>in the NavSource leg frame",
                                   "A_prior: +138%"))
        for a, b in (("yolo", "trk"), ("trk", "pairs"), ("pairs", "ransac"), ("ransac", "clamp")):
            c.edge(a, b)

    with g.subgraph(name="cluster_slow") as c:
        c.attr(label="SLOW LOOP — event-driven relocalization (15 of 401; ~355 ms candidate generation)",
               fontsize="11", fontname=FONT, style="solid", color="black", penwidth="1.0")
        c.node("graph", node_label("Video route graph", "landmark route, &#8804;10 vertices"))
        c.node("astar", node_label("Viewport + A*", "class-matched map routes in footprint"))
        c.node("cmp", node_label("Graph comparison", "internal angles + edge-length ratios", "A3b: +199%"))
        c.node("topsis", node_label("TOPSIS arbitration", "shape / ratio / constellation affinity", "A2: +26% / +50%"))
        c.node("commit", node_label("Commit", "new bindings; one-time scale lock at boot"))
        for a, b in (("graph", "astar"), ("astar", "cmp"), ("cmp", "topsis"), ("topsis", "commit")):
            c.edge(a, b)

    g.edge("cam", "yolo")
    g.edge("map", "astar")
    g.edge("nav", "clamp", style="dashed", constraint="false")
    g.edge("nav", "graph", style="dashed")
    g.edge("pairs", "graph", label="TRIGGER: boot / tracking lost / ≤2 bound pairs",
           penwidth="1.6", constraint="false")
    g.edge("commit", "pairs", label="REBIND: new track ↔ map pairs",
           penwidth="1.6", style="dashed", constraint="false")

    g.attr(label=(
        "\\lKnown limits (Level 2 evidence):\\l"
        "  1. The handoff trigger counts bound pairs, not their quality: 3 pairs with 1 inlier, or 8 stale pairs, never fire a search.\\l"
        "  2. The 8 px/frame lateral clamp paces both drift and recovery; the waypoint-derived heading prior jumps 23 deg against a 7.5 deg imaged turn.\\l"
        "  3. Translation-only solver (camera assumed north-up nadir); ground truth is an uncorrected 4-waypoint polyline, so ATE is cross-track only.\\l"
        "  4. Ablation deltas are RMSE changes measured on one sequence (GeoTest1, 401 frames, Google Earth render).\\l"))
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/assets/architecture", help="output path without extension")
    ap.add_argument("--format", default="png", choices=["png", "pdf", "svg", "eps"])
    ap.add_argument("--dpi", default="200", help="raster resolution for png")
    args = ap.parse_args()

    g = build()
    g.format = args.format
    if args.format == "png":
        g.attr(dpi=args.dpi)
    path = g.render(filename=args.out, cleanup=False)          # keeps the .gv source next to the image
    print(f"[ARCH] wrote {path} (+ {args.out}.gv)")


if __name__ == "__main__":
    main()
