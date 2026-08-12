"""
Builds the AP-WGAN methodology architecture figure (PNG).

Layout rule: strict top-to-bottom pipeline.  Each stage is one horizontal band
read left to right; consecutive bands are joined by a single wire routed through
an empty corridor between them.  No two wires ever cross.

Every numeric parameter is traceable to train_variants.py.
Run:  python make_arch_figure.py
Out:  ../Methodology Figure/fig_apwgan_architecture.png
"""
import os
import cairosvg

W, H = 1760, 1690

GREEN = ("#DCEFDF", "#5F9E72")
BLUE = ("#D9E8F6", "#3E6E9E")
CRIT = ("#FBE1C6", "#C87A28")
TEAL = ("#D8F0ED", "#2E8B84")
AMBER = ("#F9F0DB", "#B8862B")
ROSE = ("#F8E5EA", "#A8546B")
NOVEL = ("#FCE8D8", "#C0561F")
VIOLET = ("#E5DFF2", "#6B5B95")
INK = "#141414"
BODY = "#3A3A3A"
MUTE = "#6E6E6E"

o = []
A = o.append
MARKERS = set()


# ---------------------------------------------------------------- primitives
def esc(t):
    return t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def txt(x, y, t, cls="s", anchor="start", fill=None):
    f = f' style="fill:{fill}"' if fill else ""
    A(f'<text x="{x}" y="{y}" class="{cls}" text-anchor="{anchor}"{f}>{esc(t)}</text>')


def rrect(x, y, w, h, rx, fill, stroke, sw=1.8, extra=""):
    A(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" '
      f'stroke="{stroke}" stroke-width="{sw}" {extra}/>')


def card(x, y, w, h, pal, novel=False):
    fill, stroke = (NOVEL if novel else pal)
    rrect(x, y, w, h, 10, "#FFFFFF", stroke, 3.2 if novel else 1.8, 'filter="url(#sh)"')
    rrect(x, y, w, 28, 10, fill, "none", 0)
    rrect(x, y + 16, w, 12, 0, fill, "none", 0)


def plates(x, y, w, h, n, pal, dx=8, dy=-8, rx=6):
    fill, stroke = pal
    for i in range(n - 1, -1, -1):
        rrect(x + i * dx, y + i * dy, w, h, rx, fill, stroke, 1.6)


def slab(x, y, w, h, d, pal, op=1.0):
    fill, stroke = pal
    A(f'<g opacity="{op}">')
    A(f'<polygon points="{x},{y} {x+d},{y-d} {x+w+d},{y-d} {x+w},{y}" '
      f'fill="{fill}" stroke="{stroke}" stroke-width="1.4" opacity="0.75"/>')
    A(f'<polygon points="{x+w},{y} {x+w+d},{y-d} {x+w+d},{y+h-d} {x+w},{y+h}" '
      f'fill="{stroke}" stroke="{stroke}" stroke-width="1.4" opacity="0.5"/>')
    A(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="{fill}" '
      f'stroke="{stroke}" stroke-width="1.6"/>')
    A('</g>')


def band(x, y, w, h, pal, title):
    fill, stroke = pal
    rrect(x, y, w, h, 16, fill, stroke, 1.8, 'fill-opacity="0.28" stroke-dasharray="9 6"')
    tw = 24 + len(title) * 10.5
    cx = x + w / 2
    rrect(cx - tw / 2, y - 16, tw, 32, 16, "#FFFFFF", stroke, 1.8)
    txt(cx, y + 6, title, "mod", "middle", stroke)


def submod(x, y, w, h, pal, title):
    fill, stroke = pal
    rrect(x, y, w, h, 12, fill, stroke, 1.8, 'fill-opacity="0.45" stroke-dasharray="7 5"')
    tw = 20 + len(title) * 9.6
    cx = x + w / 2
    rrect(cx - tw / 2, y - 14, tw, 28, 14, "#FFFFFF", stroke, 1.8)
    txt(cx, y + 5, title, "sub2", "middle", stroke)


def pill(x, y, w, h, t, pal, cls="pl"):
    fill, stroke = pal
    rrect(x, y, w, h, h / 2, fill, stroke, 1.6)
    txt(x + w / 2, y + h / 2 + 4, t, cls, "middle", "#2A2A2A")


def hexa(cx, cy, w, h, pal, sw=2.0, novel=False):
    fill, stroke = (NOVEL if novel else pal)
    k = h / 2
    A(f'<polygon points="{cx-w/2+k},{cy-h/2} {cx+w/2-k},{cy-h/2} {cx+w/2},{cy} '
      f'{cx+w/2-k},{cy+h/2} {cx-w/2+k},{cy+h/2} {cx-w/2},{cy}" '
      f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}" filter="url(#sh)"/>')


def mk(c):
    MARKERS.add(c)
    return c


def arrow(d, color, w=2.4, dash=None):
    ds = f' stroke-dasharray="{dash}"' if dash else ""
    A(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{w}" '
      f'stroke-linecap="round" stroke-linejoin="round"{ds} '
      f'marker-end="url(#m{color[1:]})"/>')


def tag(x, y, t, color="#7A7A74"):
    w = 16 + len(t) * 6.5
    rrect(x - w / 2, y - 14, w, 23, 11, "#FFFFFF", "#E4E4DE", 1.2)
    txt(x, y + 2, t, "lbl", "middle", color)


def badge(x, y, t="NOVEL"):
    w = 22 + len(t) * 6.6
    rrect(x, y, w, 20, 10, "#C0561F", "none", 0)
    txt(x + w / 2, y + 14, t, "nb", "middle")


def lattice_chip(x, y, label="L_j"):
    """small marker meaning: constrained by the admissible level sets"""
    rrect(x, y, 74, 22, 11, NOVEL[0], NOVEL[1], 1.8)
    for r in range(2):
        for c in range(3):
            A(f'<circle cx="{x+11+c*7}" cy="{y+8+r*7}" r="2.1" fill="{NOVEL[1]}"/>')
    txt(x + 46, y + 15, label, "chip", "middle", NOVEL[1])


# ================================================================== title
txt(48, 56, "AP-WGAN: admissibility-preserving pseudo-absence generation", "ttl")
txt(48, 84, "Wildfire susceptibility modelling  ·  central Alberta, Canada  ·  "
            "3,370 presence records, 58 covariates", "sub")

for lx, ly, pal, lab in [(1150, 30, GREEN, "data / point sets"),
                         (1150, 58, BLUE, "generator"),
                         (1150, 86, CRIT, "critic")]:
    rrect(lx, ly, 22, 15, 4, pal[0], pal[1], 1.8)
    txt(lx + 32, ly + 12, lab, "leg")
for lx, ly, pal, lab in [(1410, 30, TEAL, "constraint and loss"),
                         (1410, 58, ROSE, "evaluation")]:
    rrect(lx, ly, 22, 15, 4, pal[0], pal[1], 1.8)
    txt(lx + 32, ly + 12, lab, "leg")
rrect(1410, 86, 22, 15, 4, NOVEL[0], NOVEL[1], 3.0)
txt(1442, 98, "contribution of this paper", "leg")

# ================================================================== A · inputs
band(40, 130, 1680, 200, GREEN, "A · INPUTS AND DOMAIN KNOWLEDGE")

card(64, 176, 517, 130, GREEN)
txt(84, 196, "Presence records", "hs", "start", "#2F6B45")
plates(478, 216, 76, 56, 3, GREEN, dx=6, dy=-6, rx=4)
txt(84, 236, "3,370 wildfire points, 1984–2024", "s")
txt(84, 260, "58 covariates", "m")
txt(84, 284, "13 reclassified  +  45 continuous", "ms")

card(621, 176, 517, 130, GREEN, novel=True)
badge(1042, 166)
txt(641, 196, "Admissible level sets", "hs", "start", "#A8460F")
for r in range(4):
    for c in range(4):
        A(f'<circle cx="{1038+c*17}" cy="{224+r*17}" r="3.4" fill="{NOVEL[1]}" '
          f'opacity="{0.9 if (r + c) % 3 else 0.32}"/>')
txt(641, 238, "L_j = { c·w_j : c = 0 … m_j−1 }", "m")
txt(641, 262, "read from the presence sample, not estimated", "s")
txt(641, 286, "m_j = 5   (9 for river distance)", "ms")

card(1178, 176, 517, 130, GREEN)
txt(1198, 196, "Background pool  B", "hs", "start", "#2F6B45")
rrect(1592, 216, 76, 56, 4, GREEN[0], GREEN[1], 1.5)
A(f'<g stroke="{GREEN[1]}" stroke-width="1.1" opacity="0.8"><path d="'
  'M1611,216 V272 M1630,216 V272 M1649,216 V272 '
  'M1592,235 H1668 M1592,253 H1668"/></g>')
txt(1198, 238, "grid + random candidates", "s")
txt(1198, 262, "≥ 10 km fire buffer, water-masked", "m")
txt(1198, 286, "|B| = 15,000    ·    provides x_real and τ", "ms")

arrow("M585,241 H617", mk(NOVEL[1]), 2.8)

# ================================================================== B · generator
band(40, 376, 1680, 290, BLUE, "B · GENERATOR  G")

slab(100, 470, 32, 112, 14, VIOLET)
txt(116, 606, "z ~ N(0, I)", "m", "middle")
txt(116, 628, "dim 32", "ms", "middle")

for i in range(3):
    slab(210 + i * 34, 460, 26, 132, 13, VIOLET)
txt(262, 606, "Trunk MLP", "hs", "middle", "#4B3E73")
txt(262, 628, "32 · 32 · 32 , LeakyReLU 0.2", "ms", "middle")

submod(420, 406, 580, 146, NOVEL, "ORDINAL HEADS  × 13")
badge(852, 396)
txt(444, 448, "one head per reclassified covariate", "mt")
pill(444, 460, 210, 28, "Gumbel-softmax", NOVEL, "pl")
pill(444, 498, 210, 28, "straight-through, τ_g = 0.5", NOVEL, "mt")
txt(690, 470, "x_j = < onehot(argmax) , L_j >", "m")
txt(690, 496, "so x_j ∈ L_j for every draw,", "s")
txt(690, 518, "for every seed  (Proposition 2)", "s")

submod(420, 570, 580, 86, BLUE, "CONTINUOUS HEAD")
pill(444, 606, 210, 28, "Linear  →  45 dims", BLUE, "pl")
txt(690, 626, "unconstrained, as in a standard WGAN-GP", "s")

plates(1100, 480, 112, 120, 3, NOVEL, dx=9, dy=-9)
badge(1142, 444)
txt(1250, 628, "x_fake ∈ R^58 ,  on-lattice by construction", "m", "middle")

arrow("M325,526 C360,526 372,480 414,478", mk(VIOLET[1]))
arrow("M325,526 C360,526 372,612 414,612", mk(VIOLET[1]))
arrow("M1004,478 C1040,478 1050,510 1092,510", mk(NOVEL[1]), 2.8)
arrow("M1004,612 C1040,612 1050,560 1092,560", mk(BLUE[1]))

# corridor 1 : level sets -> ordinal heads
arrow("M985,308 V402", mk(NOVEL[1]), 2.8)
tag(985, 346, "level sets  L_j", NOVEL[1])

# ================================================================== C · critic + GP
band(40, 712, 1680, 290, TEAL, "C · CRITIC AND DOMAIN-AWARE GRADIENT PENALTY")

card(64, 766, 336, 170, TEAL)
txt(84, 786, "Interpolant", "hs", "start", "#20605B")
txt(84, 828, "x_mix = u·x_real", "m")
txt(84, 852, "        + (1−u)·x_fake", "m")
txt(84, 882, "u ~ U(0, 1)", "m")
txt(84, 910, "x_real drawn from B", "ms")

hexa(560, 851, 280, 118, TEAL, 2.2)
txt(560, 826, "manifold check", "hs", "middle", "#20605B")
txt(560, 856, "d_7NN(x_mix) ≤ τ ?", "m", "middle")
txt(560, 884, "yes → ω = 1      no → ω = 0.1", "mt", "middle")

card(740, 766, 380, 170, TEAL)
txt(760, 786, "Penalty and critic loss", "hs", "start", "#20605B")
txt(760, 828, "GP = Σ_i ω_i (‖∇D(x_mix)‖₂ − 1)²", "m2")
txt(760, 850, "                        /  Σ_i ω_i", "m2")
txt(760, 882, "L_D = −E[D(x_real)] + E[D(x_fake)] + λ·GP", "m2")
txt(760, 910, "λ = 10 ,  Adam(1e-4, β = (0, 0.9))", "ms")

# --- critic network drawing
txt(1180, 780, "Critic  D", "hs", "start", "#8A4A10")
LAY = [(1210, 150, "58"), (1320, 100, "32"), (1430, 100, "32"), (1540, 34, "1")]
for i in range(len(LAY) - 1):
    x0, h0, _ = LAY[i]
    x1, h1, _ = LAY[i + 1]
    for f in (0.15, 0.5, 0.85):
        A(f'<line x1="{x0+30}" y1="{876-h0/2+h0*f}" x2="{x1}" y2="{876-h1/2+h1*f}" '
          f'stroke="{CRIT[1]}" stroke-width="1.0" opacity="0.5"/>')
for x, h, lab in LAY:
    slab(x, 876 - h / 2, 30, h, 14, CRIT)
    txt(x + 15, 962, lab, "mt", "middle")
A(f'<circle cx="1648" cy="876" r="22" fill="url(#gScore)" stroke="{CRIT[1]}" '
  f'stroke-width="2" filter="url(#sh)"/>')
for f in (0.2, 0.5, 0.8):
    A(f'<line x1="1584" y1="{859+34*f}" x2="1626" y2="876" '
      f'stroke="{CRIT[1]}" stroke-width="1.0" opacity="0.5"/>')
txt(1648, 884, "D", "score", "middle", "#8A4A10")
txt(1648, 962, "score", "mt", "middle")
txt(1180, 992, "n_critic 3  ·  batch 256  ·  4,500 iterations", "ms")

arrow("M404,851 H416", mk(TEAL[1]))
arrow("M704,851 H736", mk(TEAL[1]))
arrow("M1124,851 C1150,851 1160,876 1196,876", mk(TEAL[1]))

# operating points strip
txt(64, 992, "operating points", "hs", "start", "#20605B")
for i, s in enumerate(["strict  τ75  ω 0.0  k 5", "base  τ90  ω 0.1  k 7",
                       "relaxed  τ97.5  ω 0.3  k 15"]):
    x = 210 + i * 250
    A(f'<circle cx="{x}" cy="988" r="5" fill="{TEAL[1]}" opacity="{1-0.3*i}"/>')
    txt(x + 13, 992, s, "mt")

# corridor 2 : x_fake -> interpolant
arrow("M1170,646 V664 Q1170,682 1150,682 H252 Q232,682 232,700 V762",
      mk(NOVEL[1]), 2.8)
tag(400, 682, "x_fake", NOVEL[1])

# ================================================================== D · placement
band(40, 1048, 1680, 200, AMBER, "D · SPATIAL PLACEMENT AND COVARIATE ASSIGNMENT")

CELLS = [
    (64, False, "Generate", [("m", "25,000 vectors"), ("m", "→ PCA-20"),
                             ("ms", "covariate space only")]),
    (394, False, "Soft-kNN mapping", [("m", "k = 5,  weight 1/d"),
                                      ("m", "scored against B"),
                                      ("ms", "no coordinates generated")]),
    (724, False, "Border penalty", [("m", "1 − exp(−d_edge/0.08)"),
                                    ("m", "p95 cap, weighted draw"),
                                    ("ms", "n = 5,500, no replacement")]),
    (1054, True, "Covariate assignment", [("m", "IDW k = 7 + 0.05σ noise"),
                                          ("m", "project onto L_j"),
                                          ("ms", "on-lattice rate = 1.0")]),
    (1384, False, "Pseudo-absences", [("m", "5,500 points"),
                                      ("m", "coords + 58 covariates"),
                                      ("ms", "ready for evaluation")]),
]
for cx, nov, title, lines in CELLS:
    card(cx, 1092, 302, 132, AMBER, novel=nov)
    txt(cx + 20, 1112, title, "hs", "start", "#A8460F" if nov else "#8A6410")
    for i, (c, t) in enumerate(lines):
        txt(cx + 20, 1150 + i * 25, t, c)
    if nov:
        badge(cx + 218, 1082)
        lattice_chip(cx + 214, 1186)

for cx in (64, 394, 724, 1054):
    col = NOVEL[1] if cx == 1054 else AMBER[1]
    arrow(f"M{cx+306},1158 H{cx+326}", mk(col), 2.8 if cx == 1054 else 2.4)

# corridor 3 : trained generator -> generate
arrow("M1428,1004 V1006 Q1428,1020 1408,1020 H235 Q215,1020 215,1038 V1088",
      mk(BLUE[1]), 2.4)
tag(400, 1020, "training complete  ·  trained generator  G", BLUE[1])

# ================================================================== E · evaluation
band(40, 1294, 1680, 286, ROSE, "E · EVALUATION GATE")

hexa(420, 1436, 680, 200, ROSE, 4.0, novel=True)
badge(646, 1352)
txt(420, 1392, "ADMISSIBILITY GATE", "hg", "middle", "#A8460F")
txt(420, 1430, "s(x) = Σ_j min_v |x_j − v| ,   v ∈ L_j", "m", "middle")
txt(420, 1462, "AUC(−s) ≈ 0.500   →   proceed", "m", "middle")
txt(420, 1494, "AUC(−s) = 1.000 ⇒ separable by construction;", "s", "middle")
txt(420, 1514, "no accuracy computed below it is interpretable", "s", "middle")
lattice_chip(596, 1460)

card(820, 1330, 870, 106, ROSE)
txt(840, 1350, "Spatial fidelity", "hs", "start", "#8A3A50")
txt(840, 1388, "Ripley K-SSE over 15 radii", "m")
txt(840, 1414, "border fraction  ·  centroid offset  ·  grid variance  ·  "
               "mean NN distance,  against the presence pattern", "s")

card(820, 1454, 870, 106, ROSE)
txt(840, 1474, "Downstream evaluation", "hs", "start", "#8A3A50")
txt(840, 1512, "5-fold spatial block CV  ·  LogReg · Naive Bayes · CART · "
               "KNN(k=9) · SVM-RBF", "m2")
txt(840, 1540, "per-fold scaling fitted on training data only  ·  AUC and TSS  ·  "
               "3 seeds per arm  ·  mean ± SD", "s")

arrow("M762,1436 C790,1436 786,1383 812,1383", mk(ROSE[1]))
arrow("M762,1436 C790,1436 786,1507 812,1507", mk(ROSE[1]))

# corridor 4 : pseudo-absences -> gate
arrow("M1535,1228 V1248 Q1535,1264 1515,1264 H440 Q420,1264 420,1282 V1332",
      mk(NOVEL[1]), 2.8)
tag(1330, 1264, "5,500 pseudo-absences", NOVEL[1])

# ================================================================== footnote
A('<line x1="48" y1="1614" x2="1712" y2="1614" stroke="#DEDED8" stroke-width="1.4"/>')
txt(48, 1640, "τ  manifold validity radius   ·   ω  out-of-manifold penalty weight   ·   "
              "L_j  admissible level set of reclassified covariate j   ·   "
              "λ  gradient-penalty coefficient", "foot")
txt(48, 1662, "τ_g  Gumbel relaxation temperature   ·   B  background candidate pool   ·   "
              "d_edge  normalised distance to the study-area boundary   ·   "
              "the  L_j  chip marks every step constrained by the level sets", "foot")

# ================================================================== assemble
markers = "".join(
    f'<marker id="m{c[1:]}" viewBox="0 0 10 10" refX="8.4" refY="5" markerWidth="7" '
    f'markerHeight="7" orient="auto-start-reverse">'
    f'<path d="M0,1.2 L8.4,5 L0,8.8" fill="none" stroke="{c}" stroke-width="2.3" '
    f'stroke-linecap="round" stroke-linejoin="round"/></marker>' for c in MARKERS)

svg = f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}">
<rect x="0" y="0" width="{W}" height="{H}" fill="#FFFFFF"/>
<style>
.ttl{{font:600 31px Helvetica,Arial,sans-serif;fill:{INK}}}
.sub{{font:15px Helvetica,Arial,sans-serif;fill:#7A7A74}}
.mod{{font:600 15px Helvetica,Arial,sans-serif;letter-spacing:1.1px}}
.sub2{{font:600 13px Helvetica,Arial,sans-serif}}
.hs{{font:600 16px Helvetica,Arial,sans-serif}}
.hg{{font:600 20px Helvetica,Arial,sans-serif;letter-spacing:1.3px}}
.score{{font:600 24px Helvetica,Arial,sans-serif}}
.s{{font:13.5px Helvetica,Arial,sans-serif;fill:{BODY}}}
.m{{font:13px "DejaVu Sans Mono",monospace;fill:{BODY}}}
.m2{{font:11.5px "DejaVu Sans Mono",monospace;fill:{BODY}}}
.ms{{font:12px "DejaVu Sans Mono",monospace;fill:{MUTE}}}
.mt{{font:11px "DejaVu Sans Mono",monospace;fill:{MUTE}}}
.pl{{font:600 13.5px Helvetica,Arial,sans-serif}}
.chip{{font:600 12px "DejaVu Sans Mono",monospace}}
.nb{{font:600 10px Helvetica,Arial,sans-serif;fill:#FFFFFF;letter-spacing:1.2px}}
.lbl{{font:11.5px Helvetica,Arial,sans-serif}}
.leg{{font:12.5px Helvetica,Arial,sans-serif;fill:#4A4A4A}}
.foot{{font:12px "DejaVu Sans Mono",monospace;fill:#8A8A84}}
</style>
<defs>
<filter id="sh" x="-25%" y="-25%" width="150%" height="160%">
<feDropShadow dx="0" dy="2" stdDeviation="3.2" flood-color="#000000" flood-opacity="0.13"/></filter>
<radialGradient id="gScore" cx="0.35" cy="0.3" r="0.8">
<stop offset="0" stop-color="#FFFFFF"/><stop offset="1" stop-color="#F3BE86"/></radialGradient>
{markers}
</defs>
{chr(10).join(o)}
</svg>'''

here = os.path.dirname(os.path.abspath(__file__))
out = os.path.abspath(os.path.join(here, "..", "Methodology Figure"))
p = os.path.join(out, "_arch_build.svg")
open(p, "w", encoding="utf-8").write(svg)
cairosvg.svg2png(url=p, write_to=os.path.join(out, "fig_apwgan_architecture.png"),
                 output_width=W * 2, background_color="white")
cairosvg.svg2png(url=p, write_to="/tmp/prev.png", output_width=W, background_color="white")
print("done")
