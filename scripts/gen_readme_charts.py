"""Generate the README performance charts (docs/charts/*.svg).

The numbers mirror docs/performance.md; edit them here and rerun:
    python3 scripts/gen_readme_charts.py
"""
import sys
from pathlib import Path

ORANGE, GRAY = "#E8762B", "#9AA0A6"
STYLE = """<style>
text{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif;fill:#1F2328}
.muted{fill:#59636E}.grid{stroke:#D1D9E0}
@media (prefers-color-scheme:dark){text{fill:#F0F6FC}.muted{fill:#9198A1}.grid{stroke:#3D444D}}
</style>"""

LABEL_W, BAR_W, RIGHT, BAR_H, GAP = 230, 420, 70, 18, 4


def chart(title, legend, groups, vmax, ticks):
    """groups: [(label, sublabel, [(value, color, text)], note)]"""
    rows = []
    y = 64
    for label, sub, bars, note in groups:
        if label and y > 64:
            y += 14
        h = len(bars) * (BAR_H + GAP) - GAP
        rows.append(f'<text x="{LABEL_W-12}" y="{y+h/2-(3 if sub else -5)}" text-anchor="end" font-size="14" font-weight="600">{label}</text>')
        if sub:
            rows.append(f'<text x="{LABEL_W-12}" y="{y+h/2+13}" text-anchor="end" font-size="12" class="muted">{sub}</text>')
        for i, (v, color, txt) in enumerate(bars):
            by = y + i * (BAR_H + GAP)
            w = BAR_W * v / vmax
            rows.append(f'<rect x="{LABEL_W}" y="{by}" width="{w:.1f}" height="{BAR_H}" rx="3" fill="{color}"/>')
            weight = ' font-weight="600"' if i == 0 else ' class="muted"'
            rows.append(f'<text x="{LABEL_W+w+6:.1f}" y="{by+BAR_H-4}" font-size="12"{weight}>{txt}</text>')
        if note:
            rows.append(f'<text x="{LABEL_W+BAR_W+RIGHT-4}" y="{y+h/2+5}" text-anchor="end" font-size="13" font-weight="700" fill="{ORANGE}" style="fill:{ORANGE}">{note}</text>')
        y += h + 20
    height = y + 22
    grid = []
    for t in ticks:
        x = LABEL_W + BAR_W * t / vmax
        grid.append(f'<line x1="{x:.1f}" y1="56" x2="{x:.1f}" y2="{y-12}" class="grid" stroke-width="1"/>')
        grid.append(f'<text x="{x:.1f}" y="{y+4}" text-anchor="middle" font-size="11" class="muted">{t:,}</text>')
    lx, leg = LABEL_W, []
    for name, color in legend:
        leg.append(f'<rect x="{lx}" y="34" width="12" height="12" rx="2" fill="{color}"/><text x="{lx+18}" y="45" font-size="13">{name}</text>')
        lx += 30 + 7.5 * len(name)
    width = LABEL_W + BAR_W + RIGHT
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">{STYLE}'
            f'<text x="16" y="22" font-size="15" font-weight="700">{title}</text>'
            + "".join(leg + grid + rows) + "</svg>\n")


decode = chart(
    "Decode speed on RTX 5090 (tokens/s, higher is better)",
    [("Palito", ORANGE), ("llama.cpp CUDA", GRAY)],
    [("Qwen3-30B-A3B MoE", ".palito 11.1 GiB vs Q4_K_M 18.6 GiB", [(412, ORANGE, "412"), (340, GRAY, "340")], "+21%"),
     ("Llama 3.1 8B", ".palito 3.5 GiB vs IQ3_XXS", [(347, ORANGE, "347"), (341, GRAY, "341")], "+2%"),
     ("Llama 3.2 1B", "3.0 bpw 0.9 GiB vs IQ3_XXS", [(1361, ORANGE, "1,361"), (1053, GRAY, "1,053")], "+29%")],
    1400, [0, 350, 700, 1050, 1400])

rows = []
for model, p1, p2, f32, f16 in [
    ("Qwen3-30B-A3B MoE", "9.3k", "22.9k", (246.7, 141.0), (328.9, 242.3)),
    ("Llama 3.1 8B", "9.0k", "22.2k", (220.8, 135.5), (253.9, 207.8)),
    ("Qwen 3 8B", "9.3k", "22.9k", (200.2, 120.0), (241.5, 190.3))]:
    for i, prompt in enumerate((p1, p2)):
        gain = round((f16[i] / f32[i] - 1) * 100)
        rows.append((model if i == 0 else "", f"{prompt}-token prompt",
                     [(f16[i], ORANGE, f"{f16[i]:.1f}"), (f32[i], GRAY, f"{f32[i]:.1f}")], f"+{gain}%"))
longctx = chart(
    "Long-context decode, .palito 3.0 bpw (tokens/s)",
    [("FP16 KV cache (int8-fast)", ORANGE), ("F32 KV cache (f32-tiled)", GRAY)],
    rows, 350, [0, 100, 200, 300])

out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent / "docs" / "charts"
out.mkdir(parents=True, exist_ok=True)
(out / "decode-vs-llama-cpp.svg").write_text(decode)
(out / "long-context-decode.svg").write_text(longctx)
