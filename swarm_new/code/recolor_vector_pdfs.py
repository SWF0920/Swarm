"""Applies unified three-color semantic styling for Workers (W), Frontier (F), and Human Overseer (H) across vector PDF figures."""

from __future__ import annotations

import argparse
import pathlib
import re
from typing import List, Tuple

from pypdf import PdfReader, PdfWriter
from pypdf.generic import DecodedStreamObject, NameObject, RectangleObject

# Canonical Three-Role Semantic Palette
COL_W = (25, 100, 186)   # Cobalt Blue (#1964BA) -- Workers (W)
COL_F = (157, 23, 77)    # Royal Crimson-Plum (#9D174D) -- Frontier Orchestrator (F)
COL_H = (15, 118, 94)    # Forest Emerald-Teal (#0F765E) -- Human Overseer (H)

# Auxiliary non-colliding palette for schematics & outcomes
COL_SLATE = (45, 55, 72)       # Neutral frame headers (Two-tier swarm, Outcome)
COL_GATE = (71, 85, 105)       # Neutral Admission gate frame
COL_SAFE = (22, 138, 69)       # Harm detected shield
COL_WARN = (204, 82, 14)       # Harm undetected / Policy 2a absorbed
COL_LEARN = (79, 70, 205)      # Learned IPG book & team reward loop


def rgb_str(rgb: Tuple[int, int, int], digits: int = 5) -> str:
    return " ".join(f"{c / 255.0:.{digits}f}".rstrip("0").rstrip(".") or "0" for c in rgb)


def mix_white(rgb: Tuple[int, int, int], pct: float) -> Tuple[int, int, int]:
    """Mimics TikZ color!pct (pct in [0, 100] mixed with white)."""
    alpha = pct / 100.0
    return tuple(round(alpha * c + (1.0 - alpha) * 255.0) for c in rgb)  # type: ignore[return-value]


def mix_black(rgb: Tuple[int, int, int], pct: float) -> Tuple[int, int, int]:
    """Mimics TikZ color!pct!black."""
    alpha = pct / 100.0
    return tuple(round(alpha * c) for c in rgb)  # type: ignore[return-value]


def write_modified_pdf(src_pdf: pathlib.Path, dst_pdf: pathlib.Path, new_stream: str, widen_x: float = 0.0) -> None:
    reader = PdfReader(str(src_pdf))
    writer = PdfWriter()
    page = reader.pages[0]

    stream_obj = DecodedStreamObject()
    stream_obj.set_data(new_stream.encode("latin1"))
    page[NameObject("/Contents")] = writer._add_object(stream_obj)

    if widen_x != 0.0:
        mb = page.mediabox
        new_box = RectangleObject([float(mb.left), float(mb.bottom), float(mb.right) + widen_x, float(mb.top)])
        page.mediabox = new_box
        if "/CropBox" in page:
            page.cropbox = new_box

    writer.add_page(page)
    with dst_pdf.open("wb") as f:
        writer.write(f)


def replace_rgb_line(line: str, old_rgb: str, new_rgb_tuple: Tuple[int, int, int]) -> str:
    s = rgb_str(new_rgb_tuple)
    return line.replace(f"{old_rgb} rg", f"{s} rg").replace(f"{old_rgb} RG", f"{s} RG")


def recolor_fig_architecture(src: pathlib.Path, dst: pathlib.Path) -> None:
    data = PdfReader(str(src)).pages[0].get_contents().get_data().decode("latin1")
    lines = data.splitlines()

    old_primary = "0.09413 0.24315 0.45097"
    old_worker = "0.20392 0.4706 0.77647"
    old_worker_dark = "0.16313 0.37648 0.62119"
    old_worker_bdr = "0.32333 0.55 0.81"
    old_worker_dots = "0.44275 0.62941 0.84354"
    old_accent = "0.05882 0.46275 0.43137"
    old_slate = "0.2 0.2549 0.33333"
    old_ink = "0.11766 0.16078 0.23137"
    old_mute = "0.33725 0.39217 0.47842"
    old_warn = "0.76077 0.2549 0.04706"
    old_learn = "0.45882 0.25098 0.76863"
    old_learn_dark = "0.25235 0.13803 0.42274"

    for i, ln in enumerate(lines):
        # Stage 1 outer frame & title (lines 0..75): neutral dark slate
        if i <= 75:
            ln = replace_rgb_line(ln, "0.96376 0.96973 0.97804", mix_white(COL_SLATE, 4))
            ln = replace_rgb_line(ln, "0.76448 0.80322 0.85725", mix_white(COL_SLATE, 24))
            ln = replace_rgb_line(ln, old_primary, COL_SLATE)
        # Frontier card, robot, and title (lines 76..335) -> COL_F
        elif 76 <= i <= 335:
            ln = replace_rgb_line(ln, "0.3206 0.43236 0.58823", mix_white(COL_F, 75))
            ln = replace_rgb_line(ln, old_primary, COL_F)
        # Workers card, robots, W_1..W_N labels, Workers text (lines 385..1410) -> COL_W / COL_LEARN
        elif 385 <= i <= 1410:
            ln = replace_rgb_line(ln, old_worker_bdr, mix_white(COL_W, 85))
            ln = replace_rgb_line(ln, old_worker, COL_W)
            ln = replace_rgb_line(ln, old_worker_dots, mix_white(COL_W, 70))
            ln = replace_rgb_line(ln, old_worker_dark, COL_W)
            if 1225 <= i <= 1295:
                ln = replace_rgb_line(ln, old_ink, COL_W)
            ln = replace_rgb_line(ln, old_learn, COL_LEARN)
            ln = replace_rgb_line(ln, old_learn_dark, mix_black(COL_LEARN, 55))
        # Stage 2 Admission gate (lines 1411..1608) -> neutral COL_GATE, keeping 'admitted' in COL_W
        elif 1411 <= i <= 1608:
            ln = replace_rgb_line(ln, "0.94353 0.96777 0.96588", mix_white(COL_GATE, 6))
            ln = replace_rgb_line(ln, "0.64235 0.79585 0.78392", mix_white(COL_GATE, 36))
            ln = replace_rgb_line(ln, "0.34117 0.62393 0.60196", mix_white(COL_GATE, 70))
            ln = replace_rgb_line(ln, old_accent, COL_GATE)
            ln = replace_rgb_line(ln, old_worker_dark, COL_W)
        # Stage 3 Human Overseer H (lines 1609..2355) -> COL_H (with F and W envelopes inside inbox)
        elif 1609 <= i <= 2355:
            ln = replace_rgb_line(ln, "0.95999 0.96274 0.96666", mix_white(COL_H, 5))
            ln = replace_rgb_line(ln, "0.808 0.82118 0.84", mix_white(COL_H, 32))
            ln = replace_rgb_line(ln, "0.56 0.5902 0.63333", mix_white(COL_H, 55))
            ln = replace_rgb_line(ln, "0.68 0.70197 0.73334", mix_white(COL_H, 42))
            ln = replace_rgb_line(ln, "0.44 0.47842 0.53333", mix_white(COL_H, 70))
            ln = replace_rgb_line(ln, "0.52 0.55293 0.59999", mix_white(COL_H, 60))
            ln = replace_rgb_line(ln, old_slate, COL_H)
            ln = replace_rgb_line(ln, old_primary, COL_F)
            ln = replace_rgb_line(ln, old_worker, COL_W)
            if 2110 <= i <= 2200:
                ln = replace_rgb_line(ln, old_ink, COL_H)
            ln = replace_rgb_line(ln, old_learn, COL_LEARN)
            ln = replace_rgb_line(ln, old_learn_dark, mix_black(COL_LEARN, 55))
        # Stage 4 Outcome (lines 2356..2598)
        elif 2356 <= i <= 2598:
            ln = replace_rgb_line(ln, old_slate, COL_SLATE)
            ln = replace_rgb_line(ln, "0.29411 0.59706 0.57352", mix_white(COL_SAFE, 75))
            ln = replace_rgb_line(ln, "0.94353 0.96777 0.96588", mix_white(COL_SAFE, 6))
            ln = replace_rgb_line(ln, old_accent, COL_SAFE)
            ln = replace_rgb_line(ln, old_warn, COL_WARN)
        # Flows & Legend (lines 2599..end)
        else:
            if 2599 <= i <= 2625:
                ln = replace_rgb_line(ln, old_primary, COL_F)
                ln = replace_rgb_line(ln, old_ink, COL_F)
            else:
                ln = replace_rgb_line(ln, old_primary, COL_F)
            ln = replace_rgb_line(ln, old_worker, COL_W)
            ln = replace_rgb_line(ln, old_worker_dark, COL_W)
            if 2800 <= i <= 2830:
                ln = replace_rgb_line(ln, old_accent, COL_H)
            ln = replace_rgb_line(ln, old_warn, COL_WARN)
            ln = replace_rgb_line(ln, old_learn, COL_LEARN)
            if "[(F)45(rontier)-278(message)]TJ" in ln:
                c_f = rgb_str(COL_F)
                ln = ln.replace(
                    "[(F)45(rontier)-278(message)]TJ",
                    f"{c_f} rg [(or)15(c)10(hestrator)]TJ {old_mute} rg [-278(message)]TJ",
                )
            if "[(w)10(or)-15(k)20(er)-278(message)]TJ" in ln:
                c_w = rgb_str(COL_W)
                ln = ln.replace(
                    "[(w)10(or)-15(k)20(er)-278(message)]TJ",
                    f"{c_w} rg [(w)10(or)-15(k)20(er)]TJ {old_mute} rg [-278(message)]TJ",
                )

        lines[i] = ln

    write_modified_pdf(src, dst, "\n".join(lines))


def recolor_fig_policies(src: pathlib.Path, dst: pathlib.Path) -> None:
    data = PdfReader(str(src)).pages[0].get_contents().get_data().decode("latin1")
    lines = data.splitlines()

    old_primary = "0.09413 0.24315 0.45097"
    old_worker = "0.20392 0.4706 0.77647"
    old_worker_head = "0.17332 0.40001 0.66"
    old_worker_fill = "0.96815 0.97882 0.99106"
    old_worker_bdr = "0.6975 0.79883 0.91505"
    old_worker_dots = "0.44275 0.62941 0.84354"
    old_worker_mute = "0.64177 0.76176 0.89941"
    old_accent = "0.05882 0.46275 0.43137"
    old_accent_head = "0.05 0.39334 0.36665"
    old_accent_fill = "0.95294 0.97314 0.97157"
    old_accent_bdr = "0.62354 0.7851 0.77255"
    old_accent_card = "0.29411 0.59706 0.57352"
    old_slate = "0.2 0.2549 0.33333"
    old_ink = "0.11766 0.16078 0.23137"
    old_mute = "0.33725 0.39217 0.47842"
    old_warn = "0.76077 0.2549 0.04706"
    old_warn_head = "0.64665 0.21666 0.04"
    old_learn = "0.45882 0.25098 0.76863"
    old_learn_head = "0.39 0.21333 0.65334"
    old_learn_dark = "0.25235 0.13803 0.42274"

    f_label_lines = [idx for idx, l in enumerate(lines) if "[(F)]TJ" in l]
    w_label_lines = [idx for idx, l in enumerate(lines) if "[(W)]TJ" in l]
    h_label_lines = [idx for idx, l in enumerate(lines) if "[(H)]TJ" in l]

    for idx in f_label_lines:
        for k in range(max(0, idx - 6), idx):
            lines[k] = replace_rgb_line(lines[k], old_ink, COL_F)
    for idx in w_label_lines:
        for k in range(max(0, idx - 6), idx):
            lines[k] = replace_rgb_line(lines[k], old_ink, COL_W)
    for idx in h_label_lines:
        for k in range(max(0, idx - 6), idx):
            lines[k] = replace_rgb_line(lines[k], old_mute, COL_H)

    for i, ln in enumerate(lines):
        if 4025 <= i <= 4056:
            ln = replace_rgb_line(ln, old_accent, COL_SAFE)
        elif 2710 <= i <= 4024:
            if 3810 <= i <= 3822:
                ln = replace_rgb_line(ln, old_accent, COL_W)
            else:
                ln = replace_rgb_line(ln, old_accent_fill, mix_white(COL_F, 5))
                ln = replace_rgb_line(ln, old_accent_bdr, mix_white(COL_F, 38))
                ln = replace_rgb_line(ln, old_accent_card, mix_white(COL_F, 75))
                ln = replace_rgb_line(ln, old_accent_head, mix_black(COL_F, 88))
                ln = replace_rgb_line(ln, old_accent, COL_F)

        ln = replace_rgb_line(ln, old_primary, COL_F)
        ln = replace_rgb_line(ln, old_worker, COL_W)
        ln = replace_rgb_line(ln, old_worker_head, mix_black(COL_W, 88))
        ln = replace_rgb_line(ln, old_worker_fill, mix_white(COL_W, 4))
        ln = replace_rgb_line(ln, old_worker_bdr, mix_white(COL_W, 38))
        ln = replace_rgb_line(ln, old_worker_dots, mix_white(COL_W, 70))
        ln = replace_rgb_line(ln, old_worker_mute, mix_white(COL_W, 45))
        ln = replace_rgb_line(ln, old_slate, COL_H)
        ln = replace_rgb_line(ln, old_warn, COL_WARN)
        ln = replace_rgb_line(ln, old_warn_head, mix_black(COL_WARN, 85))
        ln = replace_rgb_line(ln, old_learn, COL_LEARN)
        ln = replace_rgb_line(ln, old_learn_head, mix_black(COL_LEARN, 85))
        ln = replace_rgb_line(ln, old_learn_dark, mix_black(COL_LEARN, 55))
        if "(Fr)20(ontier)" in ln:
            ln = ln.replace("203.646 4.566 Td [(Fr)20(ontier)", "201.5 4.566 Td [(Or)20(c)10(hestrator)")
        lines[i] = ln

    write_modified_pdf(src, dst, "\n".join(lines))


def recolor_fig_specialization(src: pathlib.Path, dst: pathlib.Path) -> None:
    data = PdfReader(str(src)).pages[0].get_contents().get_data().decode("latin1")
    lines = data.splitlines()
    old_mute = "0.33725 0.39217 0.47842"
    for idx, ln in enumerate(lines):
        if "[(Direct-Ask)]TJ" in ln or "[(w)10(or)-15(k)20(ers)]TJ" in ln:
            for k in range(max(0, idx - 5), idx):
                lines[k] = replace_rgb_line(lines[k], old_mute, COL_W)
        elif "[(T)120(r)-15(iage)]TJ" in ln:
            for k in range(max(0, idx - 5), idx):
                lines[k] = replace_rgb_line(lines[k], old_mute, COL_F)
        elif "[(e)30(xact)-278(IPG)]TJ" in ln or "[(sampled)-278(IPG)]TJ" in ln:
            for k in range(max(0, idx - 5), idx):
                lines[k] = replace_rgb_line(lines[k], old_mute, COL_H)
    write_modified_pdf(src, dst, "\n".join(lines))


def color_mathtext_bt_tokens(bt_block: str, base_rgb_cmd: str) -> str:
    """Colors W/F/H math symbols and role words inside a Matplotlib BT..ET block."""
    c_w = f"{rgb_str(COL_W)} rg"
    c_f = f"{rgb_str(COL_F)} rg"
    c_h = f"{rgb_str(COL_H)} rg"

    if "TJ" in bt_block and "Tj" not in bt_block:
        if "(Human reviews per episode)" in bt_block:
            return bt_block.replace(
                "[ (Human reviews per episode) ] TJ",
                f"{c_h} [ (Human) ] TJ {base_rgb_cmd} [ ( reviews per episode) ] TJ",
            )
        if "(A. Expected human reviews performed per episode)" in bt_block:
            return bt_block.replace(
                "[ (A. Expected human reviews performed per episode) ] TJ",
                f"[ (A. Expected ) ] TJ {c_h} [ (human reviews) ] TJ {base_rgb_cmd} [ ( performed per episode) ] TJ",
            )
        return bt_block

    tj_matches = list(re.finditer(r"\((?:\\.|[^\\)])*\)\s*Tj", bt_block))
    if not tj_matches:
        return bt_block

    chars = [re.match(r"\(((?:\\.|[^\\)])*)\)", m.group(0)).group(1) for m in tj_matches]  # type: ignore[union-attr]
    joined = "".join(chars)
    n = len(chars)
    char_colors: List[str | None] = [None] * n

    def mark_sub(sub: str, col: str) -> None:
        start = 0
        while True:
            pos = joined.find(sub, start)
            if pos == -1:
                break
            cur = 0
            for idx, tok in enumerate(chars):
                nxt = cur + len(tok)
                if nxt > pos and cur < pos + len(sub):
                    char_colors[idx] = col
                cur = nxt
            start = pos + len(sub)

    for w_tok in ("qW", "rW", "workers", "worker"):
        mark_sub(w_tok, c_w)

    for f_tok in ("AF", "dF", "Frontier"):
        mark_sub(f_tok, c_f)

    for h_tok in ("pH;0", "human oversees", "human oversee", "c0", "K=2"):
        if h_tok == "K=2":
            pos = joined.find("K=2")
            while pos != -1:
                cur = 0
                for idx, tok in enumerate(chars):
                    if cur == pos and tok == "K":
                        char_colors[idx] = c_h
                    cur += len(tok)
                pos = joined.find("K=2", pos + 1)
        else:
            mark_sub(h_tok, c_h)

    if "c0" in joined and "°" in joined:
        mark_sub("°", c_h)
    if "misclassification" in joined or "priority" in joined or "rW\"" in joined:
        mark_sub("\"", c_h)
    if ("decay" in joined or "fatigue" in joined) and "¸" in joined:
        mark_sub("¸", c_h)
    if "c\\(M\\)" in joined:
        pos = joined.find("c\\(M\\)")
        cur = 0
        for idx, tok in enumerate(chars):
            if cur == pos and tok == "c":
                char_colors[idx] = c_h
            cur += len(tok)

    if all(c is None for c in char_colors):
        return bt_block

    out_parts = []
    last_end = 0
    active_col = base_rgb_cmd
    for idx, m in enumerate(tj_matches):
        out_parts.append(bt_block[last_end : m.start()])
        target_col = char_colors[idx] or base_rgb_cmd
        if target_col != active_col:
            out_parts.append(f" {target_col} ")
            active_col = target_col
        out_parts.append(m.group(0))
        last_end = m.end()
    if active_col != base_rgb_cmd:
        out_parts.append(f" {base_rgb_cmd} ")
    out_parts.append(bt_block[last_end:])
    return "".join(out_parts)


def recolor_matplotlib_pdf(fname: str, src: pathlib.Path, dst: pathlib.Path) -> None:
    data = PdfReader(str(src)).pages[0].get_contents().get_data().decode("latin1")

    # 1. Harmonize curve, marker, & annotation colors across plots (matching line-wrapped RGB triplets as well)
    teal_re = re.compile(r"0\.0588235294\s+0\.462745098\s+0\.431372549\s+(rg|RG)\b")
    orange_re = re.compile(r"0\.8509803922\s+0\.2823529412\s+0\.0588235294\s+(rg|RG)\b")

    if fname in ("fig_crowding_vs_N.pdf", "fig_controls_priority.pdf"):
        # Policy 2b is Frontier filter (d_F = 0.7) -> COL_F
        data = teal_re.sub(lambda m: f"{rgb_str(COL_F)} {m.group(1)}", data)
    elif fname in ("fig_optimal_roster.pdf", "fig_trust_trap_basin.pdf"):
        # Noisy priority cap K=2 (Fig 5) and high-oversight 4-asker equilibrium (Fig 8) -> COL_H
        data = teal_re.sub(lambda m: f"{rgb_str(COL_H)} {m.group(1)}", data)

    data = orange_re.sub(lambda m: f"{rgb_str(COL_WARN)} {m.group(1)}", data)

    if fname == "fig_trust_trap_basin.pdf":
        # Shift Panel B title slightly left so 'IPG' is completely unclipped inside MediaBox
        data = data.replace("1 0 -0 1 386.3829955157 262.32 cm", "1 0 -0 1 372.0 262.32 cm")

    # 2. Color W, F, H math variables and role labels inside BT..ET blocks while preserving surrounding fill color
    rg_pattern = re.compile(r"([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+rg\b")
    bt_pattern = re.compile(r"BT.*?ET", flags=re.S)

    out_chunks = []
    pos = 0
    current_rg = "0.0862745098 0.1254901961 0.1647058824 rg"
    for m in bt_pattern.finditer(data):
        prefix = data[pos : m.start()]
        for rg_m in rg_pattern.finditer(prefix):
            current_rg = f"{rg_m.group(1)} {rg_m.group(2)} {rg_m.group(3)} rg"
        out_chunks.append(prefix)
        bt_str = m.group(0)
        is_neutral_text = any(
            k in current_rg
            for k in (
                "0.0862745098 0.1254901961 0.1647058824",
                "0.2901960784 0.3333333333 0.4078431373",
            )
        )
        if is_neutral_text:
            bt_str = color_mathtext_bt_tokens(bt_str, current_rg)
        out_chunks.append(bt_str)
        pos = m.end()
    out_chunks.append(data[pos:])

    write_modified_pdf(src, dst, "".join(out_chunks))


def main() -> None:
    bundle_dir = pathlib.Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description="Apply three-role (W, F, H) color harmonization to vector PDFs.")
    parser.add_argument(
        "--src-dir",
        type=pathlib.Path,
        default=bundle_dir,
        help="Directory containing input vector PDF figures (defaults to bundle root).",
    )
    parser.add_argument(
        "--dst-dir",
        type=pathlib.Path,
        default=bundle_dir,
        help="Directory where recolored vector PDF figures are written (defaults to bundle root).",
    )
    args = parser.parse_args()

    src_dir: pathlib.Path = args.src_dir.resolve()
    dst_dir: pathlib.Path = args.dst_dir.resolve()
    dst_dir.mkdir(parents=True, exist_ok=True)

    recolor_fig_architecture(src_dir / "fig_architecture.pdf", dst_dir / "fig_architecture.pdf")
    recolor_fig_policies(src_dir / "fig_policies.pdf", dst_dir / "fig_policies.pdf")
    recolor_fig_specialization(src_dir / "fig_specialization.pdf", dst_dir / "fig_specialization.pdf")

    for mpl_fig in (
        "fig_crowding_vs_N.pdf",
        "fig_scaling_laws.pdf",
        "fig_optimal_roster.pdf",
        "fig_review_volume.pdf",
        "fig_trust_trap_basin.pdf",
        "fig_controls_priority.pdf",
    ):
        recolor_matplotlib_pdf(mpl_fig, src_dir / mpl_fig, dst_dir / mpl_fig)

    print("Successfully recolored all 9 vector PDF figures with (colW, colF, colH).")


if __name__ == "__main__":
    main()
