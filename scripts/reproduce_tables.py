from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

INPUT_FILES = {
    "lug_train": ROOT / "results" / "final_model" / "Lug_3D" / "64_final_metric_lug.txt",
    "lug_val": ROOT / "results" / "final_model" / "Lug_3D" / "validation_metrics.txt",
    "plate_train": ROOT / "results" / "final_model" / "Plate_with_a_hole" / "64_final_metrics_plate.txt",
    "plate_val": ROOT / "results" / "final_model" / "Plate_with_a_hole" / "validation_metrics.txt",
}

ABLATION_FILES = {
    "GeoPINONet (full)": "results/final_model/Lug_3D/validation_metrics.txt",
    "DD + adaptive sampling": "results/ablation_study/Sampler/Data_driven_activeCSV_metrics.txt",
    "DD + standard sampling": "results/ablation_study/Sampler/Data_driven_simple_sampler_metrics.txt",
    "w/o Hooke consistency": "results/ablation_study/Hooke/without_hooke_metrics.txt",
    "w/o Fourier features": "results/ablation_study/Fourier/without_fourier_metrics.txt",
}

OUT_DIR = ROOT / "results" / "reproduced_tables"


HEADER_RE = re.compile(r"\[METRICS\]\s+Geo\s+(\d+)\s+\|\s+([A-Za-z_]+)")
VALUE_RE = re.compile(r"^\s*([A-Za-z0-9_]+)\s*:\s*([-+0-9.eE]+)\s*$")


METRIC_ALIASES = {
    "p50_err": "p50_u",
    "p90_err": "p90_u",
    "p95_err": "p95_u",
    "p99_err": "p99_u",
}


TABLE_ROWS = [
    ("$R^2_u$", "R2_u", 5),
    ("$R^2_{\\sigma_{\\mathrm{vM,ip}}}$", "R2_vm_ip", 3),
    ("$L_{2,\\mathrm{rel}}^u$ [\\%]", "L2_rel_u", 3),
    ("$p50_u$ [\\%]", "p50_u", 3),
    ("$p90_u$ [\\%]", "p90_u", 3),
    ("$p95_u$ [\\%]", "p95_u", 3),
    ("$p99_u$ [\\%]", "p99_u", 3),
    ("$p50_{\\sigma_{\\mathrm{vM,ip}}}$ [\\%]", "p50_vm_ip", 3),
    ("$p90_{\\sigma_{\\mathrm{vM,ip}}}$ [\\%]", "p90_vm_ip", 3),
    ("$p95_{\\sigma_{\\mathrm{vM,ip}}}$ [\\%]", "p95_vm_ip", 3),
    ("$p99_{\\sigma_{\\mathrm{vM,ip}}}$ [\\%]", "p99_vm_ip", 3),
]


LOAD_CASES = {
    "COMP": "Compression",
    "LAT": "Lateral bending",
    "compression": "Compression",
    "lateral_bending": "Lateral bending",
}


def parse_metrics_file(path: Path) -> pd.DataFrame:
    """
    Parse GeoPINONet metric logs with blocks of the form:

    [METRICS] Geo 1 | COMP
       L2_rel_u : ...
       R2_u     : ...
       ...

    Returns one row per geometry/load case.
    """
    if not path.exists():
        raise FileNotFoundError(f"Missing metrics file: {path}")

    rows: List[Dict[str, float | int | str]] = []
    current: Dict[str, float | int | str] | None = None

    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        header = HEADER_RE.search(line)
        if header:
            if current is not None:
                rows.append(current)

            geo_id = int(header.group(1))
            load_case_raw = header.group(2)
            load_case = LOAD_CASES.get(load_case_raw, load_case_raw)

            current = {
                "geometry_id": geo_id,
                "load_case": load_case,
            }
            continue

        if current is None:
            continue

        value = VALUE_RE.match(line)
        if value:
            key = value.group(1)
            key = METRIC_ALIASES.get(key, key)

            try:
                current[key] = float(value.group(2))
            except ValueError:
                pass

    if current is not None:
        rows.append(current)

    if not rows:
        raise ValueError(
            f"No metric blocks found in {path}. "
            "Expected lines like: [METRICS] Geo 1 | COMP"
        )

    return pd.DataFrame(rows)


def summarize(series: pd.Series, decimals: int) -> str:
    values = pd.to_numeric(series, errors="coerce").dropna()

    if values.empty:
        return "--"

    mean = values.mean()
    std = values.std(ddof=1) if len(values) > 1 else 0.0
    vmin = values.min()
    vmax = values.max()

    return (
        f"${mean:.{decimals}f} \\pm {std:.{decimals}f}$"
        f" \\; [${vmin:.{decimals}f},\\,{vmax:.{decimals}f}$]"
    )


def build_summary_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    out_rows = []

    for latex_name, metric_key, decimals in TABLE_ROWS:
        row = {"Metric": latex_name}

        for load_case in ["Compression", "Lateral bending"]:
            subset = df[df["load_case"] == load_case]

            if metric_key in subset.columns:
                row[load_case] = summarize(subset[metric_key], decimals)
            else:
                row[load_case] = "--"

        out_rows.append(row)

    return pd.DataFrame(out_rows)


def dataframe_to_latex_table(
    summary: pd.DataFrame,
    caption: str,
    label: str,
) -> str:
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        "\\small",
        "\\begin{tabular}{lcc}",
        "\\toprule",
        "Metric & Compression & Lateral bending \\\\",
        "\\midrule",
    ]

    for _, row in summary.iterrows():
        lines.append(
            f"{row['Metric']}\n"
            f"& {row['Compression']}\n"
            f"& {row['Lateral bending']} \\\\"
        )

    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table}",
            "",
        ]
    )

    return "\n".join(lines)


def caption_and_label(name: str) -> tuple[str, str]:
    captions = {
        "lug_train": (
            "In-sample accuracy on the 3D lug family over the training "
            "geometries (mean $\\pm$ std, with range in brackets; dense "
            "evaluation on the full FEM mesh).",
            "tab:lug_train_metrics",
        ),
        "lug_val": (
            "Zero-shot validation accuracy on the 3D lug family over the unseen "
            "validation geometries (mean $\\pm$ std, with range in brackets; "
            "dense evaluation on the full FEM mesh).",
            "tab:lug_val_metrics",
        ),
        "plate_train": (
            "In-sample accuracy on the plate-with-hole family over the training "
            "geometries (mean $\\pm$ std, with range in brackets; dense "
            "evaluation on the full FEM mesh).",
            "tab:plate_train_metrics",
        ),
        "plate_val": (
            "Zero-shot validation accuracy on the plate-with-hole family over "
            "the unseen validation geometries (mean $\\pm$ std, with range in "
            "brackets; dense evaluation on the full FEM mesh).",
            "tab:plate_val_metrics",
        ),
    }

    return captions[name]

def mean_metric(df: pd.DataFrame, load_case: str, metric: str) -> float:
    subset = df[df["load_case"] == load_case]
    if metric not in subset.columns:
        raise KeyError(f"Metric '{metric}' not found for load case '{load_case}'.")
    return pd.to_numeric(subset[metric], errors="coerce").dropna().mean()


def build_ablation_compact_table() -> str:
    ablation_files = {
        "GeoPINONet (full)": ROOT / "results" / "final_model" / "Lug_3D" / "validation_metrics.txt",
        "DD + adaptive sampling": ROOT / "results" / "ablation_study" / "Sampler" / "Data_driven_activeCSV_metrics.txt",
        "DD + standard sampling": ROOT / "results" / "ablation_study" / "Sampler" / "Data_driven_simple_sampler_metrics.txt",
        "w/o Hooke consistency": ROOT / "results" / "ablation_study" / "Hooke" / "without_hooke_metrics.txt",
        "w/o Fourier features": ROOT / "results" / "ablation_study" / "Fourier" / "without_fourier_metrics.txt",
    }

    rows = []

    for config_name, path in ablation_files.items():
        df = parse_metrics_file(path)

        comp_r2u = mean_metric(df, "Compression", "R2_u")
        comp_r2vm = mean_metric(df, "Compression", "R2_vm_ip")
        comp_p95 = mean_metric(df, "Compression", "p95_vm_ip")
        comp_p99 = mean_metric(df, "Compression", "p99_vm_ip")

        lat_r2u = mean_metric(df, "Lateral bending", "R2_u")
        lat_r2vm = mean_metric(df, "Lateral bending", "R2_vm_ip")
        lat_p95 = mean_metric(df, "Lateral bending", "p95_vm_ip")
        lat_p99 = mean_metric(df, "Lateral bending", "p99_vm_ip")

        rows.append(
            (
                config_name,
                comp_r2u,
                comp_r2vm,
                comp_p95,
                comp_p99,
                lat_r2u,
                lat_r2vm,
                lat_p95,
                lat_p99,
            )
        )

    lines = [
        "\\begin{table}[htb]",
        "\\centering",
        "\\caption{Ablation study on the 3D lug zero-shot benchmark. Values are",
        "averages over 16 held-out validation geometries. The data-driven",
        "variant with adaptive sampling achieves competitive stress percentiles",
        "relative to the full model, but at the cost of displacement accuracy",
        "and tensorial integrity (see text). The Hooke ablation is complemented",
        "by a tensor-integrity analysis reported separately, since its primary",
        "failure mode is not captured by scalar von~Mises metrics alone.}",
        "\\label{tab:ablation_compact}",
        "\\setlength{\\tabcolsep}{4pt}",
        "\\small",
        "\\begin{tabular}{lcccc|cccc}",
        "\\toprule",
        "& \\multicolumn{4}{c|}{Compression} & \\multicolumn{4}{c}{Lateral bending} \\\\",
        "\\cmidrule(lr){2-5}\\cmidrule(lr){6-9}",
        "Configuration",
        "& $R^2_u$",
        "& $R^2_{\\sigma_{\\mathrm{vM,ip}}}$",
        "& $p95_{\\sigma_{\\mathrm{vM,ip}}}$",
        "& $p99_{\\sigma_{\\mathrm{vM,ip}}}$",
        "& $R^2_u$",
        "& $R^2_{\\sigma_{\\mathrm{vM,ip}}}$",
        "& $p95_{\\sigma_{\\mathrm{vM,ip}}}$",
        "& $p99_{\\sigma_{\\mathrm{vM,ip}}}$ \\\\",
        "\\midrule",
    ]

    for (
        config_name,
        comp_r2u,
        comp_r2vm,
        comp_p95,
        comp_p99,
        lat_r2u,
        lat_r2vm,
        lat_p95,
        lat_p99,
    ) in rows:
        lines.append(
            f"{config_name}\n"
            f"& ${comp_r2u:.4f}$ & ${comp_r2vm:.4f}$ & ${comp_p95:.2f}\\%$ & ${comp_p99:.2f}\\%$\n"
            f"& ${lat_r2u:.4f}$ & ${lat_r2vm:.4f}$ & ${lat_p95:.2f}\\%$ & ${lat_p99:.2f}\\%$ \\\\"
        )

    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table}",
            "",
        ]
    )

    return "\n".join(lines)

def build_vm_vs_vmip_all_table() -> str:
    """
    Build a compact LaTeX table comparing full 3D von Mises metrics
    against in-plane von Mises metrics for both in-sample and zero-shot
    benchmarks.

    Uses:
        - INPUT_FILES["lug_train"]
        - INPUT_FILES["lug_val"]
        - INPUT_FILES["plate_train"]
        - INPUT_FILES["plate_val"]
    """
    dataset_map = [
        ("In-sample", "Lug 3D", INPUT_FILES["lug_train"]),
        ("Zero-shot", "Lug 3D", INPUT_FILES["lug_val"]),
        ("In-sample", "Plate-with-hole", INPUT_FILES["plate_train"]),
        ("Zero-shot", "Plate-with-hole", INPUT_FILES["plate_val"]),
    ]

    rows = []

    for split_name, benchmark_name, path in dataset_map:
        df = parse_metrics_file(path)

        for load_case in ["Compression", "Lateral bending"]:
            rows.append(
                (
                    split_name,
                    benchmark_name,
                    load_case,
                    mean_metric(df, load_case, "R2_vm"),
                    mean_metric(df, load_case, "p95_vm"),
                    mean_metric(df, load_case, "p99_vm"),
                    mean_metric(df, load_case, "R2_vm_ip"),
                    mean_metric(df, load_case, "p95_vm_ip"),
                    mean_metric(df, load_case, "p99_vm_ip"),
                )
            )

    lines = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\caption{Comparison between full 3D von~Mises metrics, "
        "$\\sigma_{\\mathrm{vM}}$, and in-plane von~Mises metrics, "
        "$\\sigma_{\\mathrm{vM,ip}}$, for both in-sample and zero-shot "
        "evaluation. Values denote means over the corresponding geometry sets.}",
        "\\label{tab:vm_vs_vmip_all}",
        "\\small",
        "\\begin{tabular}{lllcccccc}",
        "\\toprule",
        "Split & Benchmark & Load case "
        "& $R^2_{\\sigma_{\\mathrm{vM}}}$ "
        "& $p95_{\\sigma_{\\mathrm{vM}}}$ [\\%] "
        "& $p99_{\\sigma_{\\mathrm{vM}}}$ [\\%] "
        "& $R^2_{\\sigma_{\\mathrm{vM,ip}}}$ "
        "& $p95_{\\sigma_{\\mathrm{vM,ip}}}$ [\\%] "
        "& $p99_{\\sigma_{\\mathrm{vM,ip}}}$ [\\%] \\\\",
        "\\midrule",
    ]

    current_split = None
    for (
        split_name,
        benchmark_name,
        load_case,
        r2_vm,
        p95_vm,
        p99_vm,
        r2_vm_ip,
        p95_vm_ip,
        p99_vm_ip,
    ) in rows:
        if current_split is not None and split_name != current_split:
            lines.append("\\midrule")
        current_split = split_name

        lines.append(
            f"{split_name} & {benchmark_name} & {load_case} "
            f"& ${r2_vm:.3f}$ "
            f"& ${p95_vm:.2f}\\%$ "
            f"& ${p99_vm:.2f}\\%$ "
            f"& ${r2_vm_ip:.3f}$ "
            f"& ${p95_vm_ip:.2f}\\%$ "
            f"& ${p99_vm_ip:.2f}\\%$ \\\\"
        )

    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table*}",
            "",
        ]
    )

    return "\n".join(lines)
def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    all_tables = []

    for name, path in INPUT_FILES.items():
        print(f"[INFO] Reading {name}: {path}")

        df = parse_metrics_file(path)
        raw_out = OUT_DIR / f"{name}_raw_metrics.csv"
        df.to_csv(raw_out, index=False)

        summary = build_summary_dataframe(df)
        summary_out = OUT_DIR / f"{name}_summary.csv"
        summary.to_csv(summary_out, index=False)

        caption, label = caption_and_label(name)
        latex = dataframe_to_latex_table(summary, caption, label)

        tex_out = OUT_DIR / f"{name}_table.tex"
        tex_out.write_text(latex, encoding="utf-8")

        all_tables.append(f"% ===== {name} =====\n{latex}")

        print(f"[OK] Raw metrics     -> {raw_out}")
        print(f"[OK] Summary table   -> {summary_out}")
        print(f"[OK] LaTeX table     -> {tex_out}")

    all_tex = "\n\n".join(all_tables)
    all_tex_out = OUT_DIR / "all_reproduced_tables.tex"
    all_tex_out.write_text(all_tex, encoding="utf-8")

    print(f"\n[DONE] All tables written to: {OUT_DIR}")
    print(f"[DONE] Combined LaTeX file: {all_tex_out}")

    ablation_tex = build_ablation_compact_table()
    ablation_out = OUT_DIR / "ablation_compact_table.tex"
    ablation_out.write_text(ablation_tex, encoding="utf-8")
    print(f"[OK] Ablation compact table -> {ablation_out}")

    vm_compare_tex = build_vm_vs_vmip_all_table()
    vm_compare_out = OUT_DIR / "vm_vs_vmip_all_table.tex"
    vm_compare_out.write_text(vm_compare_tex, encoding="utf-8")
    print(f"[OK] VM vs VM_IP full comparison table -> {vm_compare_out}")


if __name__ == "__main__":
    main()