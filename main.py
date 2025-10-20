from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Any

from plots import plots


def run_plot(
    plot_name: str,
    *,
    output_dir: Path,
    context_label: str | None = None,
    **kwargs: Any,
) -> Path:
    """
    Dispatch to the plotting interface and save the resulting figure.

    Args:
        plot_name: Identifier understood by ``plots.plot_interface``.
        output_dir: Directory where the plot image should be written.
        **kwargs: Additional keyword arguments forwarded to the plot factory.

    Returns:
        Path of the saved plot image.
    """

    dpi = int(kwargs.pop("dpi", 150))
    fig = plots.plot_interface(plot=plot_name, **kwargs)

    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    parts = [plot_name]
    if context_label:
        parts.append(context_label)
    if kwargs.get("envelope_key"):
        parts.append(str(kwargs["envelope_key"]))
    filename = "-".join(part for part in parts if part)
    filename = filename.replace("plot_", "")
    filepath = output_dir / f"{filename}-{timestamp}.png"

    fig.savefig(filepath, dpi=dpi)
    return filepath


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run plotting utilities.")
    parser.add_argument(
        "--plot",
        required=True,
        help="Name of the plot to generate. Can plot 'growth' or 'envelope' or 'kernel'",
    )
    parser.add_argument(
        "--envelope-path",
        dest="envelope_path",
        help="Path to a disturbance envelope YAML definition. "
        "If omitted for disturbance plots, all envelopes in data/disturbances/envelopes are used.",
    )
    parser.add_argument(
        "--kernel-path",
        dest="kernel_path",
        help="Path to a disturbance kernel YAML definition. "
        "If omitted for kernel plots, all kernels in data/disturbances/kernels are used.",
    )
    parser.add_argument(
        "--envelope-key",
        dest="envelope_key",
        help="Optional specific envelope key/class to render.",
    )
    parser.add_argument(
        "--output-dir",
        default="plots",
        help="Directory where the generated plot image will be saved.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="Resolution (dots per inch) for the saved figure.",
    )
    parser.add_argument(
        "--stand",
        dest="stand_name",
        default="ucp_baseline",
        help="Name of the growth stand profile to simulate for growth plots.",
    )
    parser.add_argument(
        "--years",
        type=float,
        default=10.0,
        help="Number of years to project when plotting growth trajectories.",
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=1.0,
        help="Time step in years for growth trajectory plots (default 1).",
    )
    parser.add_argument(
        "--compare-unthinned",
        action="store_true",
        help="When plotting growth, also simulate and overlay an unthinned stand.",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    plot_kwargs: dict[str, Any] = {}
    if args.plot == "envelope" and args.envelope_key is not None:
        plot_kwargs["envelope_key"] = args.envelope_key
    if args.plot == "growth":
        plot_kwargs["stand_name"] = args.stand_name
        plot_kwargs["years"] = args.years
        plot_kwargs["dt"] = args.dt
        if args.compare_unthinned:
            plot_kwargs["compare_unthinned"] = True

    if args.plot == "envelope" and not args.envelope_path:
        envelope_dir = Path("data/disturbances/envelopes")
        if not envelope_dir.exists():
            print(f"Envelope directory not found: {envelope_dir}")
            return

        envelope_files = sorted(
            (p for p in envelope_dir.iterdir() if p.is_file() and p.suffix in {".yaml", ".yml"}),
            key=lambda p: p.name,
        )

        if not envelope_files:
            print(f"No envelope YAML files found in {envelope_dir}")
            return

        saved_paths: list[Path] = []
        for path in envelope_files:
            try:
                per_plot_kwargs = dict(plot_kwargs)
                per_plot_kwargs["envelope_path"] = str(path)
                output_path = run_plot(
                    args.plot,
                    output_dir=output_dir,
                    context_label=path.stem,
                    dpi=args.dpi,
                    **per_plot_kwargs,
                )
            except Exception as exc:
                print(f"Skipping '{path.name}': {exc}")
                continue
            else:
                saved_paths.append(output_path)

        if saved_paths:
            print("Saved plots:")
            for saved in saved_paths:
                print(f" - {saved}")
        else:
            print("No plots were generated.")
        return

    if args.plot == "kernel" and not args.kernel_path:
        kernel_dir = Path("data/disturbances/kernels")
        if not kernel_dir.exists():
            print(f"Kernel directory not found: {kernel_dir}")
            return

        kernel_files = sorted(
            (p for p in kernel_dir.iterdir() if p.is_file() and p.suffix in {".yaml", ".yml"}),
            key=lambda p: p.name,
        )

        if not kernel_files:
            print(f"No kernel YAML files found in {kernel_dir}")
            return

        saved_paths: list[Path] = []
        for path in kernel_files:
            try:
                per_plot_kwargs = dict(plot_kwargs)
                per_plot_kwargs["kernel_path"] = str(path)
                output_path = run_plot(
                    args.plot,
                    output_dir=output_dir,
                    context_label=path.stem,
                    dpi=args.dpi,
                    **per_plot_kwargs,
                )
            except Exception as exc:
                print(f"Skipping '{path.name}': {exc}")
                continue
            else:
                saved_paths.append(output_path)

        if saved_paths:
            print("Saved plots:")
            for saved in saved_paths:
                print(f" - {saved}")
        else:
            print("No plots were generated.")
        return

    final_kwargs = dict(plot_kwargs)
    if args.envelope_path:
        final_kwargs["envelope_path"] = args.envelope_path
    if args.kernel_path:
        final_kwargs["kernel_path"] = args.kernel_path

    output_path = run_plot(
        args.plot,
        output_dir=output_dir,
        context_label=(
            Path(args.envelope_path).stem
            if args.envelope_path
            else Path(args.kernel_path).stem
            if args.kernel_path
            else args.stand_name
            if args.plot == "growth" and args.stand_name
            else None
        ),
        dpi=args.dpi,
        **final_kwargs,
    )
    print(f"Saved plot to {output_path}")


if __name__ == "__main__":
    main()
