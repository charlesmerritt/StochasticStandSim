from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Any
import gymnasium as gym
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
        "--model-path",
        dest="model_path",
        help="Path to a Stable Baselines3 policy (.zip) when using plot=policy_rollout.",
    )
    parser.add_argument(
        "--fixed-rotation",
        type=float,
        default=25.0,
        help="Rotation length in years for fixed policy plots.",
    )
    parser.add_argument(
        "--fixed-thin-years",
        default="10,18",
        help="Comma-separated relative years for thinning in the fixed policy.",
    )
    parser.add_argument(
        "--fixed-fert-years",
        default="12,20",
        help="Comma-separated relative years for fertilisation in the fixed policy.",
    )
    parser.add_argument(
        "--envelope-key",
        dest="envelope_key",
        help="Optional specific envelope key/class to render.",
    )
    parser.add_argument(
        "--invert",
        action="store_true",
        help="Invert envelope plot (show 1−p so higher severities plot above mild).",
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
    parser.add_argument(
        "--disturbance",
        choices=["fire", "wind"],
        default="fire",
        help="Disturbance type for comparison plot.",
    )
    parser.add_argument("--dist-start", type=float, default=15.0, help="Disturbance start age.")
    parser.add_argument("--severity", type=float, help="Fixed severity in (0,1). If omitted, uses --seed.")
    parser.add_argument("--seed", type=int, default=123, help="RNG seed for disturbance.")
    parser.add_argument(
    "--si25",
    type=float,
    default=60.0,
    help="Site index (base age 25, ft) for the stand used in comparison plots.",
    )
    parser.add_argument(
        "--tpa0",
        type=float,
        default=600.0,
        help="Initial trees per acre for the stand used in comparison plots.",
    )
    parser.add_argument(
        "--rollout-steps",
        type=int,
        default=200,
        help="Maximum number of steps to simulate for policy rollout plots.",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Use deterministic policy actions for policy rollout plots.",
    )
    parser.add_argument(
        "--discount-rate",
        dest="discount_rate",
        type=float,
        help="Override EnvConfig.discount_rate for policy rollout plots.",
    )
    parser.add_argument(
        "--horizon-years",
        dest="horizon_years",
        type=float,
        help="Override EnvConfig.horizon_years for policy rollout plots.",
    )
    parser.add_argument(
        "--age0",
        dest="age0",
        type=float,
        help="Override EnvConfig.age0 for policy rollout plots.",
    )
    parser.add_argument(
        "--rng-seed",
        dest="rng_seed",
        type=int,
        help="Override EnvConfig RNG seed for policy rollout plots.",
    )
    parser.add_argument(
        "--disturbances",
        action="store_true",
        help="Enable disturbances when simulating rollouts.",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    plot_kwargs: dict[str, Any] = {}

    def _parse_years_list(value: str) -> list[float]:
        if not value:
            return []
        return [float(part.strip()) for part in value.split(",") if part.strip()]

    env_overrides: dict[str, Any] = {}
    if args.discount_rate is not None:
        env_overrides["discount_rate"] = args.discount_rate
    if args.horizon_years is not None:
        env_overrides["horizon_years"] = args.horizon_years
    if args.age0 is not None:
        env_overrides["age0"] = args.age0
    if args.dt is not None:
        env_overrides["dt"] = args.dt
    if args.si25 is not None:
        env_overrides["si25"] = args.si25
    if args.tpa0 is not None:
        env_overrides["tpa0"] = args.tpa0
    if args.rng_seed is not None:
        env_overrides["rng_seed"] = args.rng_seed
    if args.disturbances:
        env_overrides["disturbance_enabled"] = True
    if args.plot == "envelope" and args.envelope_key is not None:
        plot_kwargs["envelope_key"] = args.envelope_key
    if args.plot == "growth":
        plot_kwargs["stand_name"] = args.stand_name
        plot_kwargs["years"] = args.years
        plot_kwargs["dt"] = args.dt
        if args.compare_unthinned:
            plot_kwargs["compare_unthinned"] = True
    if args.plot == "compare":
        plot_kwargs["disturbance"] = args.disturbance
        plot_kwargs["start_age"] = args.dist_start
        plot_kwargs["seed"] = args.seed
        if args.severity is not None:
            plot_kwargs["severity"] = args.severity
        plot_kwargs["years"] = args.years
        plot_kwargs["dt"] = args.dt
        plot_kwargs["si25"] = args.si25
        plot_kwargs["tpa0"] = args.tpa0
    if args.plot == "policy_rollout":
        if not args.model_path:
            parser.error("--model-path is required when plot=policy_rollout")
        plot_kwargs["model_path"] = args.model_path
        plot_kwargs["steps"] = args.rollout_steps
        plot_kwargs["deterministic"] = args.deterministic
        if env_overrides:
            plot_kwargs["env_overrides"] = env_overrides
    if args.plot == "fixed_policy":
        plot_kwargs["steps"] = args.rollout_steps
        plot_kwargs["rotation_years"] = args.fixed_rotation
        plot_kwargs["thin_years"] = _parse_years_list(args.fixed_thin_years)
        plot_kwargs["fert_years"] = _parse_years_list(args.fixed_fert_years)
        if env_overrides:
            plot_kwargs["env_overrides"] = env_overrides
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
        
        if args.plot == "envelope" and args.invert:
            plot_kwargs["invert"] = True

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
