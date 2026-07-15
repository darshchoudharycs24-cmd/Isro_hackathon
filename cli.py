#!/usr/bin/env python3
"""
cli.py - Command-line interface for the LISS-IV preprocessing pipeline.

Usage::

    python cli.py \\
        --current  backend/data/current/image.tif \\
        --historical backend/data/historical/ \\
        --sar      backend/data/sar/s1.tif \\
        --cloud-mask backend/data/masks/mask.png \\
        --output   backend/data/output/ \\
        --date     2023-06-15

All arguments except --current and --historical are optional.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import date as _date
from datetime import datetime
from pathlib import Path

# ─── Set up basic logging before imports ────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("cli")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="cli.py",
        description="ISRO LISS-IV Cloud Removal — Preprocessing Pipeline CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Minimal (current + historical only)
  python cli.py --current data/current/img.tif --historical data/historical/

  # Full run with SAR, cloud mask, and output directory
  python cli.py \\
      --current  data/current/img.tif \\
      --historical data/historical/ \\
      --sar      data/sar/s1.tif \\
      --cloud-mask data/masks/mask.png \\
      --output   data/outputs/ \\
      --date     2023-06-15

  # Custom config
  python cli.py --current img.tif --historical hist/ \\
      --config backend/configs/pipeline_config.yaml
        """,
    )

    parser.add_argument(
        "--current", "-c",
        required=True,
        metavar="PATH",
        help="Path to current (possibly cloudy) LISS-IV image (GeoTIFF/PNG/JPEG).",
    )
    parser.add_argument(
        "--historical", "-H",
        required=True,
        metavar="DIR",
        help="Directory containing one or more cloud-free historical images.",
    )
    parser.add_argument(
        "--sar", "-s",
        default=None,
        metavar="PATH",
        help="(Optional) Path to Sentinel-1 SAR GeoTIFF.",
    )
    parser.add_argument(
        "--cloud-mask", "-m",
        default=None,
        metavar="PATH",
        dest="cloud_mask",
        help="(Optional) Path to cloud mask PNG/TIF from the Cloud Detection module.",
    )
    parser.add_argument(
        "--output", "-o",
        default="backend/data/output",
        metavar="DIR",
        help="Output directory root (default: backend/data/output).",
    )
    parser.add_argument(
        "--date", "-d",
        default=None,
        metavar="YYYY-MM-DD",
        help="Acquisition date of the current image for temporal scoring.",
    )
    parser.add_argument(
        "--config", "-C",
        default=None,
        metavar="PATH",
        help="Path to pipeline_config.yaml (default: backend/configs/pipeline_config.yaml).",
    )
    parser.add_argument(
        "--sample-id",
        default=None,
        metavar="ID",
        help="Sample identifier used for the output sub-directory. "
             "Defaults to the current image filename stem.",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable DEBUG logging.",
    )

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # ── Validate inputs ────────────────────────────────────────────────────
    current_path = Path(args.current)
    historical_dir = Path(args.historical)

    if not current_path.exists():
        logger.error("Current image not found: %s", current_path)
        return 1

    if not historical_dir.is_dir():
        logger.error("Historical directory not found: %s", historical_dir)
        return 1

    sar_path = Path(args.sar) if args.sar else None
    if sar_path and not sar_path.exists():
        logger.warning("SAR file not found: %s — continuing without SAR.", sar_path)
        sar_path = None

    cloud_mask_path = Path(args.cloud_mask) if args.cloud_mask else None
    if cloud_mask_path and not cloud_mask_path.exists():
        logger.warning("Cloud mask not found: %s — continuing without mask.", cloud_mask_path)
        cloud_mask_path = None

    # ── Parse date ─────────────────────────────────────────────────────────
    acq_date: _date | None = None
    if args.date:
        try:
            acq_date = datetime.strptime(args.date, "%Y-%m-%d").date()
        except ValueError:
            logger.warning("Invalid date format '%s'. Expected YYYY-MM-DD.", args.date)

    sample_id = args.sample_id or current_path.stem

    # ── Load pipeline ──────────────────────────────────────────────────────
    try:
        from backend.preprocessing.pipeline import PreprocessingPipeline  # noqa: PLC0415
        from backend.preprocessing.exporter import export_for_reconstruction  # noqa: PLC0415
        from backend.preprocessing.cloud_mask import load_cloud_mask  # noqa: PLC0415
    except ImportError as exc:
        logger.error("Import failed: %s\nEnsure you are running from the project root.", exc)
        return 1

    pipeline = PreprocessingPipeline(args.config)
    output_dir = Path(args.output)

    logger.info("=" * 60)
    logger.info("LISS-IV Preprocessing Pipeline")
    logger.info("  current    : %s", current_path)
    logger.info("  historical : %s", historical_dir)
    logger.info("  SAR        : %s", sar_path or "not provided")
    logger.info("  cloud mask : %s", cloud_mask_path or "not provided")
    logger.info("  output     : %s", output_dir)
    logger.info("  sample id  : %s", sample_id)
    logger.info("=" * 60)

    t0 = time.perf_counter()

    # ── Run pipeline ───────────────────────────────────────────────────────
    try:
        result = pipeline.run(
            current_path=current_path,
            historical_dir=historical_dir,
            sar_path=sar_path,
            current_date=acq_date,
            output_dir=output_dir / sample_id / "pipeline_out",
            cloud_mask_path=cloud_mask_path,
            sample_id=sample_id,
        )
    except FileNotFoundError as exc:
        logger.error("File not found: %s", exc)
        return 1
    except Exception as exc:  # noqa: BLE001
        logger.exception("Pipeline failed: %s", exc)
        return 1

    elapsed = time.perf_counter() - t0

    # ── Print summary ──────────────────────────────────────────────────────
    logger.info("")
    logger.info("=" * 60)
    logger.info("✔  Pipeline complete in %.2f s", elapsed)
    logger.info("")
    logger.info("Output files:")

    recon_out = result.reconstruction_output
    if recon_out:
        for k, v in recon_out.to_dict().items():
            if v and k not in ("warnings",):
                logger.info("  %-30s %s", k, v)
        if recon_out.warnings:
            logger.warning("Warnings:")
            for w in recon_out.warnings:
                logger.warning("  ⚠  %s", w)
    else:
        for k, v in result.output_paths.items():
            logger.info("  %-30s %s", k, v)

    logger.info("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
