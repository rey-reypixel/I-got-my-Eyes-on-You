import argparse
import sys
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent

def discover_video_source(explicit_source: Optional[str] = None) -> Optional[str]:
    """Locate a video from CLI input or the repository's raw data directories."""
    if explicit_source:
        candidate = Path(explicit_source)
        if candidate.is_file():
            return str(candidate)
        if not candidate.is_absolute():
            candidate = ROOT / candidate
        if candidate.is_file():
            return str(candidate)
        return explicit_source  # Might be a webcam index or RTSP string

    # Search recursively for a video in the workspace
    for pattern in ("*.mp4", "*.avi", "*.mkv", "*.mov", "*.mpg"):
        for candidate in sorted(ROOT.rglob(pattern)):
            if candidate.is_file() and not ".venv" in candidate.parts:
                return str(candidate)

    return None

def main():
    parser = argparse.ArgumentParser(description="I-got-my-Eyes-on-You Live Surveillance Pipeline")
    parser.add_argument("--source", type=str, default=None, help="Video file path, RTSP stream URL, or webcam index (e.g. 0)")
    parser.add_argument("--config", type=str, default="config.json", help="Path to config.json file")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="Path to YOLO weight file")
    parser.add_argument("--pose-model", type=str, default="yolov8n-pose.pt", help="Path to YOLO pose weight file (only loaded when a gated limb-motion check triggers)")
    parser.add_argument("--device", type=str, default="cpu", help="Computation device (e.g. cpu, cuda)")
    parser.add_argument("--max-frames", type=int, default=None, help="Maximum number of frames to process")
    parser.add_argument("--no-show", action="store_true", help="Disable OpenCV rendering window")
    parser.add_argument("--event-log", type=str, default=None, help="Path to a SQLite DB for persisting alert/state-change history for this run (enables event logging when set; query it with mcp_server/server.py)")
    args = parser.parse_args()

    # Import detection engine
    try:
        from src.detection import MultiModelDetectionEngine
    except ImportError as e:
        sys.exit(f"Failed to import core detection engine: {e}")

    # Resolve source
    source = args.source
    # If it's a numeric string, treat it as a camera index
    if source is not None and source.isdigit():
        source = int(source)
    else:
        resolved_source = discover_video_source(source)
        if resolved_source:
            source = resolved_source
        elif source is None:
            sys.exit("Error: No video source was provided, and no sample video could be auto-discovered.")

    print("====================================================")
    print("Starting I-got-my-Eyes-on-You Surveillance Engine")
    print(f"   Source: {source}")
    print(f"   Config: {args.config}")
    print(f"   Model:  {args.model}")
    print(f"   Device: {args.device}")
    if args.event_log:
        print(f"   Event log: {args.event_log}")
    print("====================================================")

    engine = MultiModelDetectionEngine(
        source=source,
        model_path=args.model,
        pose_model_path=args.pose_model,
        device=args.device,
        config_path=args.config,
        max_frames=args.max_frames,
        show_window=not args.no_show,
        event_log_path=args.event_log,
    )

    try:
        engine.run()
        print("Surveillance run completed successfully.")
    except Exception as e:
        sys.exit(f"Surveillance execution failed: {e}")

if __name__ == "__main__":
    main()
