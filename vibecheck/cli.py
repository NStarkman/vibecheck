"""Command line: `vibecheck <spotify link | tracks.csv>` or `vibecheck --demo`."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="vibecheck", description="Judge a Spotify playlist.")
    ap.add_argument("source", nargs="?", help="playlist/album/track link, 'liked', 'top', 'recent', or a CSV of tracks")
    ap.add_argument("--history", help="Spotify data export (zip, folder or JSON) for listening-behaviour analysis")
    ap.add_argument("--tz", default=None, help="your time zone for listening times, e.g. America/New_York (default: this computer's offset)")
    ap.add_argument("--out", help="folder to save charts + report.md")
    ap.add_argument("--demo", action="store_true", help="run on built-in synthetic data, no network")
    args = ap.parse_args(argv)

    import matplotlib

    matplotlib.use("Agg")
    from . import Vibecheck, load_tracks_csv
    from .report import save_all

    if args.demo:
        from .demo import demo_history, demo_playlist

        vc = Vibecheck(offline=True)
        df = demo_playlist()
        prof = vc.analyze(df, meta={"kind": "playlist", "name": "Demo playlist"}, history=demo_history(df), tz=args.tz or "America/New_York")
    elif args.source and args.source.lower().endswith(".csv") and Path(args.source).exists():
        vc = Vibecheck()
        prof = vc.analyze(load_tracks_csv(args.source), meta={"kind": "csv", "name": Path(args.source).stem},
                          history=args.history, tz=args.tz or "local")
    elif args.source:
        vc = Vibecheck()
        prof = vc.analyze(args.source, history=args.history, tz=args.tz or "local")
    elif args.history:
        vc = Vibecheck()
        prof = vc.analyze_history(args.history, tz=args.tz or "local")
    else:
        ap.print_help()
        return 2

    print()
    print(prof.to_markdown())
    if args.out:
        paths = save_all(prof, args.out)
        print(f"\nSaved {len(paths)} charts and report.md to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
