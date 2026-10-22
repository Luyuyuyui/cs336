"""Analyze saved CS336 BPE artifacts without training again."""

import argparse
import csv
import pickle
from collections import Counter
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("outputs/tinystories_10k"))
    parser.add_argument("--output", type=Path, help="Defaults to the input directory")
    parser.add_argument(
        "--special-token", action="append", default=None,
        help="Repeat for each special token; replaces the default <|endoftext|>",
    )
    args = parser.parse_args()
    output = args.output or args.input
    # Load only your own trusted training artifacts: pickle can execute code.
    with (args.input / "vocab.pkl").open("rb") as f:
        vocab = pickle.load(f)
    with (args.input / "merges.pkl").open("rb") as f:
        merges = pickle.load(f)
    if not isinstance(vocab, dict) or not all(
        type(i) is int and isinstance(token, bytes)
        for i, token in vocab.items()
    ):
        raise ValueError("Expected vocab.pkl to contain dict[int, bytes]")
    if not isinstance(merges, (list, tuple)) or not all(
        isinstance(pair, (list, tuple)) and len(pair) == 2
        and all(isinstance(part, bytes) for part in pair)
        for pair in merges
    ):
        raise ValueError("Expected merges.pkl to contain a sequence of byte pairs")

    names = args.special_token if args.special_token is not None else ["<|endoftext|>"]
    special_bytes = {name.encode("utf-8") for name in names}
    special = {i: token for i, token in vocab.items() if token in special_bytes}
    ordinary = {i: token for i, token in vocab.items() if token not in special_bytes}
    missing = special_bytes - set(special.values())
    lengths = Counter(len(token) for token in ordinary.values())
    total = len(ordinary)
    lines = [
        f"Input directory: {args.input.resolve()}",
        f"Vocabulary size: {len(vocab)}",
        f"Number of merges: {len(merges)}",
        f"Special token entries: {len(special)}",
        f"Ordinary token entries: {total}",
        "", "Special tokens (exact byte match):",
    ]
    for i, token in sorted(special.items()):
        lines.append(f"  id={i}, bytes={len(token)}, token={token!r}")
    if not special:
        lines.append("  (none found)")
    for token in sorted(missing):
        lines.append(f"WARNING: configured special token absent: {token!r}")
    lines.extend(["", "Longest 20 ordinary tokens:"])
    for rank, (i, token) in enumerate(
        sorted(ordinary.items(), key=lambda item: (-len(item[1]), item[0]))[:20], 1
    ):
        # repr preserves leading spaces; bytes repr preserves invalid UTF-8.
        display = repr(token.decode("utf-8", errors="backslashreplace"))
        lines.append(f"{rank:2}. id={i}, bytes={len(token)}, text={display}, raw={token!r}")
    lines.extend(["", "Byte-length distribution (ordinary vocabulary entries):",
                  "byte_length,count,percentage"])
    rows = [(n, count, f"{100 * count / total:.6f}")
            for n, count in sorted(lengths.items())]
    for n, count, percentage in rows:
        lines.append(f"{n},{count},{percentage}")
    output.mkdir(parents=True, exist_ok=True)
    report = "\n".join(lines) + "\n"
    (output / "analysis.txt").write_text(report, encoding="utf-8")
    with (output / "token_lengths.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["byte_length", "count", "percentage"])
        writer.writerows(rows)
    print(report, end="")
    print(f"Saved reports to: {output.resolve()}")


if __name__ == "__main__":
    main()
