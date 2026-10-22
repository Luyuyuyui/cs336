
import time
import pickle
import argparse
from pathlib import Path

from cs336_basics.bpe import train_bpe


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input", required=True)
    parser.add_argument("--vocab-size", type=int, required=True)
    parser.add_argument("--output", required=True)

    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Starting BPE training...")

    start = time.perf_counter()

    vocab, merges = train_bpe(
        input_path=args.input,
        vocab_size=args.vocab_size,
        special_tokens=["<|endoftext|>"],
    )

    elapsed = time.perf_counter() - start

    with open(output_dir / "vocab.pkl", "wb") as f:
        pickle.dump(vocab, f)

    with open(output_dir / "merges.pkl", "wb") as f:
        pickle.dump(merges, f)

    print(f"Training time: {elapsed:.2f}s")
    print(f"Vocabulary size: {len(vocab)}")
    print(f"Number of merges: {len(merges)}")

    # 找出最长的普通 token
    special = {b"<|endoftext|>"}

    normal_tokens = [
        (idx, token)
        for idx, token in vocab.items()
        if token not in special
    ]

    idx, longest = max(
        normal_tokens,
        key=lambda item: len(item[1]),
    )

    print(f"Longest token ID: {idx}")
    print(f"Longest token length: {len(longest)} bytes")
    print(
        "Longest token:",
        longest.decode("utf-8", errors="replace")
    )


if __name__ == "__main__":
    main()
