"""Build the Chroma vector index from local baguwen markdown files."""
import argparse
import os
import sys
from pathlib import Path

# Allow running as `python scripts/build_index.py` from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

load_dotenv()


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Chroma index from markdown files")
    parser.add_argument("--data-dir", default="data/baguwen", help="Directory with *.md files")
    parser.add_argument("--persist-dir", default=None, help="Override CHROMA_PERSIST_DIR")
    parser.add_argument(
        "--no-reset",
        action="store_true",
        default=False,
        help="Append to existing index instead of rebuilding",
    )
    args = parser.parse_args()

    if args.persist_dir:
        os.environ["CHROMA_PERSIST_DIR"] = args.persist_dir

    from src.rag.ingest import load_markdown_files
    from src.rag.store import get_collection

    data_dir = args.data_dir
    if not Path(data_dir).exists():
        print(f"Error: data directory '{data_dir}' does not exist.", file=sys.stderr)
        sys.exit(1)

    print(f"Loading markdown files from '{data_dir}'...")
    chunks = load_markdown_files(data_dir)
    if not chunks:
        print("No chunks found. Check that *.md files exist in the data directory.")
        sys.exit(1)

    print(f"Found {len(chunks)} chunks. Building index...")
    col = get_collection(reset=not args.no_reset)

    ids, docs, metas = [], [], []
    for i, ch in enumerate(chunks):
        chunk_id = f"{Path(ch.metadata['source']).stem}-{i}"
        ids.append(chunk_id)
        docs.append(ch.text)
        metas.append(ch.metadata)

    # Chroma add in batches to avoid memory issues with large corpora
    batch_size = 500
    for start in range(0, len(ids), batch_size):
        col.add(
            ids=ids[start : start + batch_size],
            documents=docs[start : start + batch_size],
            metadatas=metas[start : start + batch_size],
        )
        print(f"  Indexed {min(start + batch_size, len(ids))}/{len(ids)} chunks...")

    persist_dir = os.environ.get("CHROMA_PERSIST_DIR", ".chroma/jobagent")
    print(f"\nDone. {len(ids)} chunks indexed into collection '{col.name}' at '{persist_dir}'.")


if __name__ == "__main__":
    main()
