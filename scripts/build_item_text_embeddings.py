"""
Build item text embeddings using sentence-transformers.

Text rule: title + " [SEP] " + description
Fallback: parent_asin (when both title and description are empty)

Full-run requires explicit --run_full flag to prevent accidental large jobs.
"""

import argparse
import json
import os
import sys
from datetime import datetime

import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description="Build item text embeddings (M5)")
    p.add_argument("--dataset_name", default="McAuley-Lab/Amazon-Reviews-2023")
    p.add_argument("--config_name", default="raw_meta_Movies_and_TV")
    p.add_argument("--model_name", default="sentence-transformers/all-MiniLM-L6-v2")
    p.add_argument(
        "--item2id_path",
        default="data/processed/movies_tv_5core/item2id.json",
    )
    p.add_argument(
        "--output_dir",
        default="outputs/item_text_embeddings/movies_tv_5core",
    )
    p.add_argument("--cache_dir", default="/workspace/.hf_home/datasets")
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument(
        "--limit_items",
        type=int,
        default=None,
        help="Only encode this many items (smoke test). Omit to use --run_full.",
    )
    p.add_argument(
        "--run_full",
        action="store_true",
        help="Required to encode all items. Without this and without --limit_items, script exits.",
    )
    p.add_argument("--device", default="auto")
    p.add_argument("--trust_remote_code", action="store_true", default=True)
    p.add_argument("--script_version", default="1.0.0")
    return p.parse_args()


def build_text(row: dict) -> tuple[str, bool, str]:
    """Returns (text, is_fallback, text_type).

    text_type: "title_and_desc" | "title_only" | "desc_only" | "fallback"
    """
    title = row.get("title") or ""
    title = str(title).strip()

    desc_raw = row.get("description")
    if desc_raw is None:
        description = ""
    elif isinstance(desc_raw, list):
        description = " ".join(str(x) for x in desc_raw).strip()
    else:
        description = str(desc_raw).strip()

    has_title = bool(title)
    has_desc = bool(description)

    if has_title and has_desc:
        return f"{title} [SEP] {description}", False, "title_and_desc"
    if has_title:
        return f"{title} [SEP] {description}", False, "title_only"
    if has_desc:
        return f"{title} [SEP] {description}", False, "desc_only"

    parent_asin = str(row.get("parent_asin", "")).strip()
    return parent_asin, True, "fallback"


def main():
    args = parse_args()

    # Guard: require explicit intent for full run
    if args.limit_items is None and not args.run_full:
        print(
            "[ERROR] Neither --limit_items nor --run_full was specified.\n"
            "        Pass --limit_items N for a smoke test, or --run_full to encode all items.",
            file=sys.stderr,
        )
        sys.exit(1)

    # --- Load item2id ---
    print(f"[1/5] Loading item2id from {args.item2id_path}")
    with open(args.item2id_path) as f:
        item2id: dict[str, int] = json.load(f)

    num_items_total = len(item2id)
    # Build reverse map for alignment
    asin_set = set(item2id.keys())
    print(f"      {num_items_total} items in item2id")

    # --- Load HuggingFace metadata ---
    print(f"[2/5] Loading {args.config_name} from {args.dataset_name}")
    from datasets import load_dataset

    ds = load_dataset(
        args.dataset_name,
        args.config_name,
        split="full",
        cache_dir=args.cache_dir,
        trust_remote_code=args.trust_remote_code,
    )
    print(f"      metadata rows: {len(ds)}")

    # --- Build asin -> text mapping from metadata ---
    print("[3/5] Building text for each item_idx")

    # index metadata by parent_asin (keep first occurrence)
    asin_to_row: dict[str, dict] = {}
    for row in ds:
        asin = row.get("parent_asin")
        if asin and asin not in asin_to_row:
            asin_to_row[asin] = row

    # Prepare ordered list by item_idx
    idx_to_asin = {v: k for k, v in item2id.items()}

    limit = args.limit_items if args.limit_items is not None else num_items_total
    encode_indices = list(range(min(limit, num_items_total)))

    texts: list[str] = []
    has_text_mask = np.zeros(num_items_total, dtype=bool)
    preview_info: list[dict] = []
    num_missing_metadata = 0
    num_empty_text_fallback = 0
    num_title_and_desc = 0
    num_title_only = 0
    num_desc_only = 0

    for item_idx in encode_indices:
        asin = idx_to_asin[item_idx]
        if asin in asin_to_row:
            text, is_fallback, text_type = build_text(asin_to_row[asin])
            if is_fallback:
                num_empty_text_fallback += 1
            elif text_type == "title_and_desc":
                num_title_and_desc += 1
            elif text_type == "title_only":
                num_title_only += 1
            elif text_type == "desc_only":
                num_desc_only += 1
        else:
            text = asin
            is_fallback = True
            text_type = "fallback"
            num_missing_metadata += 1
            num_empty_text_fallback += 1

        texts.append(text)
        has_text_mask[item_idx] = not is_fallback
        if item_idx < 3:
            meta_row = asin_to_row.get(asin, {})
            preview_info.append(
                {
                    "item_idx": item_idx,
                    "parent_asin": asin,
                    "title": meta_row.get("title", ""),
                    "text_preview": text[:120],
                    "is_fallback": is_fallback,
                    "text_type": text_type,
                }
            )

    num_items_encoded = len(encode_indices)
    print(f"      items to encode   : {num_items_encoded}")
    print(f"      title + desc      : {num_title_and_desc}")
    print(f"      title only        : {num_title_only}")
    print(f"      desc only         : {num_desc_only}")
    print(f"      missing metadata  : {num_missing_metadata}")
    print(f"      empty text fallback: {num_empty_text_fallback}")

    # --- Encode ---
    print(f"[4/5] Encoding with {args.model_name}")
    from sentence_transformers import SentenceTransformer

    device = args.device
    if device == "auto":
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"      device: {device}")

    model = SentenceTransformer(args.model_name, device=device)
    embedding_dim = model.get_embedding_dimension()

    embeddings = model.encode(
        texts,
        batch_size=args.batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=False,
    )
    embeddings = embeddings.astype(np.float32)
    print(f"      encoded shape: {embeddings.shape}")

    # --- Build full-size array aligned to item_idx ---
    # For smoke test (limit < total), only encode_indices rows are filled;
    # remaining rows are left as zeros. The npy always has num_items_total rows
    # so that item_idx == row index invariant holds.
    if args.run_full and args.limit_items is None:
        # Full run: shape is exactly [num_items_total, embedding_dim]
        output_array = embeddings
    else:
        # Smoke test: build full-size array, fill only encoded rows
        output_array = np.zeros((num_items_total, embedding_dim), dtype=np.float32)
        for local_i, item_idx in enumerate(encode_indices):
            output_array[item_idx] = embeddings[local_i]

    print(f"      output array shape: {output_array.shape}")

    # --- Save ---
    print(f"[5/5] Saving to {args.output_dir}")
    os.makedirs(args.output_dir, exist_ok=True)

    npy_path = os.path.join(args.output_dir, "item_text_embedding.npy")
    has_text_path = os.path.join(args.output_dir, "item_has_text.npy")
    meta_path = os.path.join(args.output_dir, "item_text_meta.json")

    np.save(npy_path, output_array)
    np.save(has_text_path, has_text_mask)

    meta = {
        "dataset_name": args.dataset_name,
        "config_name": args.config_name,
        "model_name": args.model_name,
        "embedding_dim": embedding_dim,
        "num_items_total": num_items_total,
        "num_items_encoded": num_items_encoded,
        "num_items_missing_metadata": num_missing_metadata,
        "num_empty_text_fallback": num_empty_text_fallback,
        "num_title_and_desc": num_title_and_desc,
        "num_title_only": num_title_only,
        "num_desc_only": num_desc_only,
        "item2id_path": args.item2id_path,
        "text_rule": "title + ' [SEP] ' + description; fallback=parent_asin",
        "description_rule": "join list with space; None -> empty string",
        "is_smoke_test": args.limit_items is not None,
        "limit_items": args.limit_items,
        "created_at": datetime.utcnow().isoformat() + "Z",
        "script_version": args.script_version,
        "output_npy": npy_path,
        "output_has_text_npy": has_text_path,
        "dtype": "float32",
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"      npy:  {npy_path}  shape={output_array.shape}  dtype={output_array.dtype}")
    print(f"      has_text: {has_text_path}  true={int(has_text_mask.sum())}/{len(has_text_mask)}")
    print(f"      meta: {meta_path}")

    # --- Preview ---
    print("\n--- Preview (first 3 items) ---")
    for p in preview_info:
        print(
            f"  item_idx={p['item_idx']}  asin={p['parent_asin']}  type={p['text_type']}\n"
            f"    title  : {str(p['title'])[:80]}\n"
            f"    text   : {p['text_preview']}\n"
            f"    fallback: {p['is_fallback']}"
        )

    print("\n[DONE]")
    print(f"  shape             : {output_array.shape}")
    print(f"  dtype             : {output_array.dtype}")
    print(f"  embedding_dim     : {embedding_dim}")
    print(f"  num_items_total   : {num_items_total}")
    print(f"  num_items_encoded : {num_items_encoded}")
    print(f"  title_and_desc    : {num_title_and_desc}")
    print(f"  title_only        : {num_title_only}")
    print(f"  desc_only         : {num_desc_only}")
    print(f"  missing_metadata  : {num_missing_metadata}")
    print(f"  empty_fallback    : {num_empty_text_fallback}")
    print(f"  is_smoke_test     : {args.limit_items is not None}")


if __name__ == "__main__":
    main()
