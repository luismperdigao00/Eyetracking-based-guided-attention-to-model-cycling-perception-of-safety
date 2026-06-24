"""Comparison-table loading, filtering, splitting, and split summaries."""
from __future__ import annotations

import math
import os
import pickle

import pandas as pd
from sklearn.model_selection import train_test_split

from egpcs.training.metrics import compute_class_weights_from_df
from egpcs.utils.filesystem import ensure_directory


def read_data(args):
    # ------------------------------------------------------------------
    # 0) LOAD
    # ------------------------------------------------------------------
    try:
        comparisons_df = pickle.load(open(args.comparisons, "rb"))
    except Exception:
        comparisons_df = pd.read_pickle(args.comparisons)

    print(f"[read_data] Loaded raw rows: {len(comparisons_df):,}")
    print(f"[read_data] Raw columns: {list(comparisons_df.columns)}")

    # ------------------------------------------------------------------
    # 1) KEEP ONLY COLUMNS THAT EXIST
    # ------------------------------------------------------------------
    cols_we_need = [
        "score",
        "image_l", "image_r",
        "dataset",
        "has_eyetracker",
        "npy_file_l", "npy_file_r",
        "survey_id", "trial_id",
    ]
    existing_cols = [c for c in cols_we_need if c in comparisons_df.columns]
    comparisons_df = comparisons_df[existing_cols].copy()

    print(f"[read_data] Columns kept: {existing_cols}")
    print(f"[read_data] Rows after column selection: {len(comparisons_df):,}")

    # ------------------------------------------------------------------
    # 2) CITY / DATASET FILTER
    # ------------------------------------------------------------------
    if "dataset" in comparisons_df.columns:
        print("\n[read_data] Available datasets:")
        print(comparisons_df["dataset"].value_counts())

        cities_arg = getattr(args, "cities", "all")
        if cities_arg.lower() != "all":
            selected_cities = [c.strip() for c in cities_arg.split(",") if c.strip()]
            print(f"\n[read_data] Filtering to cities: {selected_cities}")

            before = len(comparisons_df)
            comparisons_df = comparisons_df[
                comparisons_df["dataset"].isin(selected_cities)
            ].copy()
            after = len(comparisons_df)

            print(f"[read_data] Rows after city filter: {after}/{before}")
            if after == 0:
                print("[WARN] City filter removed all rows.")

    # ------------------------------------------------------------------
    # 3) IMAGE FILENAME NORMALIZATION
    # ------------------------------------------------------------------
    for side in ("image_l", "image_r"):
        if side in comparisons_df.columns:
            comparisons_df[side] = (
                comparisons_df[side]
                .astype(str)
                .apply(lambda x: x if x.lower().endswith(".jpg") else f"{x}.jpg")
            )

    # ------------------------------------------------------------------
    # 4) GAZE FLAG NORMALIZATION + FILTERING
    # ------------------------------------------------------------------
    if "has_eyetracker" in comparisons_df.columns:
        comparisons_df["has_eyetracker"] = (
            comparisons_df["has_eyetracker"]
            .replace({"True": True, "False": False, "true": True, "false": False})
            .fillna(False)
            .astype(bool)
        )

        print(
            "\n[read_data] has_eyetracker distribution:",
            comparisons_df["has_eyetracker"].value_counts(dropna=False).to_dict(),
        )

        if str(getattr(args, "eyetracker_filter", "all")).lower().strip() == "only":
            before = len(comparisons_df)
            comparisons_df = comparisons_df[comparisons_df["has_eyetracker"]].copy()
            after = len(comparisons_df)

            print(f"[read_data] Rows after eyetracker_filter=only: {after}/{before}")
            if after == 0:
                print("[WARN] --eyetracker_filter only removed all rows.")

    # ------------------------------------------------------------------
    # 5) LABEL HANDLING / TIES
    # ------------------------------------------------------------------
    if "score" not in comparisons_df.columns:
        raise ValueError("[read_data] Missing required column: 'score'")

    if not args.ties:
        before = len(comparisons_df)
        comparisons_df = comparisons_df[comparisons_df["score"] != 0].copy()
        after = len(comparisons_df)

        print(f"\n[read_data] Rows after ties=False filter: {after}/{before}")
        if after == 0:
            print("[WARN] ties=False removed all rows.")

        comparisons_df["score_classification"] = comparisons_df["score"].replace(
            {-1: 0, +1: 1}
        )
    else:
        comparisons_df["score_classification"] = comparisons_df["score"] + 1

    # ------------------------------------------------------------------
    # 6) FINAL SANITY CHECK
    # ------------------------------------------------------------------
    print("\n[read_data] FINAL ROW COUNT:", len(comparisons_df))
    if len(comparisons_df) > 0:
        print(
            "[read_data] Final score distribution:",
            comparisons_df["score"].value_counts(dropna=False).to_dict(),
        )
    else:
        print("[FATAL] Dataset is EMPTY after all filters.")

    return comparisons_df


def _boolish_series_to_bool_mask(s: pd.Series) -> pd.Series:
    return s.astype(str).str.lower().str.strip().isin(["1", "true", "t", "yes", "y"])


def _load_or_split(
    df: pd.DataFrame,
    seed: int,
    comparisons_path: str,
    splits_dir: str = "splits",
    train_pct: float = 0.67,
    val_pct: float = 0.13,
    test_pct: float = 0.20,
    load_if_exists: bool = True,
    save_splits: bool = True,
    train_gaze_frac: float | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Standard random split + optional post-split gaze rebalancing.

    train_gaze_frac:
      - None: no gaze rebalancing (default behavior)
      - float in [0, 1]: target fraction of all has_eyetracker=True rows that should be in TRAIN.
        Typical range: [train_pct, 1.0]. 1.0 forces all gaze rows into train (when possible).
    """

    ensure_directory(splits_dir)

    total = float(train_pct + val_pct + test_pct)
    if abs(total - 1.0) > 1e-9:
        raise ValueError(f"train_pct + val_pct + test_pct must sum to 1.0, got {total}")

    if test_pct <= 0.0 or test_pct >= 1.0:
        raise ValueError("test_pct must be in (0, 1)")

    if val_pct < 0.0 or train_pct < 0.0:
        raise ValueError("train_pct and val_pct must be >= 0")

    # ----------------------------
    # Resolve split cache filenames
    # ----------------------------
    split_prefix = os.path.splitext(os.path.basename(comparisons_path))[0]

    gaze_tag = ""
    if train_gaze_frac is not None:
        g = float(train_gaze_frac)
        g = max(0.0, min(1.0, g))
        gaze_tag = f"_trainGaze{int(round(100.0 * g)):03d}"

    train_path = os.path.join(splits_dir, f"{split_prefix}{gaze_tag}_train.pkl")
    val_path = os.path.join(splits_dir, f"{split_prefix}{gaze_tag}_val.pkl")
    test_path = os.path.join(splits_dir, f"{split_prefix}{gaze_tag}_test.pkl")

    if load_if_exists and all(os.path.exists(p) for p in (train_path, val_path, test_path)):
        return (
            pd.read_pickle(train_path),
            pd.read_pickle(val_path),
            pd.read_pickle(test_path),
        )

    # ----------------------------
    # 1) Baseline random split
    # ----------------------------
    X_train, X_test = train_test_split(
        df,
        test_size=test_pct,
        shuffle=True,
        random_state=int(seed),
    )

    remaining = 1.0 - test_pct
    if remaining <= 0.0:
        raise ValueError("test_pct leaves no data for train/val")

    val_size_of_train = val_pct / remaining

    X_train, X_val = train_test_split(
        X_train,
        test_size=val_size_of_train,
        shuffle=True,
        random_state=int(seed),
    )

    # ----------------------------
    # 2) Optional gaze rebalancing
    # ----------------------------
    if train_gaze_frac is not None:
        if "has_eyetracker" in X_train.columns and "has_eyetracker" in X_val.columns and "has_eyetracker" in X_test.columns:
            frac = float(train_gaze_frac)
            frac = max(0.0, min(1.0, frac))

            X_train = X_train.copy()
            X_val = X_val.copy()
            X_test = X_test.copy()

            tr_g = _boolish_series_to_bool_mask(X_train["has_eyetracker"])
            va_g = _boolish_series_to_bool_mask(X_val["has_eyetracker"])
            te_g = _boolish_series_to_bool_mask(X_test["has_eyetracker"])

            total_gaze = int(tr_g.sum() + va_g.sum() + te_g.sum())
            if total_gaze > 0:
                desired_train_gaze = int(math.ceil(frac * float(total_gaze)))
                current_train_gaze = int(tr_g.sum())
                need = desired_train_gaze - current_train_gaze

                if need > 0:
                    # Candidate gaze rows to pull into train
                    val_gaze_idx = X_val.index[va_g].to_list()
                    test_gaze_idx = X_test.index[te_g].to_list()

                    if len(val_gaze_idx) > 0:
                        val_gaze_idx = X_val.loc[val_gaze_idx].sample(frac=1.0, random_state=int(seed) + 101).index.to_list()
                    if len(test_gaze_idx) > 0:
                        test_gaze_idx = X_test.loc[test_gaze_idx].sample(frac=1.0, random_state=int(seed) + 202).index.to_list()

                    def _swap_from_source(
                        X_src: pd.DataFrame,
                        src_gaze_idx: list,
                        n_take: int,
                        X_train_local: pd.DataFrame,
                        rng_seed: int,
                    ) -> tuple[pd.DataFrame, pd.DataFrame, int]:
                        if n_take <= 0 or len(src_gaze_idx) == 0:
                            return X_train_local, X_src, 0

                        take_idx = src_gaze_idx[:n_take]
                        take_rows = X_src.loc[take_idx]
                        X_src = X_src.drop(index=take_idx)

                        tr_mask = _boolish_series_to_bool_mask(X_train_local["has_eyetracker"])
                        train_no_gaze_idx = X_train_local.index[~tr_mask].to_list()

                        if len(train_no_gaze_idx) >= n_take:
                            swap_idx = (
                                X_train_local.loc[train_no_gaze_idx]
                                .sample(n=n_take, random_state=int(rng_seed))
                                .index.to_list()
                            )
                            swap_rows = X_train_local.loc[swap_idx]
                            X_train_local = X_train_local.drop(index=swap_idx)
                            X_src = pd.concat([X_src, swap_rows], axis=0)

                        X_train_local = pd.concat([X_train_local, take_rows], axis=0)
                        return X_train_local, X_src, len(take_idx)

                    # Pull from val first, then test
                    take_val = min(need, len(val_gaze_idx))
                    X_train, X_val, got_val = _swap_from_source(X_val, val_gaze_idx, take_val, X_train, int(seed) + 303)
                    need -= got_val

                    if need > 0:
                        take_test = min(need, len(test_gaze_idx))
                        X_train, X_test, got_test = _swap_from_source(X_test, test_gaze_idx, take_test, X_train, int(seed) + 404)
                        need -= got_test

    # ----------------------------
    # 3) Save splits
    # ----------------------------
    if save_splits:
        X_train.to_pickle(train_path)
        X_val.to_pickle(val_path)
        X_test.to_pickle(test_path)

    return X_train, X_val, X_test


def _print_filtered_dataset_summary(args, df: pd.DataFrame) -> None:
    print("\n=== Effective Dataset (after all filters, before split) ===")
    print(f"Comparisons file : {args.comparisons}")
    print(f"Cities requested : {args.cities}")
    print(f"Model variant        : {getattr(args, 'model_variant', 'Baseline')}  (rows kept depend on has_eyetracker + eyetracker_filter)")
    print(f"Ties enabled     : {args.ties}  (ties=False removes score==0 rows)")
    print(f"Final row count  : {len(df):,}")

    if "score" in df.columns:
        print("\nScore distribution (post-filtering):")
        score_counts = df["score"].value_counts().sort_index()
        total_rows = len(df)
        for s, c in score_counts.items():
            print(f"  score={int(s):>2}: {int(c):>6,} ({(100.0 * float(c) / max(1, total_rows)):5.2f}%)")

    if "has_eyetracker" in df.columns:
        print("\nEyetracker availability (post-filtering):")
        et_counts = df["has_eyetracker"].value_counts(dropna=False)
        for k, v in et_counts.items():
            print(f"  {str(k):>5}: {int(v):>6,} ({(100.0 * float(v) / max(1, len(df))):5.2f}%)")

    print("\nExample rows (post-filtering):")
    print(df.head(3))
    print("========================================================\n")


def _print_image_overlap_stats(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    left_col: str = "image_l",
    right_col: str = "image_r",
    dataset_col: str | None = "dataset",
) -> None:
    def _unique_images(df: pd.DataFrame) -> set[str]:
        if len(df) == 0:
            return set()

        cols = [c for c in [left_col, right_col, dataset_col] if c is not None and c in df.columns]
        df2 = df[cols].copy()
        df2[left_col] = df2[left_col].astype(str)
        df2[right_col] = df2[right_col].astype(str)

        if dataset_col is not None and dataset_col in df2.columns:
            ds = df2[dataset_col].astype(str)
            left = ds + "/" + df2[left_col]
            right = ds + "/" + df2[right_col]
        else:
            left = df2[left_col]
            right = df2[right_col]

        return set(pd.concat([left, right], ignore_index=True).tolist())

    def _pct(n: int, d: int) -> float:
        return 0.0 if d == 0 else (100.0 * n / d)

    tr = _unique_images(X_train)
    va = _unique_images(X_val)
    te = _unique_images(X_test)

    tr_va = tr & va
    tr_te = tr & te
    va_te = va & te

    print(f"Images : train={len(tr)} | val={len(va)} | test={len(te)}")

    print(f"train∩val : {len(tr_va)} | % of train={_pct(len(tr_va), len(tr)):.2f}% | % of val={_pct(len(tr_va), len(va)):.2f}%")
    print(f"train∩test: {len(tr_te)} | % of train={_pct(len(tr_te), len(tr)):.2f}% | % of test={_pct(len(tr_te), len(te)):.2f}%")
    print(f"val∩test  : {len(va_te)} | % of val={_pct(len(va_te), len(va)):.2f}% | % of test={_pct(len(va_te), len(te)):.2f}%")


def _print_has_eyetracker_by_split(X_train: pd.DataFrame, X_val: pd.DataFrame, X_test: pd.DataFrame) -> None:
    def _counts(df_: pd.DataFrame) -> tuple[int, int, float]:
        s = _boolish_series_to_bool_mask(df_["has_eyetracker"])
        n_true = int(s.sum())
        n_total = int(len(df_))
        return n_true, n_total, (n_true / max(1, n_total))

    if "has_eyetracker" not in X_train.columns:
        print("\nColumn 'has_eyetracker' not found in comparisons_df.")
        return

    tr_n, tr_tot, tr_rate = _counts(X_train)
    va_n, va_tot, va_rate = _counts(X_val)
    te_n, te_tot, te_rate = _counts(X_test)

    print("\nhas_eyetracker per split:")
    print(f"  Train: {tr_n}/{tr_tot} = {tr_rate:.2%}")
    print(f"  Val  : {va_n}/{va_tot} = {va_rate:.2%}")
    print(f"  Test : {te_n}/{te_tot} = {te_rate:.2%}")


def _print_split_sizes(df: pd.DataFrame, X_train: pd.DataFrame, X_val: pd.DataFrame, X_test: pd.DataFrame) -> None:
    total = max(1, len(df))
    print("=== Splits (on the filtered dataset above) ===")
    print(f"- Train: {len(X_train):,}  [{len(X_train)/total:.2%}]")
    print(f"- Val  : {len(X_val):,}  [{len(X_val)/total:.2%}]")
    print(f"- Test : {len(X_test):,}  [{len(X_test)/total:.2%}]")
    print("========================================================\n")


def _print_label_distribution_by_split(X_train: pd.DataFrame, X_val: pd.DataFrame, X_test: pd.DataFrame) -> None:
    print("=== Label distribution per split (score, after filtering) ===")
    for part_name, df_part in [("Train", X_train), ("Val", X_val), ("Test", X_test)]:
        counts = df_part["score"].value_counts().sort_index()
        total_part = max(1, len(df_part))
        print(f"- {part_name}: {len(df_part):,} samples")
        for cls_val, cls_count in counts.items():
            pct = 100.0 * float(cls_count) / float(total_part)
            print(f"    score={int(cls_val):>2d}: {int(cls_count):>6,} ({pct:5.2f}%)")
    print("============================================================")


def _compute_and_attach_class_weights(args, X_train: pd.DataFrame) -> None:
    args.class_weights = compute_class_weights_from_df(
        X_train["score_classification"],
        use_ties=args.ties,
        enable_weights=args.use_class_weights,
    )

    if args.use_class_weights and args.class_weights is not None:
        cw = args.class_weights.detach().cpu().numpy().tolist()
        print(f"Class weights: ON  (computed from Train split) → {cw}")
    else:
        print("Class weights: OFF")
    print()
