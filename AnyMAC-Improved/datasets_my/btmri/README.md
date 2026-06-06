# BTMRI

Brain Tumor MRI dataset. Single-file JSON layout.

Download the dataset and place:

- `btmri.json`            — list of records (question / options / answer_idx / image path)
- `image_embeddings/`     — optional pre-computed embeddings (`.pt` files; ignored by git)

## Download

<add download URL / HF dataset ID here>

## Override paths

| Env var | Purpose |
|---|---|
| `BTMRI_JSON` | Path to `btmri.json` if you don't place it under `datasets_my/btmri/` |
| `BTMRI_PATH_PREFIX_OLD` | Absolute prefix that appears in `btmri.json` records (e.g. `/some/cluster/path/Brain Tumor MRI/`). Leave unset if your JSON already points at local files. |
| `BTMRI_PATH_PREFIX_NEW` | Local directory the loader should rewrite the old prefix to (where the images actually live) |
| `MEDSETS_EXTRACT_DIR` | Alternative to `BTMRI_PATH_PREFIX_NEW` — if set, images are looked up under `$MEDSETS_EXTRACT_DIR/btmri_data/` |
