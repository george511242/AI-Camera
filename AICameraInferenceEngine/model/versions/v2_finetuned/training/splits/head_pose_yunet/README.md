# v2 Head Pose and YuNet fixed splits

Generated with seed `20260905` by `tools/prepare_v2_pose_yunet_splits.py`.
Dataset files remain outside the repository under
`/home/g2004/datasets/vac-distance/source/`.

## Head Pose

- 300W-LP variants are grouped by original source before splitting.
- Train: 104,172 images from 3,261 source groups.
- Validation: 18,278 images from 576 source groups.
- Frozen external test: 2,000 AFLW2000 images.
- No source group crosses train and validation; AFLW2000 is not training data.

## YuNet

- Standard WIDER FACE train/validation partitions are retained.
- Landmark annotations are the official InsightFace SCRFD `labelv2` files.
- Train: 12,876 images, 159,393 faces; 75,925 have complete five landmarks.
- Validation: 3,226 images and 39,697 faces with complete five landmarks.
- Four empty WIDER train headers are excluded, matching the official parser.
- Frozen external test: 130 SCFace frontal sources.

## Integrity and licensing

Archive extraction completed without CRC errors. Every 300W-LP manifest image
has a matching `.mat` with finite `Pose_Para`; every WIDER record references an
existing image. Counts and hashes are in `metadata.json` and
`dataset_provenance.json`.

The Kaggle mirrors report license `unknown`. Review licensing before commercial
use or redistribution. Do not commit archives or dataset images.
