# YuNet training provenance

- Repository: `https://github.com/ShiqiYu/libfacedetection.train`
- Pinned upstream commit: `dca340aa082c71081a68d17db8e58b33a58a914b`
- Upstream commit date: `2026-06-14T17:48:23+08:00`
- Upstream subject: `Merge pull request #100 from Wwupup/pr-refactor`
- Official checkpoint: `weights/yunet_n.pth`
- Checkpoint SHA256:
  `9c7c9e14a4d60e57491d77e4d8b0d451bd834f300e05c0dea7fd1f6533817b06`
- Checkpoint strict-load result: PASS
- Parameter count: 75,856

The vendored checkout intentionally has a small local diff:

1. packaging-only license metadata compatibility for editable installation;
2. explicit `--init-weights`, separate from resume;
3. deterministic `--seed`;
4. opt-in `--small-face-policy`;
5. CUDA peak allocated/reserved logging.

The policy adds crop choices `2.0` and `2.5` to retain more context and create
smaller effective faces. Camera degradation is fixed: 60% clean-resolution,
25% 0.75x and 15% 0.5x down/upscale; independently 15% mild Gaussian blur,
20% JPEG quality 75-95, 25% brightness/contrast, and 15% mild Gaussian noise.
Default upstream behavior is unchanged unless `--small-face-policy` is passed.

