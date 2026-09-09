YUNET_MODEL_WEIGHT_PATHS = {
    "rk3566": "/model/versions/v1_baseline/rknn/rk3566/yunet_n_640_640_fp16.rknn",
    "rk3588": "/model/versions/v1_baseline/rknn/rk3588/yunet_n_640_640_fp16.rknn",
}


def resolve_yunet_model_path(platform: str = "rk3566") -> str:
    normalized_platform = platform.strip().lower()
    try:
        return YUNET_MODEL_WEIGHT_PATHS[normalized_platform]
    except KeyError as error:
        supported = ", ".join(sorted(YUNET_MODEL_WEIGHT_PATHS))
        raise ValueError(
            f"Unsupported RKNPU platform '{platform}'. Supported platforms: {supported}"
        ) from error
