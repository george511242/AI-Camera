AGE_MODELS = {
    "v2": {
        "implementation": "ssrnet",
        "paths": {
            "rk3566": "/model/versions/v2_finetuned/rknn/rk3566/age_ssrnet_v2_64x64_fp16.rknn",
            "rk3588": "/model/versions/v2_finetuned/rknn/rk3588/age_ssrnet_v2_64x64_fp16.rknn",
        },
    },
    "v4": {
        "implementation": "mobilenetv3",
        "paths": {
            "rk3566": "/model/versions/v4_age_mobilenetv3/rknn/rk3566/age_v4_mobilenetv3_large_112_fp16.rknn",
            "rk3588": "/model/versions/v4_age_mobilenetv3/rknn/rk3588/age_v4_mobilenetv3_large_112_fp16.rknn",
        },
    },
}


def resolve_age_model(version: str = "v2", platform: str = "rk3566") -> dict[str, str]:
    normalized_version = version.strip().lower()
    normalized_platform = platform.strip().lower()
    try:
        model = AGE_MODELS[normalized_version]
    except KeyError as error:
        supported = ", ".join(sorted(AGE_MODELS))
        raise ValueError(
            f"Unsupported Age model version '{version}'. Supported versions: {supported}"
        ) from error
    try:
        path = model["paths"][normalized_platform]
    except KeyError as error:
        supported = ", ".join(sorted(model["paths"]))
        raise ValueError(
            f"Unsupported RKNPU platform '{platform}' for Age model "
            f"'{normalized_version}'. Supported platforms: {supported}"
        ) from error
    return {
        "version": normalized_version,
        "implementation": model["implementation"],
        "path": path,
    }


def resolve_age_model_path(platform: str = "rk3566", version: str = "v2") -> str:
    return resolve_age_model(version=version, platform=platform)["path"]
