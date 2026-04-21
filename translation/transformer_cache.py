import os


def resolve_local_hf_snapshot(model_name):
    repo_dir_name = model_name.replace("/", "--")
    cache_root = os.path.expanduser(
        os.path.join("~", ".cache", "huggingface", "hub", f"models--{repo_dir_name}")
    )
    snapshots_dir = os.path.join(cache_root, "snapshots")

    if not os.path.isdir(snapshots_dir):
        return model_name

    candidates = []
    main_ref_path = os.path.join(cache_root, "refs", "main")
    if os.path.isfile(main_ref_path):
        with open(main_ref_path, "r", encoding="utf-8") as f:
            revision = f.read().strip()
        if revision:
            candidates.append(os.path.join(snapshots_dir, revision))

    for entry in sorted(os.listdir(snapshots_dir), reverse=True):
        candidate = os.path.join(snapshots_dir, entry)
        if candidate not in candidates:
            candidates.append(candidate)

    for candidate in candidates:
        config_path = os.path.join(candidate, "config.json")
        bin_path = os.path.join(candidate, "pytorch_model.bin")
        safetensors_path = os.path.join(candidate, "model.safetensors")
        if os.path.isfile(config_path) and (
            os.path.isfile(bin_path) or os.path.isfile(safetensors_path)
        ):
            return candidate

    return model_name
