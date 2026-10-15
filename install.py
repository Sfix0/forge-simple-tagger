import launch


def requirements_met():
    try:
        import onnxruntime  # noqa: F401
        import huggingface_hub  # noqa: F401
        return True
    except ImportError:
        return False


if not requirements_met():
    # Deliberately NOT installing tensorflow / jax — they break Forge
    # (protobuf/transformers conflicts). Only onnxruntime (CPU), which is
    # already in the venv and safe, plus huggingface_hub for downloads.
    launch.run_pip("install onnxruntime huggingface_hub", "simple-tagger")
