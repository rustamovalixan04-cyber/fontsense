from __future__ import annotations

import ast
import json
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = ROOT / "notebooks" / "07_colab_demo.ipynb"


def load_notebook() -> dict:
    return json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))


def notebook_source(notebook: dict) -> str:
    return "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
    )


def test_colab_notebook_has_valid_clean_structure():
    notebook = load_notebook()

    assert notebook["nbformat"] == 4
    assert notebook["metadata"]["kernelspec"]["name"] == "python3"
    assert len(notebook["cells"]) == 18
    for index, cell in enumerate(notebook["cells"], start=1):
        assert cell["cell_type"] in {"markdown", "code"}
        if cell["cell_type"] == "code":
            assert cell.get("execution_count") is None
            assert cell.get("outputs") == []
            ast.parse(
                "".join(cell["source"]),
                filename=f"{NOTEBOOK_PATH}#cell-{index}",
            )


def test_colab_notebook_documents_the_frozen_demo_contract():
    source = notebook_source(load_notebook())

    required_text = (
        "Rustamov Alixan",
        "test macro F1 of 0.8653",
        "test accuracy of 86.67%",
        "broad font categories",
        "display",
        "handwriting",
        "monospace",
        "sans serif",
        "serif",
        "c98cf0d1a02503a02b8f8242fec462ea",
        "2a0c455380238ec54fc4f62fdb13bb2f",
        "predictor.threshold == 0.60",
        "tuple(smoke_tensor.shape) == (1, 48, 112)",
        "len(smoke_probabilities) == 5",
        "build_demo()",
        "share=True",
        "inline=True",
    )
    for expected in required_text:
        assert expected in source


def test_colab_notebook_uses_minimal_cpu_dependencies_and_real_repository():
    source = notebook_source(load_notebook())

    assert (
        "https://github.com/rustamovalixan04-cyber/fontsense.git"
        in source
    )
    assert "torch==2.13.0+cpu" in source
    assert "torchvision==0.28.0+cpu" in source
    assert "https://download.pytorch.org/whl/cpu" in source
    assert "gradio>=4.44" in source
    assert "Pillow>=10.0" in source
    assert "numpy>=1.26,<3" in source
    assert '"--no-deps",' in source
    assert '"-e",' in source
    assert "personal access token" in source


def test_colab_notebook_has_no_local_paths_credentials_or_data_access():
    raw = NOTEBOOK_PATH.read_text(encoding="utf-8")
    source = notebook_source(load_notebook())

    windows_path = re.search(
        r'(?i)(?:^|[\s"\'`(])(?:[a-z]:[\\/]|\\\\[^\\\s]+\\)',
        raw,
    )
    credential = re.search(
        r"(?:ghp_|github_pat_|gho_|ghu_|ghs_)[A-Za-z0-9_]+",
        raw,
    )
    forbidden_runtime_access = (
        "full_manifest.csv",
        "google_fonts_final_family_split.csv",
        "data/processed",
        "data\\processed",
        "read_csv(",
        "train_cnn",
        "evaluate_final_cnn",
        "fontsense.final_evaluation",
    )

    assert windows_path is None
    assert credential is None
    assert not any(term in source for term in forbidden_runtime_access)
