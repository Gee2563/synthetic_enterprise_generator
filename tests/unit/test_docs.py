from pathlib import Path


def test_docs_exist_and_cover_required_topics() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    readme = (repo_root / "README.md").read_text(encoding="utf-8")

    expected_docs = [
        "docs/README.md",
        "docs/architecture.md",
        "docs/data-model.md",
        "docs/grounding-and-noise.md",
        "docs/operations.md",
        "docs/examples.md",
        "docs/limitations.md",
    ]

    for relative_path in expected_docs:
        assert (repo_root / relative_path).exists(), relative_path

    expected_readme_phrases = [
        "Architecture Overview",
        "Data Model Overview",
        "How Relevance Is Grounded",
        "How Noise Is Generated",
        "Run A Small Dataset",
        "Scale To Large Datasets",
        "Evaluate Output Quality",
        "Limitations And Ethics",
    ]

    for phrase in expected_readme_phrases:
        assert phrase in readme
