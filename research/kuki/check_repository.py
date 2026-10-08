"""Check the published file inventory, documentation, and internal Python imports."""

import ast
import hashlib
import json
import re
from pathlib import Path
from urllib.parse import unquote, urlsplit

ARTIFACT_MANIFEST = Path("research/configs/published-artifacts.json")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def check_artifacts():
    manifest = json.loads(ARTIFACT_MANIFEST.read_text(encoding="utf-8"))
    require(manifest["algorithm"] == "sha256", "Unsupported artifact hash algorithm")
    require(bool(manifest["files"]), "Empty artifact inventory")
    for name, expected in manifest["files"].items():
        path = Path(name)
        require(not path.is_absolute() and ".." not in path.parts, f"Invalid path: {name}")
        require(path.is_file(), f"Missing published artifact: {name}")
        require(hashlib.sha256(path.read_bytes()).hexdigest() == expected,
                f"Published artifact hash mismatch: {name}")
    return len(manifest["files"])


def check_documents():
    paths = [Path("README.md"), Path("REPRODUCING.md")]
    paths += sorted(Path("paper").rglob("*.md"))
    paths += sorted(Path("research").rglob("*.md"))
    count = 0
    for path in paths:
        text = path.read_text(encoding="utf-8")
        for destination in re.findall(r"\]\(([^)]+)\)", text):
            target = urlsplit(destination)
            if target.scheme or target.netloc or not target.path:
                continue
            require((path.parent / unquote(target.path)).exists(),
                    f"Broken local link in {path}: {destination}")
            count += 1
    return count


def check_python():
    paths = sorted(Path("research").rglob("*.py"))
    imports = 0
    for path in paths:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            elif (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                  and node.func.attr == "import_module" and node.args
                  and isinstance(node.args[0], ast.Constant)):
                names = [node.args[0].value]
            for name in names:
                if isinstance(name, str) and name.startswith("research."):
                    module = Path(name.replace(".", "/"))
                    require(module.with_suffix(".py").is_file() or module.is_dir(),
                            f"Missing internal module in {path}:{node.lineno}: {name}")
                    imports += 1
    return len(paths), imports


def check_configs():
    data = json.loads(Path("research/configs/data.json").read_text(encoding="utf-8"))
    for field in ["split_manifest", "prepared_path", "policy_config"]:
        require(Path(data[field]).is_file(), f"Missing data input: {data[field]}")
    split = json.loads(Path(data["split_manifest"]).read_text(encoding="utf-8"))
    require(Path(split["source"]).is_file(), f"Missing source corpus: {split['source']}")
    encoder = json.loads(Path("research/configs/encoder.json").read_text(encoding="utf-8"))
    require(re.fullmatch(r"[0-9a-f]{40}", encoder["revision"]), "Unpinned encoder revision")
    require(re.fullmatch(r"[^@]+@[0-9a-f]{40}", encoder["attention"]),
            "Unpinned attention kernel revision")


def main():
    require(Path("pyproject.toml").is_file(), "Run from the repository root")
    artifacts = check_artifacts()
    links = check_documents()
    python_files, imports = check_python()
    check_configs()
    print(json.dumps(dict(status="passed", artifacts=artifacts, local_links=links,
                          python_files=python_files, internal_imports=imports), indent=2))


if __name__ == "__main__":
    main()
