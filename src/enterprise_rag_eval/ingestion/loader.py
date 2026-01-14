from __future__ import annotations

from pathlib import Path

from enterprise_rag_eval.models import SourceDocument


class DocumentLoader:
    """Load plain-text compliance documents from a domain-organized folder tree."""

    def __init__(self, root: Path) -> None:
        self.root = root

    def load(self) -> list[SourceDocument]:
        documents: list[SourceDocument] = []
        for path in sorted(self.root.rglob("*.txt")):
            domain = path.parent.name.replace("_", " ")
            text = path.read_text(encoding="utf-8").strip()
            if not text:
                continue
            doc_id = path.relative_to(self.root).with_suffix("").as_posix().replace("/", "__")
            documents.append(
                SourceDocument(
                    id=doc_id,
                    title=path.stem.replace("_", " ").title(),
                    domain=domain,
                    path=str(path),
                    text=text,
                )
            )
        return documents
