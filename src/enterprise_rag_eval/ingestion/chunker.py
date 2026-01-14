from __future__ import annotations

from enterprise_rag_eval.config import ChunkingConfig
from enterprise_rag_eval.models import Chunk, SourceDocument
from enterprise_rag_eval.text import split_sentences, tokenize


class SemanticChunker:
    """Sentence-aware chunker with token overlap."""

    def __init__(self, config: ChunkingConfig | None = None) -> None:
        self.config = config or ChunkingConfig()

    def chunk(self, documents: list[SourceDocument]) -> list[Chunk]:
        chunks: list[Chunk] = []
        for document in documents:
            chunks.extend(self._chunk_document(document))
        return chunks

    def _chunk_document(self, document: SourceDocument) -> list[Chunk]:
        sentences = split_sentences(document.text)
        chunks: list[Chunk] = []
        current: list[str] = []
        current_tokens = 0

        for sentence in sentences:
            sentence_tokens = len(tokenize(sentence))
            would_exceed = current and current_tokens + sentence_tokens > self.config.max_tokens
            if would_exceed:
                chunks.append(self._build_chunk(document, len(chunks), current))
                current = self._overlap_tail(current)
                current_tokens = len(tokenize(" ".join(current)))

            current.append(sentence)
            current_tokens += sentence_tokens

        if current_tokens >= self.config.min_tokens or not chunks:
            chunks.append(self._build_chunk(document, len(chunks), current))
        return chunks

    def _overlap_tail(self, sentences: list[str]) -> list[str]:
        tail: list[str] = []
        total = 0
        for sentence in reversed(sentences):
            total += len(tokenize(sentence))
            if total > self.config.overlap_tokens:
                break
            tail.insert(0, sentence)
        return tail

    @staticmethod
    def _build_chunk(document: SourceDocument, index: int, sentences: list[str]) -> Chunk:
        text = " ".join(sentences).strip()
        return Chunk(
            id=f"{document.id}::chunk-{index:03d}",
            document_id=document.id,
            title=document.title,
            domain=document.domain,
            text=text,
            token_count=len(tokenize(text)),
            metadata={"source_path": document.path},
        )
