"""Command-line extractive QA pipeline."""

import time
import argparse

from .chunker import chunk_document
from .retriever import build_index
from .qa_engine import extractive_qa
from .audit import create_audit_entry, write_audit_log


class ExtractiveQAPipeline:
    """Chunk, search, and answer from one document."""

    def __init__(self, window_size=200, overlap=40, theta=0.0,
                 alpha=0.5, beta=0.3, gamma=0.2, top_k=5):
        # Save scoring and chunking settings.
        self.window_size = window_size
        self.overlap = overlap
        self.theta = theta
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.top_k = top_k

        self.chunks = None
        self.vectors = None
        self.df = None
        self.N = None
        self.audit_entries = []

    def load_document(self, text):
        """Load and index a document."""
        # Rebuild the index for the new text.
        self.chunks = chunk_document(text, self.window_size, self.overlap)
        if self.chunks:
            self.vectors, self.df, self.N = build_index(self.chunks)
        return len(self.chunks)

    def load_file(self, path):
        """Load and index a document from a file path."""
        # Read text as UTF-8 throughout the project.
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        return self.load_document(text)

    def ask(self, query):
        """Answer a question from the loaded document."""
        # A query needs an index to search.
        if self.chunks is None:
            raise RuntimeError("No document loaded. Call load_document() first.")

        t0 = time.perf_counter()
        # Time only retrieval and answer selection.
        answer, confidence = extractive_qa(
            query, self.chunks, self.vectors, self.df, self.N,
            theta=self.theta, alpha=self.alpha, beta=self.beta,
            gamma=self.gamma, top_k=self.top_k,
        )
        latency_ms = (time.perf_counter() - t0) * 1000

        # Keep an in-memory audit until explicitly saved.
        entry = create_audit_entry(query, answer, confidence, latency_ms)
        self.audit_entries.append(entry)

        return {
            "answer": answer,
            "confidence": confidence,
            "latency_ms": latency_ms,
            "refused": answer is None,
        }

    def save_audit_log(self, path="audit_log.jsonl"):
        """Write all audit entries to file."""
        # Write after the session to reduce file access.
        write_audit_log(self.audit_entries, path)


def main():
    # Keep command-line setup in one place.
    parser = argparse.ArgumentParser(
        description="Extractive QA Pipeline — zero-dependency document assistant"
    )
    parser.add_argument("--document", required=True, help="Path to document file")
    parser.add_argument("--query", help="Single query to answer")
    parser.add_argument("--interactive", action="store_true", help="Interactive mode")
    parser.add_argument("--theta", type=float, default=0.0, help="Refusal threshold")
    parser.add_argument("--window-size", type=int, default=200, help="Chunk window size")
    parser.add_argument("--overlap", type=int, default=40, help="Chunk overlap")
    parser.add_argument("--alpha", type=float, default=0.5, help="Coverage weight")
    parser.add_argument("--beta", type=float, default=0.3, help="Density weight")
    parser.add_argument("--gamma", type=float, default=0.2, help="Length weight")
    parser.add_argument("--audit-log", default="audit_log.jsonl", help="Audit log path")
    args = parser.parse_args()

    pipeline = ExtractiveQAPipeline(
        window_size=args.window_size, overlap=args.overlap,
        theta=args.theta, alpha=args.alpha, beta=args.beta, gamma=args.gamma,
    )

    n_chunks = pipeline.load_file(args.document)
    print(f"Loaded document: {n_chunks} chunks indexed.")

    if args.query:
        result = pipeline.ask(args.query)
        if result["refused"]:
            print(f"Refused (confidence: {result['confidence']:.4f})")
        else:
            print(f"Answer: {result['answer']}")
            print(f"Confidence: {result['confidence']:.4f}")
            print(f"Latency: {result['latency_ms']:.3f} ms")

    elif args.interactive:
        # Reuse the same index for every question.
        print("Interactive mode. Type 'quit' to exit.\n")
        while True:
            query = input("Query: ").strip()
            if query.lower() in ("quit", "exit", "q"):
                break
            if not query:
                continue
            result = pipeline.ask(query)
            if result["refused"]:
                print(f"  Insufficient evidence (confidence: {result['confidence']:.4f})")
            else:
                print(f"  Answer: {result['answer']}")
                print(f"  Confidence: {result['confidence']:.4f} | Latency: {result['latency_ms']:.3f} ms")
            print()

    pipeline.save_audit_log(args.audit_log)
    print(f"Audit log saved to {args.audit_log}")


if __name__ == "__main__":
    main()
