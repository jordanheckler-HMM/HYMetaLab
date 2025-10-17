#!/usr/bin/env python3
"""
Guardian v7 - Corpus Indexer
Builds lightweight index of document claims and stances for consistency checking

v7 Memory & Consistency: Cross-document tracking without embeddings
"""
import hashlib
import json
import re
import sys
from datetime import datetime
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "qc" / "guardian_v4"))

try:
    from nlp.claim_classifier import ClaimClassifier

    V6_AVAILABLE = True
except ImportError:
    V6_AVAILABLE = False


class CorpusIndexer:
    """
    Build and maintain corpus-wide index of claims and stances
    Enables cross-document consistency checking
    """

    def __init__(self, root_path: Path = None):
        self.root = root_path or Path.cwd()
        self.memory_dir = self.root / "qc" / "guardian_v4" / "memory"
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.memory_dir / "index_v7.json"

        # Load claim classifier if available
        if V6_AVAILABLE:
            self.claim_classifier = ClaimClassifier()
        else:
            self.claim_classifier = None

        # Claim key patterns (topics we track)
        self.claim_patterns = {
            "openness": ["openness", "open", "transparency", "sharing"],
            "cooperation": ["cooperation", "collaborate", "collective", "shared"],
            "competition": ["competition", "competitive", "contest", "rivalry"],
            "resilience": ["resilience", "robust", "stability", "survive"],
            "coherence": ["coherence", "cci", "collective coherence", "integration"],
            "hazard": ["hazard", "risk", "threat", "failure"],
            "meaning": ["meaning", "purpose", "significance", "value"],
            "causality": ["causal", "cause", "effect", "mechanism"],
        }

        # Stance indicators
        self.positive_stance = {
            "increases",
            "improves",
            "enhances",
            "boosts",
            "strengthens",
            "promotes",
            "supports",
            "facilitates",
            "positively",
            "benefits",
        }

        self.negative_stance = {
            "decreases",
            "reduces",
            "weakens",
            "harms",
            "undermines",
            "inhibits",
            "negatively",
            "damages",
            "impairs",
            "degrades",
        }

        self.neutral_stance = {
            "relates",
            "associates",
            "correlates",
            "links",
            "connects",
            "suggests",
            "indicates",
            "shows",
            "demonstrates",
        }

    def compute_file_hash(self, file_path: Path) -> str:
        """Compute SHA256 hash of file content"""
        try:
            content = file_path.read_text(encoding="utf-8", errors="ignore")
            return hashlib.sha256(content.encode()).hexdigest()[:16]  # Short hash
        except Exception:
            return "unknown"

    def extract_claim_key(self, text: str) -> str:
        """Identify which topic this claim is about"""
        text_lower = text.lower()

        # Check each pattern
        for key, patterns in self.claim_patterns.items():
            if any(pattern in text_lower for pattern in patterns):
                return key

        return "general"

    def detect_stance(self, sentence: str) -> str:
        """
        Detect stance: positive, negative, or neutral
        Returns: 'positive', 'negative', 'neutral', or 'none'
        """
        sentence_lower = sentence.lower()
        words = set(re.findall(r"\b\w+\b", sentence_lower))

        positive_count = len(words & self.positive_stance)
        negative_count = len(words & self.negative_stance)
        neutral_count = len(words & self.neutral_stance)

        if positive_count > negative_count:
            return "positive"
        elif negative_count > positive_count:
            return "negative"
        elif neutral_count > 0:
            return "neutral"
        else:
            return "none"

    def extract_claims(self, text: str, file_path: Path) -> list[dict]:
        """
        Extract claims from document text
        Returns list of claim dictionaries
        """
        # Split into sentences
        sentences = re.split(r"[.!?]+\s+", text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 20]

        claims = []

        for i, sentence in enumerate(sentences):
            # Check if this looks like a claim (has stance + topic)
            stance = self.detect_stance(sentence)

            if stance in ["positive", "negative"]:
                claim_key = self.extract_claim_key(sentence)

                if claim_key != "general":  # Only index topical claims
                    claims.append(
                        {
                            "text": sentence[:200],  # Truncate
                            "claim_key": claim_key,
                            "stance": stance,
                            "sentence_index": i,
                            "file": str(file_path.relative_to(self.root)),
                        }
                    )

        return claims

    def index_document(self, file_path: Path) -> dict:
        """
        Index a single document
        Returns document entry for index
        """
        try:
            content = file_path.read_text(encoding="utf-8", errors="ignore")
        except Exception as e:
            return {"error": str(e), "file": str(file_path)}

        # Compute hash
        file_hash = self.compute_file_hash(file_path)

        # Extract claims
        claims = self.extract_claims(content, file_path)

        # Get file metadata
        stat = file_path.stat()

        return {
            "file": str(file_path.relative_to(self.root)),
            "hash": file_hash,
            "claims": claims,
            "claim_count": len(claims),
            "indexed_at": datetime.now().isoformat(),
            "modified_at": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            "size_bytes": stat.st_size,
        }

    def build_index(self, patterns: list[str] = None) -> dict:
        """
        Build complete corpus index

        Args:
            patterns: File patterns to index (default: docs, discovery_results, qc)
        """
        if patterns is None:
            patterns = ["docs/**/*.md", "discovery_results/**/*.md", "qc/*.md", "*.md"]

        print("🔍 Building Guardian v7 corpus index...")

        # Find all matching files
        all_files = []
        for pattern in patterns:
            all_files.extend(self.root.glob(pattern))

        # Remove duplicates
        all_files = list(set(all_files))

        # Filter out certain files
        exclude_patterns = ["node_modules", "__pycache__", ".git", "backups"]
        all_files = [
            f for f in all_files if not any(ex in str(f) for ex in exclude_patterns)
        ]

        print(f"   Found {len(all_files)} documents to index")

        # Load existing index if present
        existing_index = {}
        if self.index_path.exists():
            try:
                existing_index = json.load(open(self.index_path))
                print(
                    f"   Loaded existing index with {len(existing_index.get('documents', {}))} docs"
                )
            except Exception:
                pass

        # Index documents
        documents = {}
        indexed_count = 0
        cached_count = 0

        for i, file_path in enumerate(all_files):
            if (i + 1) % 50 == 0:
                print(f"   Progress: {i+1}/{len(all_files)}")

            # Check if cached
            file_hash = self.compute_file_hash(file_path)
            file_key = str(file_path.relative_to(self.root))

            if file_key in existing_index.get("documents", {}):
                existing_doc = existing_index["documents"][file_key]
                if existing_doc.get("hash") == file_hash:
                    # Use cached version
                    documents[file_key] = existing_doc
                    cached_count += 1
                    continue

            # Index document
            doc_entry = self.index_document(file_path)
            documents[file_key] = doc_entry
            indexed_count += 1

        print(f"   Indexed: {indexed_count} new, {cached_count} cached")

        # Build claim key index (for fast lookup)
        claim_key_index = {}
        for doc_key, doc_entry in documents.items():
            for claim in doc_entry.get("claims", []):
                key = claim["claim_key"]
                if key not in claim_key_index:
                    claim_key_index[key] = []

                claim_key_index[key].append(
                    {
                        "file": doc_entry["file"],
                        "text": claim["text"],
                        "stance": claim["stance"],
                        "modified_at": doc_entry.get("modified_at"),
                    }
                )

        # Compute statistics
        total_claims = sum(doc.get("claim_count", 0) for doc in documents.values())

        # Build index
        index = {
            "version": "7.0",
            "generated_at": datetime.now().isoformat(),
            "document_count": len(documents),
            "total_claims": total_claims,
            "indexed_count": indexed_count,
            "cached_count": cached_count,
            "documents": documents,
            "claim_key_index": claim_key_index,
            "claim_keys": list(self.claim_patterns.keys()),
        }

        # Save index
        self.index_path.write_text(json.dumps(index, indent=2))
        print("\n✅ Index complete:")
        print(f"   Documents: {len(documents)}")
        print(f"   Total claims: {total_claims}")
        print(f"   Claim keys: {len(claim_key_index)}")
        print(f"   Saved: {self.index_path}")

        return index

    def load_index(self) -> dict:
        """Load existing index"""
        if not self.index_path.exists():
            print("⚠️  No index found. Run build_index() first.")
            return {}

        return json.load(open(self.index_path))

    def query_claims(self, claim_key: str, stance: str = None) -> list[dict]:
        """
        Query claims by key and optional stance

        Args:
            claim_key: Topic key (e.g., 'openness', 'cooperation')
            stance: Optional stance filter ('positive', 'negative')

        Returns:
            List of matching claims
        """
        index = self.load_index()

        if not index:
            return []

        claim_key_index = index.get("claim_key_index", {})

        if claim_key not in claim_key_index:
            return []

        claims = claim_key_index[claim_key]

        if stance:
            claims = [c for c in claims if c["stance"] == stance]

        return claims


def main():
    """CLI interface for corpus indexer"""
    import argparse

    parser = argparse.ArgumentParser(description="Guardian v7 Corpus Indexer")
    parser.add_argument(
        "command", choices=["build", "query", "stats"], help="Command to execute"
    )
    parser.add_argument("--claim-key", type=str, help="Claim key for query")
    parser.add_argument(
        "--stance",
        type=str,
        choices=["positive", "negative"],
        help="Stance filter for query",
    )
    parser.add_argument(
        "--patterns", type=str, nargs="+", help="File patterns to index"
    )

    args = parser.parse_args()

    indexer = CorpusIndexer()

    if args.command == "build":
        print("📚 Building corpus index...")
        index = indexer.build_index(patterns=args.patterns)
        print(f"\n✅ Index built with {index['document_count']} documents")

    elif args.command == "query":
        if not args.claim_key:
            print("❌ Error: --claim-key required for query")
            return

        print(f"🔍 Querying claims: {args.claim_key} (stance: {args.stance or 'any'})")
        claims = indexer.query_claims(args.claim_key, args.stance)

        print(f"\n✅ Found {len(claims)} matching claims:\n")
        for i, claim in enumerate(claims[:10], 1):  # Show first 10
            print(f"{i}. [{claim['stance'].upper()}] {claim['text'][:80]}...")
            print(f"   File: {claim['file']}")
            print()

        if len(claims) > 10:
            print(f"   ... and {len(claims) - 10} more")

    elif args.command == "stats":
        print("📊 Index statistics:")
        index = indexer.load_index()

        if not index:
            print("❌ No index found. Run 'build' first.")
            return

        print(f"\n   Version: {index.get('version')}")
        print(f"   Generated: {index.get('generated_at')}")
        print(f"   Documents: {index.get('document_count')}")
        print(f"   Total claims: {index.get('total_claims')}")
        print(f"   Claim keys: {len(index.get('claim_key_index', {}))}")

        print("\n   Claim key distribution:")
        for key, claims in index.get("claim_key_index", {}).items():
            positive = sum(1 for c in claims if c["stance"] == "positive")
            negative = sum(1 for c in claims if c["stance"] == "negative")
            print(f"     • {key}: {len(claims)} total (+{positive}, -{negative})")


if __name__ == "__main__":
    main()
