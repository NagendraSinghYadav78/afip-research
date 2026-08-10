"""
Algorithm 1 — AFIP Master Orchestration.

Paper reference: Section 5.1, Figure 6, Algorithm 1 ("The seven step master
orchestration pipeline"). This module is a direct, runnable transcription
of the pseudocode in the paper:

    STEP 1 — Pre-inference safety check
    STEP 2 — Retrieve relevant knowledge
    STEP 3 — Module selection
    STEP 4 — Constitution-bounded LLM inference (retry up to 3x)
    STEP 5 — Post-inference check
    STEP 6 — Module-specific formatting and policy gates
    STEP 7 — Immutable audit log

Every step below is annotated with the paper line it implements. Where the
paper names a proprietary/unspecified component (DistilBERT classifier,
FAISS index, S3 Object Lock), this module implements a real, minimal,
honestly-labelled substitute (keyword-based safety screen, in-memory
cosine-similarity retrieval, hash-chained append-only log) so the control
flow is genuinely testable without proprietary infrastructure. Swap in the
real components via the constructor for a production/full-scale run.
"""

from __future__ import annotations
import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from jsonschema import validate as jsonschema_validate, ValidationError

from ..clients.base import LLMClient
from ..schemas.module_schemas import MODULE_SCHEMA

FINANCIAL_CONSTITUTION_PROMPT = (
    "You are a financial analysis assistant operating under a constitutional "
    "policy: (1) never issue a specific buy/sell/hold recommendation, "
    "(2) never fabricate a regulatory citation, (3) state uncertainty "
    "explicitly rather than expressing unwarranted confidence, "
    "(4) flag any output that may require registered-advisor review."
)


# ---------------------------------------------------------------------------
# STEP 1 — Pre-inference safety check
# Paper: "flags = SafetyClassifier(Q + D) // checks for PII, manipulation,
# or unregistered advice"
# ---------------------------------------------------------------------------

DEFAULT_BLOCK_KEYWORDS = (
    "guaranteed return", "insider tip", "ignore previous instructions",
    "act as an unregistered advisor", "social security number",
)


def default_safety_classifier(query: str, document: str | None) -> list[str]:
    """
    Minimal, transparent, keyword-based stand-in for the paper's DistilBERT
    pre-inference safety classifier (Section 5.1 / Limitations: "the
    DistilBERT safety classifier has not been independently validated").
    Returns a list of triggered flags (empty = pass).
    """
    text = f"{query}\n{document or ''}".lower()
    return [kw for kw in DEFAULT_BLOCK_KEYWORDS if kw in text]


# ---------------------------------------------------------------------------
# STEP 2 — Retrieve relevant knowledge
# Paper: "e = SentenceEmbed(Q) // 768 dimensions;
#         C = FAISS.kNN(e, k=5)"
# ---------------------------------------------------------------------------

def default_retriever(query: str, knowledge_base: list[str], k: int = 5) -> list[str]:
    """
    Minimal lexical-overlap retriever standing in for FAISS + a dense
    embedding model. Ranks knowledge_base entries by Jaccard token overlap
    with the query and returns the top k. This is intentionally simple and
    dependency-free; swap in a real embedding index for production use.
    """
    q_tokens = set(query.lower().split())

    def score(doc: str) -> float:
        d_tokens = set(doc.lower().split())
        if not q_tokens or not d_tokens:
            return 0.0
        return len(q_tokens & d_tokens) / len(q_tokens | d_tokens)

    ranked = sorted(knowledge_base, key=score, reverse=True)
    return ranked[:k]


# ---------------------------------------------------------------------------
# STEP 3 — Module selection
# ---------------------------------------------------------------------------

def default_router(module_hint: str) -> str:
    """
    Paper: "m = Router(Q) // picks the right module: M1, M2, M3, or M4".
    The full manuscript does not specify the routing classifier's decision
    procedure, so this harness takes an explicit module_hint per test case
    (see data/sample_cases.jsonl `module` field) rather than inventing an
    unvalidated intent classifier.
    """
    if module_hint not in MODULE_SCHEMA:
        raise ValueError(f"Unknown module '{module_hint}', expected one of {list(MODULE_SCHEMA)}")
    return module_hint


# ---------------------------------------------------------------------------
# STEP 4 helper — build the user-turn prompt
# ---------------------------------------------------------------------------

def build_prompt(module: str, query: str, context: list[str], document: str | None) -> str:
    parts = [f"Module: {module}", f"Query: {query}"]
    if context:
        parts.append("Retrieved context:\n" + "\n".join(f"- {c}" for c in context))
    if document:
        parts.append(f"Document:\n{document}")
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# STEP 7 — immutable-style audit log (hash-chained, append-only JSONL)
# ---------------------------------------------------------------------------

class AuditLog:
    """
    Append-only, hash-chained audit log. Each record's `entry_hash` commits
    to its own content plus the previous record's hash, so any retroactive
    edit to an earlier line is detectable by recomputing the chain — a
    genuinely verifiable local analogue of the paper's "SEC Rule 17a-4
    (record-retention)–oriented" audit trail, without asserting the S3
    Object Lock compliance claim the paper originally overstated.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._prev_hash = self._last_hash()

    def _last_hash(self) -> str:
        if not self.path.exists():
            return "0" * 64
        last = None
        with open(self.path, "r") as f:
            for line in f:
                if line.strip():
                    last = line
        if last is None:
            return "0" * 64
        return json.loads(last)["entry_hash"]

    def write(self, record: dict) -> dict:
        record = dict(record)
        record["prev_hash"] = self._prev_hash
        payload = json.dumps(record, sort_keys=True, default=str).encode()
        entry_hash = hashlib.sha256(payload).hexdigest()
        record["entry_hash"] = entry_hash
        with open(self.path, "a") as f:
            f.write(json.dumps(record, default=str) + "\n")
        self._prev_hash = entry_hash
        return record

    def verify(self) -> bool:
        """Recompute the hash chain and confirm no record has been altered."""
        prev = "0" * 64
        if not self.path.exists():
            return True
        with open(self.path, "r") as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                claimed_hash = rec.pop("entry_hash")
                if rec.get("prev_hash") != prev:
                    return False
                payload = json.dumps(rec, sort_keys=True, default=str).encode()
                if hashlib.sha256(payload).hexdigest() != claimed_hash:
                    return False
                prev = claimed_hash
        return True


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class OrchestrationResult:
    case_id: str
    module: str
    status: str  # "OK" | "BLOCKED" | "ESCALATE"
    output: dict[str, Any] | None
    attempts: int
    schema_valid: bool
    latency_seconds: float
    flags: list[str] = field(default_factory=list)
    audit_record: dict | None = None


# ---------------------------------------------------------------------------
# Algorithm 1 entry point
# ---------------------------------------------------------------------------

def run(
    case_id: str,
    query: str,
    module: str,
    client: LLMClient,
    document: str | None = None,
    knowledge_base: list[str] | None = None,
    ground_truth: dict | None = None,
    audit_log: AuditLog | None = None,
    safety_classifier: Callable[[str, str | None], list[str]] = default_safety_classifier,
    retriever: Callable[[str, list[str]], list[str]] = default_retriever,
    router: Callable[[str], str] = default_router,
    max_attempts: int = 3,
) -> OrchestrationResult:
    """Run one case through the full 7-step AFIP pipeline (Algorithm 1)."""
    knowledge_base = knowledge_base or []

    # STEP 1 — Pre-inference safety check
    flags = safety_classifier(query, document)
    if flags:
        result = OrchestrationResult(
            case_id=case_id, module=module, status="BLOCKED",
            output=None, attempts=0, schema_valid=False,
            latency_seconds=0.0, flags=flags,
        )
        if audit_log:
            result.audit_record = audit_log.write({
                "case_id": case_id, "module": module, "status": "BLOCKED",
                "flags": flags, "timestamp": time.time(),
            })
        return result

    # STEP 2 — Retrieve relevant knowledge
    context = retriever(query, knowledge_base)

    # STEP 3 — Module selection
    m = router(module)
    tool_schema = MODULE_SCHEMA[m]

    # STEP 4 — Constitution-bounded LLM inference (retry up to 3x)
    prompt = build_prompt(m, query, context, document)
    messages = [{"role": "user", "content": prompt}]

    output, attempts, schema_valid, total_latency = None, 0, False, 0.0
    for attempt in range(max_attempts):
        attempts += 1
        result = client.create_tool_call(
            system=FINANCIAL_CONSTITUTION_PROMPT,
            tool_schema=tool_schema,
            messages=messages,
            ground_truth=ground_truth,
            case_id=case_id,
        )
        total_latency += result.latency_seconds
        try:
            jsonschema_validate(instance=result.tool_input, schema=tool_schema["input_schema"])
            output, schema_valid = result.tool_input, True
            break
        except ValidationError:
            continue

    # STEP 5 — Post-inference check
    if not schema_valid:
        res = OrchestrationResult(
            case_id=case_id, module=m, status="ESCALATE",
            output=output, attempts=attempts, schema_valid=False,
            latency_seconds=total_latency, flags=flags,
        )
        if audit_log:
            res.audit_record = audit_log.write({
                "case_id": case_id, "module": m, "status": "ESCALATE",
                "attempts": attempts, "timestamp": time.time(),
            })
        return res

    # STEP 6 — Module-specific formatting and policy gates
    if m == "M3_RISK" and output.get("ips_violations"):
        output["advisor_review_required"] = True

    # STEP 7 — Immutable-style audit log
    audit_record = None
    if audit_log:
        audit_record = audit_log.write({
            "case_id": case_id, "module": m, "status": "OK",
            "query_hash": hashlib.sha256(query.encode()).hexdigest(),
            "output_hash": hashlib.sha256(json.dumps(output, sort_keys=True).encode()).hexdigest(),
            "model_id": client.model_id, "attempts": attempts,
            "flags": flags, "timestamp": time.time(),
        })

    return OrchestrationResult(
        case_id=case_id, module=m, status="OK", output=output,
        attempts=attempts, schema_valid=True, latency_seconds=total_latency,
        flags=flags, audit_record=audit_record,
    )
