"""Generate the demo PDFs in sample_docs/.

Run once with the backend venv:
    backend\\.venv\\Scripts\\python.exe sample_docs\\make_samples.py

Writes plain, text-extractable PDFs (no external dependencies beyond pypdf's
own writer would allow, so we emit minimal PDF syntax directly).
"""

from pathlib import Path

HERE = Path(__file__).resolve().parent


def _escape(text: str) -> str:
    return text.replace("\\", r"\\").replace("(", r"\(").replace(")", r"\)")


def write_pdf(path: Path, pages: list[list[str]]) -> None:
    """Emit a minimal single-font PDF. Each page is a list of text lines."""
    objects: list[bytes] = []

    def add(body: bytes) -> int:
        objects.append(body)
        return len(objects)

    font_id = add(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")

    page_ids: list[int] = []
    content_ids: list[int] = []
    for lines in pages:
        stream_lines = ["BT", "/F1 11 Tf", "14 TL", "56 760 Td"]
        for line in lines:
            stream_lines.append(f"({_escape(line)}) Tj T*")
        stream_lines.append("ET")
        stream = "\n".join(stream_lines).encode("latin-1", "replace")
        content_ids.append(add(b"<< /Length %d >>\nstream\n%s\nendstream" % (len(stream), stream)))
        page_ids.append(0)  # placeholder, filled below

    pages_id = len(objects) + len(pages) + 1

    for i, content_id in enumerate(content_ids):
        page_ids[i] = add(
            b"<< /Type /Page /Parent %d 0 R /MediaBox [0 0 612 792] "
            b"/Resources << /Font << /F1 %d 0 R >> >> /Contents %d 0 R >>"
            % (pages_id, font_id, content_id)
        )

    kids = b" ".join(b"%d 0 R" % pid for pid in page_ids)
    real_pages_id = add(b"<< /Type /Pages /Kids [%s] /Count %d >>" % (kids, len(page_ids)))
    assert real_pages_id == pages_id, (real_pages_id, pages_id)
    catalog_id = add(b"<< /Type /Catalog /Pages %d 0 R >>" % pages_id)

    out = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for i, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += b"%d 0 obj\n" % i + body + b"\nendobj\n"

    xref_at = len(out)
    out += b"xref\n0 %d\n" % (len(objects) + 1)
    out += b"0000000000 65535 f \n"
    for off in offsets[1:]:
        out += b"%010d 00000 n \n" % off
    out += b"trailer\n<< /Size %d /Root %d 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (
        len(objects) + 1, catalog_id, xref_at,
    )

    path.write_bytes(bytes(out))
    print(f"wrote {path.name} ({len(pages)} pages, {len(out)} bytes)")


# ---------------------------------------------------------------------------

API_SPEC = [
    [
        "Orbital Payments API - Reference (v2.4)",
        "",
        "1. Authentication",
        "",
        "All requests must include a bearer token in the Authorization header.",
        "Tokens are issued by POST /v2/auth/token and expire after 3600 seconds.",
        "Refresh tokens are valid for 30 days and are single-use: each refresh",
        "invalidates the previous refresh token.",
        "",
        "Requests without a valid token return HTTP 401 with error code",
        "auth_token_invalid. Requests with an expired token return HTTP 401 with",
        "error code auth_token_expired.",
        "",
        "2. Rate Limiting",
        "",
        "The default rate limit is 120 requests per minute per API key. Bulk",
        "endpoints under /v2/bulk/ have a separate limit of 10 requests per minute.",
        "Exceeding a limit returns HTTP 429 with a Retry-After header expressed in",
        "seconds. The limit can be raised to 600 requests per minute on the",
        "Enterprise plan by contacting support.",
        "",
        "Rate limit state is reported on every response through three headers:",
        "X-RateLimit-Limit, X-RateLimit-Remaining, and X-RateLimit-Reset.",
    ],
    [
        "Orbital Payments API - Reference (v2.4)",
        "",
        "3. Payouts",
        "",
        "POST /v2/payouts creates a payout. Required fields are amount_cents",
        "(integer), currency (ISO 4217), and destination_id (string). The optional",
        "idempotency_key field prevents duplicate payouts when a request is retried;",
        "keys are retained for 24 hours.",
        "",
        "GET /v2/payouts/{payout_id} retrieves a single payout.",
        "GET /v2/payouts lists payouts, paginated with limit (max 100) and cursor.",
        "POST /v2/payouts/{payout_id}/cancel cancels a payout that is still pending.",
        "",
        "A payout moves through the states: pending, processing, paid, failed,",
        "cancelled. Only payouts in the pending state can be cancelled.",
        "",
        "4. Webhooks",
        "",
        "Webhook payloads are signed with HMAC-SHA256 using your endpoint secret.",
        "The signature is sent in the X-Orbital-Signature header. Verify it before",
        "trusting the payload. Failed deliveries are retried 8 times with",
        "exponential backoff over roughly 24 hours.",
        "",
        "Supported events: payout.created, payout.paid, payout.failed,",
        "payout.cancelled, and account.updated.",
    ],
]

RUNBOOK = [
    [
        "Platform Runbook - Incident Response",
        "",
        "1. Severity Levels",
        "",
        "SEV1 - Complete outage or data loss. Page the on-call lead immediately.",
        "       Target acknowledgement: 5 minutes. Status page update: 15 minutes.",
        "SEV2 - Major degradation affecting many customers. Acknowledge in 15",
        "       minutes. Status page update within 30 minutes.",
        "SEV3 - Minor degradation or a single-customer issue. Handle during",
        "       business hours. No status page update required.",
        "",
        "2. Escalation",
        "",
        "The primary on-call engineer owns the incident until it is resolved or",
        "formally handed over. If the primary does not acknowledge a SEV1 page",
        "within 10 minutes, PagerDuty automatically escalates to the secondary,",
        "and after a further 10 minutes to the engineering manager.",
        "",
        "3. Rollback Procedure",
        "",
        "Deployments are rolled back with: orbctl release rollback --env prod",
        "--to-previous. A rollback takes about 4 minutes to propagate across all",
        "regions. Database migrations are NOT reversed by a rollback and must be",
        "reverted separately using the down migration.",
        "",
        "Always post the rollback in #incidents before running it.",
    ],
    [
        "Platform Runbook - Service Ownership",
        "",
        "4. Service Registry",
        "",
        "Each service has a numeric identifier used in dashboards and alerts.",
        "",
        "ID 1  - payments-api       Owner: Priya Raman      Tier: 1",
        "        On-call rota: payments-primary. SLO: 99.95% availability.",
        "",
        "ID 2  - ledger-worker      Owner: Marcus Webb      Tier: 1",
        "        On-call rota: payments-primary. SLO: 99.9% availability.",
        "",
        "ID 3  - notification-svc   Owner: Aisha Khan       Tier: 2",
        "        On-call rota: platform-secondary. SLO: 99.5% availability.",
        "",
        "ID 4  - reporting-batch    Owner: Tom Iverson      Tier: 3",
        "        On-call rota: none (business hours only). No availability SLO.",
        "",
        "Tier 1 services require a two-person review for any production change.",
        "Tier 3 services may be deployed by a single engineer.",
        "",
        "5. Maintenance Windows",
        "",
        "Planned maintenance runs Sundays 02:00-04:00 UTC. Tier 1 services require",
        "14 days notice to customers; Tier 2 and 3 require 3 days notice.",
    ],
]


if __name__ == "__main__":
    write_pdf(HERE / "orbital-api-reference.pdf", API_SPEC)
    write_pdf(HERE / "platform-runbook.pdf", RUNBOOK)
