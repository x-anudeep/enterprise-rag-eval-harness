from __future__ import annotations

from pathlib import Path

DOMAINS = {
    "hipaa_synthetic": {
        "prefix": "HIPAA operating procedure",
        "subjects": [
            "minimum necessary access",
            "audit controls",
            "workforce training",
            "breach notification",
            "risk analysis",
            "transmission security",
            "contingency planning",
            "device disposal",
            "role based access",
            "business associate review",
        ],
        "controls": [
            "privacy officer approval",
            "quarterly access review",
            "documented exception handling",
            "annual policy attestation",
            "incident response escalation",
        ],
    },
    "fda_synthetic": {
        "prefix": "FDA quality guidance",
        "subjects": [
            "software validation",
            "supplier qualification",
            "corrective action",
            "electronic records",
            "audit trail review",
            "design controls",
            "complaint handling",
            "process validation",
            "data integrity",
            "management review",
        ],
        "controls": [
            "risk based verification",
            "traceable requirements",
            "documented effectiveness checks",
            "change control approval",
            "inspection ready evidence",
        ],
    },
    "clinical_trials_synthetic": {
        "prefix": "Clinical trial protocol",
        "subjects": [
            "informed consent",
            "adverse event reporting",
            "data monitoring committee",
            "protocol deviations",
            "participant withdrawal",
            "source data verification",
            "safety oversight",
            "eligibility criteria",
            "randomization controls",
            "investigator training",
        ],
        "controls": [
            "sponsor notification",
            "institutional review board review",
            "documented medical assessment",
            "timely regulator reporting",
            "monitoring plan reconciliation",
        ],
    },
}


def build_document(domain: str, index: int, profile: dict[str, list[str] | str]) -> str:
    subjects = profile["subjects"]
    controls = profile["controls"]
    subject = subjects[index % len(subjects)]
    secondary = subjects[(index + 3) % len(subjects)]
    control = controls[index % len(controls)]
    cadence = ["monthly", "quarterly", "semiannual", "annual"][index % 4]
    threshold = 30 + index % 60
    prefix = profile["prefix"]
    title = f"{prefix} {index:04d}: {subject.title()}"
    return (
        f"{title}\n\n"
        f"This synthetic benchmark document describes {subject} requirements for the "
        f"{domain.replace('_synthetic', '').replace('_', ' ')} compliance program. "
        f"Teams must maintain {control} evidence and update the control owner when "
        f"exceptions affect patient safety, privacy, quality, or trial data integrity.\n\n"
        f"The procedure connects {subject} with {secondary} so retrieval tests can verify "
        f"both keyword matching and semantic matching. Records should be reviewed on a "
        f"{cadence} cadence, retained for at least {threshold} months, and made available "
        f"for audit, inspection, or sponsor oversight when requested.\n\n"
        f"Responsible teams should document the rationale, source system, reviewer, due date, "
        f"and closure evidence for every {subject} activity. Escalation is required when "
        f"the activity is overdue, incomplete, unsupported by evidence, or inconsistent "
        f"with the approved compliance procedure.\n"
    )


def main() -> None:
    root = Path("data/raw")
    per_domain = 334
    written = 0
    for domain, profile in DOMAINS.items():
        target_dir = root / domain
        target_dir.mkdir(parents=True, exist_ok=True)
        for index in range(1, per_domain + 1):
            path = target_dir / f"{domain}_benchmark_{index:04d}.txt"
            path.write_text(build_document(domain, index, profile), encoding="utf-8")
            written += 1
    print(f"Wrote {written} synthetic benchmark documents.")


if __name__ == "__main__":
    main()
