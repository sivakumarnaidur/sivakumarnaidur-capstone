"""src/debug/golden.py — writes the W12 golden set to disk.

The pipeline (pipeline.py) is fully synthetic and deterministic (no OpenAI, no
Qdrant, no cost), so this script just needs to materialise the 20-question
golden set as JSONL before the pipeline/scenario demos read it. Running this
file is idempotent: it always (re)writes the same fixed content.

The 20 questions are split into three difficulty bands (see each entry's
"notes" field):
    happy-path — single-hop questions answerable from one policy section (g001-g014)
    harder     — multi-hop questions needing two+ sections, or nuanced process detail (g015-g018)
    edge       — questions whose correct answer is a graceful refusal/redirect (g019-g020)
"""
from pathlib import Path

GOLDEN_JSONL = r"""{"id": "g001", "question": "What is the company's leave policy?", "ideal_answer": "Employees get 20 days of paid leave per year, accrued monthly at 1.67 days per month. Unused leave caps at 30 days; anything above that lapses on 31 December.", "notes": "happy-path. Source: HR-policy.pdf §3.1."}
{"id": "g002", "question": "How do I request a reimbursement for a work-related expense under $100?", "ideal_answer": "Submit the receipt in the expense portal within 30 days of the purchase. Items under $100 are auto-approved if they fit a standard category (travel, meals, supplies). Submission outside 30 days requires manager approval.", "notes": "happy-path. Source: Finance-policy.pdf §5.4."}
{"id": "g003", "question": "What is the standard working week?", "ideal_answer": "The standard working week is 40 hours, Monday to Friday, 09:00 to 17:30 with a one-hour lunch. Flexible start times between 08:00 and 10:00 are permitted with manager approval.", "notes": "happy-path. Source: HR-policy.pdf §1.2."}
{"id": "g004", "question": "What's the policy on bringing personal devices to work (BYOD)?", "ideal_answer": "Personal devices may be used to access company email and approved SaaS apps via the MDM enrollment. Devices must be password-protected, encrypted, and patched within 30 days of OS updates. Local storage of company documents is prohibited.", "notes": "happy-path. Source: IT-security.pdf §2.3."}
{"id": "g005", "question": "How do I request parental leave?", "ideal_answer": "Submit a parental leave request via the HR portal at least 8 weeks before the expected leave start date. Primary caregivers are entitled to 16 weeks paid; secondary caregivers to 4 weeks paid. Additional unpaid leave up to 12 months may be granted with manager approval.", "notes": "happy-path. Source: HR-policy.pdf §3.4."}
{"id": "g006", "question": "What's the dress code for client meetings?", "ideal_answer": "Business smart for in-person client meetings: collared shirt or blouse, smart trousers or skirt, closed-toe shoes. Jeans and sneakers are acceptable for internal meetings but not for client-facing ones.", "notes": "happy-path. Source: HR-policy.pdf §2.1."}
{"id": "g007", "question": "How do I report a security incident?", "ideal_answer": "Email security@company.com immediately and create a ticket in the IT portal flagged as 'Security Incident'. For incidents involving customer data, also notify the Data Protection Officer within 24 hours. Do not investigate the incident yourself — preserve evidence.", "notes": "happy-path. Source: IT-security.pdf §4.1."}
{"id": "g008", "question": "What's the process for requesting new software?", "ideal_answer": "Software requests go through the IT portal. Pre-approved software (listed in the IT catalogue) is auto-installed within 1 business day. New requests require justification, a security review (2-5 business days), and budget approval if the cost exceeds $500/year.", "notes": "happy-path. Source: IT-policy.pdf §3.2."}
{"id": "g009", "question": "Who do I contact about my pay or benefits?", "ideal_answer": "For pay queries, contact Payroll at payroll@company.com or via the HR portal under 'Pay & Benefits'. For benefits (health insurance, retirement, etc.), contact HR Benefits at benefits@company.com.", "notes": "happy-path. Source: HR-policy.pdf §6."}
{"id": "g010", "question": "What's the policy on working from home?", "ideal_answer": "Employees may work remotely up to 3 days per week with manager approval. Remote work requires a stable internet connection, a quiet workspace, and availability during core hours (10:00-15:00 local time). Full-time remote arrangements require VP-level approval.", "notes": "happy-path. Source: HR-policy.pdf §1.5."}
{"id": "g011", "question": "How do I book a meeting room?", "ideal_answer": "Meeting rooms are booked via the Outlook calendar. Rooms over 12 people require booking at least 24 hours in advance. Recurring meetings should not block premium rooms (those with video conferencing) for more than 2 hours per week.", "notes": "happy-path. Source: Facilities-policy.pdf §1.1."}
{"id": "g012", "question": "What's the company's policy on accepting gifts from clients?", "ideal_answer": "Gifts under $50 in value may be accepted with line-manager notification. Gifts above $50 must be declined or, if culturally inappropriate to decline, donated to charity with HR notified. Cash, gift cards, and personal services may never be accepted.", "notes": "happy-path. Source: Compliance-policy.pdf §2.1."}
{"id": "g013", "question": "How do I onboard a new contractor?", "ideal_answer": "Submit a contractor onboarding request via the HR portal at least 5 business days before the start date. Include scope of work, contract duration, budget code, and the contractor's NDA status. IT provisioning takes 2 business days from approval.", "notes": "happy-path. Source: HR-policy.pdf §5.1."}
{"id": "g014", "question": "What training is required for new employees?", "ideal_answer": "All new employees complete three mandatory trainings within 30 days of joining: (1) Information Security & Data Protection, (2) Code of Conduct & Anti-Harassment, (3) Health & Safety. Tracked via the L&D portal; manager is notified if not completed by day 30.", "notes": "happy-path. Source: HR-policy.pdf §4.2."}
{"id": "g015", "question": "How does the remote work policy interact with the requirement to attend in-person team meetings?", "ideal_answer": "Remote work allowance (up to 3 days per week) is subject to attending mandatory in-person team meetings as scheduled by the manager. Teams typically schedule one anchor in-person day per week. Recurring conflicts should be resolved by the team's manager, who may approve exceptions on a case-by-case basis.", "notes": "harder · multi-hop. Cross-references HR-policy §1.5 (remote) AND §2.2 (team meetings). Tests integration across two source sections."}
{"id": "g016", "question": "If I'm on parental leave and a serious security incident happens in my product area, am I expected to respond?", "ideal_answer": "No. Employees on parental leave are not expected to respond to work matters, including security incidents. The on-call rotation and the team's secondary contact (named in each team's runbook) cover incident response during any team member's absence. If the matter is critical and only the absent employee has the knowledge, the line manager escalates to the VP for coverage.", "notes": "harder · multi-hop. Cross-references HR-policy §3.4 (parental leave) AND IT-security §4.1 (incident response). Tests respect for leave + named escalation path."}
{"id": "g017", "question": "What's the right channel to report a potential conflict of interest involving a vendor my partner works for?", "ideal_answer": "Disclose the potential conflict via the Compliance portal under 'Conflict of Interest Disclosure'. Include the vendor name, your partner's role, and any business relationship you have with the vendor. The Compliance team reviews disclosures within 5 business days and may require recusal from procurement or vendor-management decisions involving that vendor.", "notes": "harder. Combines Compliance-policy §3 (disclosures) and process detail. Tests nuanced conflict-of-interest reporting."}
{"id": "g018", "question": "Can I expense a Coursera certification that's not in the L&D catalogue if it directly supports my project?", "ideal_answer": "Yes, with manager pre-approval. Submit a request via the L&D portal under 'Out-of-catalogue learning' with a one-paragraph business justification. Certifications up to $1,000 per year per employee are typically approved for project-relevant skills. Reimbursement requires successful completion (certificate issued) within 6 months.", "notes": "harder. Cross-references L&D-policy §2.4 and Finance §5. Tests handling of off-catalogue requests with conditions."}
{"id": "g019", "question": "What is the company's stance on cryptocurrency investing as part of the benefits package?", "ideal_answer": "Cryptocurrency investing is not part of the benefits package. The answer is not in the company's policy documents. The assistant should say so clearly and suggest the employee contact HR for clarification.", "notes": "edge. Tests graceful refusal — answer NOT in docs. Should not hallucinate."}
{"id": "g020", "question": "Will the company sponsor my visa for a personal trip to a country requiring a business invitation letter?", "ideal_answer": "Visa support is only provided for company-sanctioned business travel — see Travel-policy §3.2. Personal travel visas are the employee's responsibility. The company does not issue invitation letters for personal trips.", "notes": "edge. Tests boundary-setting — adjacent to a real policy but clearly outside its scope. Should redirect, not invent."}"""
# Anchor the output to week12/ (this file's grandparent), not the caller's cwd,
# so the golden set always lands in the same place regardless of where this is run from.
REPO_ROOT = Path(__file__).resolve().parents[2]  # week12/
GOLDEN_PATH = REPO_ROOT / "golden_set" / "capstone_v1.jsonl"


def write_golden_set(path: Path = GOLDEN_PATH) -> int:
    """(Re)write the golden set JSONL to `path`. Returns the question count."""
    path.parent.mkdir(exist_ok=True)
    path.write_text(GOLDEN_JSONL + "\n")
    return len(GOLDEN_JSONL.splitlines())


if __name__ == "__main__":
    n = write_golden_set()
    print(f"golden set written: {n} questions · pipeline is offline + deterministic")
    print(f"→ {GOLDEN_PATH}")
