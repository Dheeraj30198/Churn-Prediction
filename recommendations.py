from __future__ import annotations

from typing import Any


REASON_TO_ACTION = {
    "Competitor had better devices": "Offer a premium-device bundle upgrade at discounted price.",
    "Competitor made better offer": "Trigger a targeted retention offer with bill credits for 3 months.",
    "Competitor offered more data": "Upgrade plan data quota and waive upgrade fee for first cycle.",
    "Moved": "Offer relocation support and digital-only continuation plan.",
    "Price too high": "Recommend lower-cost plan or loyalty discount with same core benefits.",
    "Product dissatisfaction": "Schedule success call and tailor package to usage pattern.",
    "Service dissatisfaction": "Open priority support case and follow up within 24 hours.",
    "Network reliability": "Run service-quality diagnostics and offer backup connectivity option.",
    "Long distance charges": "Suggest plan with included long-distance/international calling.",
    "Lack of self-service on Website": "Guide customer to app onboarding + assign assisted digital setup.",
    "Poor expertise of online support": "Escalate to senior support team with dedicated owner.",
    "Limited range of services": "Offer bundle expansion (streaming/add-ons) at introductory discount.",
    "Extra data charges": "Recommend predictable unlimited/pooled data plan.",
    "Attitude of support person": "Route future interactions to retention-specialist support queue.",
    "Competitor": "Offer competitive bundle with time-bound retention discount.",
    "Pricing": "Offer right-sized plan and targeted discount based on usage.",
    "Product/Service": "Create proactive service-improvement case with retention follow-up.",
    "Support Experience": "Escalate to senior support and assign dedicated case owner.",
    "Digital Experience": "Provide assisted app/web onboarding and self-service walkthrough.",
    "Relocation": "Offer relocation continuity package and flexible contract terms.",
    "Other": "Schedule retention callback and tailor offer after needs assessment.",
}


def _is_yes(value: Any) -> bool:
    return str(value).strip().lower() in {"yes", "1", "true"}


def recommend_next_action(row: dict[str, Any], churn_probability: float, predicted_reason: str) -> str:
    monthly = float(row.get("Monthly Charges", 0) or 0)
    contract = str(row.get("Contract", "")).strip()
    tenure = float(row.get("Tenure Months", 0) or 0)
    tech_support = str(row.get("Tech Support", "")).strip()

    if churn_probability < 0.35:
        return "Low churn risk: continue engagement with standard loyalty messaging."

    if predicted_reason and predicted_reason in REASON_TO_ACTION:
        return REASON_TO_ACTION[predicted_reason]

    if contract == "Month-to-month" and tenure < 12:
        return "Offer annual contract conversion incentive with first-month discount."
    if monthly >= 85:
        return "Offer personalized cost-optimization plan with targeted discount."
    if not _is_yes(tech_support):
        return "Proactively enroll customer in priority technical support."

    return "Schedule retention outreach call and provide personalized plan review."
