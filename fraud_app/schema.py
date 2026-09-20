"""
fraud_app.schema

Single description of every model feature: type, UI group, label and a short help text.

  - create_ieee_dataset.py writes these columns (plus ``is_fraud`` and ``TransactionDT``)
  - train_ieee_lgbm.py takes the numeric / categorical split from here
  - GET /api/schema serves it to the UI, which builds the input form from it

The trained pipeline stays the source of truth for what the *model* uses (meta["features"]);
the schema only adds labels and grouping, so unknown features still work (generic label).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

NUMERIC = "numeric"
CATEGORICAL = "categorical"


@dataclass(frozen=True)
class Feature:
    name: str
    kind: str
    group: str
    label: str
    help: str = ""
    # display names for categorical values, e.g. {"0": "C"}; values not listed show as-is
    option_labels: Dict[str, str] = field(default_factory=dict)


G_TXN = "Transaction"
G_HISTORY = "Sender history"
G_CARD = "Card & billing"
G_EMAIL = "Email & device"
G_RISK = "Location & risk signals"
G_C = "Vesta counters (C1–C14)"
G_D = "Vesta time deltas (D1–D15)"
G_M = "Vesta match flags (M1–M9)"
G_V = "Vesta engineered (V, selected)"
G_ID = "Identity & device (id_*)"

GROUP_ORDER: List[str] = [G_TXN, G_HISTORY, G_CARD, G_EMAIL, G_RISK, G_C, G_D, G_M, G_V, G_ID]


def _n(name, group, label, help=""):
    return Feature(name, NUMERIC, group, label, help)


def _c(name, group, label, help="", option_labels=None):
    return Feature(name, CATEGORICAL, group, label, help, option_labels or {})


_C_HELP = "Vesta counter (masked): counts such as addresses / devices linked to the card."
_D_HELP = "Vesta time delta in days (masked), e.g. since a previous transaction. Empty = unknown."
_M_HELP = "Vesta match flag (T/F, e.g. name or address match). Empty = unknown."

FEATURES: List[Feature] = [
    # --- Transaction (the original 12-feature demo set keeps its names) ---
    _n("amount", G_TXN, "Amount", "Transaction amount."),
    _n("amount_decimal", G_TXN, "Amount, decimal part", "Fractional part of the amount; odd fractions hint at currency conversion."),
    _n("hour", G_TXN, "Hour (0–23)", "Hour of day derived from TransactionDT."),
    _n("dow", G_TXN, "Day of cycle (0–6)", "(TransactionDT / 86400) mod 7. The real weekday offset is unknown."),
    _c(
        "mcc",
        G_TXN,
        "Product code",
        "ProductCD encoded as 0–4 (there is no real MCC in the IEEE data).",
        {"0": "0 – C", "1": "1 – H", "2": "2 – R", "3": "3 – S", "4": "4 – W"},
    ),
    # --- Sender history: aggregates over the card's earlier transactions (causal) ---
    _n("card_age_months", G_HISTORY, "Card age (months)", "Months since the card (card1) was first seen."),
    _n("sender_txn_24h", G_HISTORY, "Sender txn 24h", "Transactions of this card in the trailing 24h, including this one."),
    _n("sender_avg_amount", G_HISTORY, "Sender avg amount", "Mean amount of the card's previous transactions."),
    # --- Card & billing ---
    _n("card2", G_CARD, "card2", "Masked card attribute."),
    _n("card3", G_CARD, "card3", "Masked card attribute."),
    _c("card4", G_CARD, "Card network", "visa / mastercard / american express / discover."),
    _n("card5", G_CARD, "card5", "Masked card attribute."),
    _c("card6", G_CARD, "Card type", "debit / credit / charge card."),
    _n("addr1", G_CARD, "Billing region (addr1)", "Masked billing region."),
    _n("addr2", G_CARD, "Billing country (addr2)", "Masked billing country."),
    # --- Email & device ---
    _c("P_emaildomain", G_EMAIL, "Purchaser email domain", "e.g. gmail.com. Empty = unknown."),
    _c("R_emaildomain", G_EMAIL, "Recipient email domain", "e.g. gmail.com. Empty = unknown."),
    _n("has_identity", G_EMAIL, "Has identity record", "1 if a device/identity record exists for the transaction."),
    _c("DeviceType", G_EMAIL, "Device type", "mobile / desktop. Empty = unknown."),
    # --- Location & risk signals ---
    _n("distance_km", G_RISK, "Distance (dist1, 0 if unknown)", "Original demo feature: dist1 (or dist2), 0 when both missing."),
    _n("ip_risk", G_RISK, "IP risk (0–1)", "Proxy: min-max scaled mean |V1..V50|. Not a real IP score."),
    _c("country_risk", G_RISK, "Country risk", "Proxy: addr1 quantile bucket (0 low – 2 high or missing).", {"0": "0 – low", "1": "1 – medium", "2": "2 – high"}),
    _c("is_foreign", G_RISK, "Foreign transaction", "Proxy: dist1 > 500 or addr1 missing.", {"0": "No", "1": "Yes"}),
]

FEATURES += [_n(f"C{i}", G_C, f"C{i}", _C_HELP) for i in range(1, 15)]
FEATURES += [_n(f"D{i}", G_D, f"D{i}", _D_HELP) for i in range(1, 16)]
FEATURES += [_c(f"M{i}", G_M, f"M{i}", _M_HELP) for i in range(1, 10)]

# The 30 V columns (of V1..V339) that carry signal. Chosen by select_v_features.py from the
# training block only; using all 339 does not help.
V_SELECTED: List[str] = [
    "V13", "V20", "V30", "V44", "V45", "V53", "V54", "V67", "V70", "V76", "V87", "V133", "V149",
    "V156", "V188", "V189", "V201", "V206", "V223", "V243", "V244", "V258", "V283", "V294",
    "V310", "V313", "V315", "V317", "V318", "V323",
]
_V_HELP = "Vesta engineered feature (masked). One of the most useful V1–V339. Empty = unknown."
FEATURES += [_n(v, G_V, v, _V_HELP) for v in V_SELECTED]

# Identity / device attributes, present for about a quarter of the transactions.
ID_NUMERIC = [
    "id_01", "id_02", "id_03", "id_04", "id_05", "id_06", "id_07", "id_08", "id_09", "id_10",
    "id_11", "id_13", "id_14", "id_17", "id_18", "id_19", "id_20", "id_21", "id_22", "id_24",
    "id_25", "id_26", "id_32",
]
ID_CATEGORICAL = [
    "id_12", "id_15", "id_16", "id_23", "id_27", "id_28", "id_29", "id_30", "id_31", "id_33",
    "id_34", "id_35", "id_36", "id_37", "id_38", "DeviceInfo",
]
_ID_HELP = "Identity / network attribute (masked). Empty = unknown."
_ID_LABELS = {
    "id_30": ("Operating system", "OS name and version. Empty = unknown."),
    "id_31": ("Browser", "Browser name and version. Empty = unknown."),
    "id_33": ("Screen resolution", "e.g. 1920x1080. Empty = unknown."),
    "DeviceInfo": ("Device model", "Device model / build string. Empty = unknown."),
}
FEATURES += [_n(c, G_ID, c, _ID_HELP) for c in ID_NUMERIC]
FEATURES += [_c(c, G_ID, *_ID_LABELS.get(c, (c, _ID_HELP))) for c in ID_CATEGORICAL]

BY_NAME: Dict[str, Feature] = {f.name: f for f in FEATURES}
CATEGORICAL_NAMES: List[str] = [f.name for f in FEATURES if f.kind == CATEGORICAL]
NUMERIC_NAMES: List[str] = [f.name for f in FEATURES if f.kind == NUMERIC]


def describe(name: str) -> Feature:
    """Schema entry for ``name``; unknown features get a generic numeric entry."""
    return BY_NAME.get(name) or Feature(name, NUMERIC, "Other", name)


def group_index(group: str) -> int:
    return GROUP_ORDER.index(group) if group in GROUP_ORDER else len(GROUP_ORDER)


def find(name: str) -> Optional[Feature]:
    return BY_NAME.get(name)
