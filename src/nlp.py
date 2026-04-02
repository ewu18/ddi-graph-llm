"""
NLP module for extracting interaction-type labels from DDI description text.
Updated after Phase 1 data exploration — covers 99.997% of dataset.
"""

import re
import pandas as pd
from typing import Optional


INTERACTION_PATTERNS = [
    # Metabolism (~40K edges)
    (r"metabolism.*(?:increased|decreased)", "metabolism"),
    (r"active metabolites", "metabolism"),

    # Serum concentration / bioavailability (~34K edges)
    (r"serum concentration.*(?:increased|decreased|reduced)", "concentration"),
    (r"bioavailability.*(?:increased|decreased)", "concentration"),
    (r"protein binding", "concentration"),

    # Adverse effects (~61K edges)
    (r"risk or severity of adverse effects", "adverse_effects"),
    (r"risk or severity of.*(?:prolongation|bleeding|hypotension|hypertension|heart failure|hyperkalemia)", "adverse_effects"),
    (r"risk of a hypersensitivity", "adverse_effects"),

    # Therapeutic efficacy (~8K edges)
    (r"therapeutic efficacy.*(?:decreased|increased)", "efficacy"),
    (r"decrease effectiveness", "efficacy"),

    # Absorption (~1K edges)
    (r"decrease in the absorption", "absorption"),
    (r"absorption.*(?:increased|decreased)", "absorption"),

    # Excretion (~1.9K edges)
    (r"excretion rate", "excretion"),

    # Activity changes (~45K edges, catch-all)
    (r"(?:increase|decrease).*(?:activities|activity)", "activity"),
]


def extract_label(description: str) -> Optional[str]:
    desc_lower = description.lower()
    for pattern, label in INTERACTION_PATTERNS:
        if re.search(pattern, desc_lower):
            return label
    return None


def extract_labels(df: pd.DataFrame, desc_col: str = "Interaction Description") -> pd.DataFrame:
    df = df.copy()
    df["label"] = df[desc_col].apply(extract_label)

    n_total = len(df)
    n_labeled = df["label"].notna().sum()

    print(f"Total: {n_total:,}  Labeled: {n_labeled:,} ({n_labeled/n_total:.2%})")
    print(df["label"].value_counts().to_string())

    return df
