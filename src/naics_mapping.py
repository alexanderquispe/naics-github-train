"""
NAICS Code Mappings and Descriptions.

This module provides mappings between NAICS (North American Industry
Classification System) codes and their descriptions.
"""

from typing import Dict, List, Optional

# Standard NAICS 2-digit codes and their descriptions
NAICS_CODE_TO_DESCRIPTION: Dict[str, str] = {
    "11": "Agriculture, Forestry, Fishing and Hunting",
    "21": "Mining, Quarrying, and Oil and Gas Extraction",
    "22": "Utilities",
    "23": "Construction",
    "31": "Manufacturing",
    "32": "Manufacturing",
    "33": "Manufacturing",
    "31-33": "Manufacturing",
    "42": "Wholesale Trade",
    "44": "Retail Trade",
    "45": "Retail Trade",
    "44-45": "Retail Trade",
    "48": "Transportation and Warehousing",
    "49": "Transportation and Warehousing",
    "48-49": "Transportation and Warehousing",
    "51": "Information",
    "52": "Finance and Insurance",
    "53": "Real Estate and Rental and Leasing",
    "54": "Professional, Scientific, and Technical Services",
    "55": "Management of Companies and Enterprises",
    "56": "Administrative and Support and Waste Management Services",
    "61": "Educational Services",
    "62": "Health Care and Social Assistance",
    "71": "Arts, Entertainment, and Recreation",
    "72": "Accommodation and Food Services",
    "81": "Other Services (except Public Administration)",
    "92": "Public Administration",
}

# Primary 2-digit NAICS codes (no range codes)
PRIMARY_NAICS_CODES: List[str] = [
    "11", "21", "22", "23", "31", "32", "33", "42", "44", "45",
    "48", "49", "51", "52", "53", "54", "55", "56", "61", "62",
    "71", "72", "81", "92"
]

# Unique NAICS sectors (collapsing ranges)
NAICS_SECTORS: Dict[str, str] = {
    "11": "Agriculture, Forestry, Fishing and Hunting",
    "21": "Mining, Quarrying, and Oil and Gas Extraction",
    "22": "Utilities",
    "23": "Construction",
    "31-33": "Manufacturing",
    "42": "Wholesale Trade",
    "44-45": "Retail Trade",
    "48-49": "Transportation and Warehousing",
    "51": "Information",
    "52": "Finance and Insurance",
    "53": "Real Estate and Rental and Leasing",
    "54": "Professional, Scientific, and Technical Services",
    "55": "Management of Companies and Enterprises",
    "56": "Administrative and Support and Waste Management Services",
    "61": "Educational Services",
    "62": "Health Care and Social Assistance",
    "71": "Arts, Entertainment, and Recreation",
    "72": "Accommodation and Food Services",
    "81": "Other Services (except Public Administration)",
    "92": "Public Administration",
}


def get_naics_description(code: str) -> Optional[str]:
    """
    Get the description for a NAICS code.

    Args:
        code: NAICS code (2-digit or range like "31-33")

    Returns:
        Description string if found, None otherwise
    """
    code = str(code).strip()
    return NAICS_CODE_TO_DESCRIPTION.get(code)


def get_all_naics_codes() -> List[str]:
    """
    Get all valid NAICS codes.

    Returns:
        List of all NAICS code strings
    """
    return list(NAICS_CODE_TO_DESCRIPTION.keys())


def get_primary_naics_codes() -> List[str]:
    """
    Get primary 2-digit NAICS codes (excluding range codes).

    Returns:
        List of primary NAICS codes
    """
    return PRIMARY_NAICS_CODES.copy()


def get_naics_sectors() -> Dict[str, str]:
    """
    Get NAICS sectors with consolidated ranges.

    Returns:
        Dictionary mapping sector codes to descriptions
    """
    return NAICS_SECTORS.copy()


def normalize_naics_code(code: str) -> str:
    """
    Normalize a NAICS code to a standard 2-digit form.

    For codes that are part of a range (31, 32, 33 -> Manufacturing),
    this returns the individual code, not the range.

    Args:
        code: NAICS code to normalize

    Returns:
        Normalized code string
    """
    code = str(code).strip()

    # Handle 3+ digit codes by taking first 2 digits
    if len(code) > 2 and "-" not in code:
        code = code[:2]

    return code


def is_valid_naics_code(code: str) -> bool:
    """
    Check if a code is a valid NAICS code.

    Args:
        code: Code to validate

    Returns:
        True if valid, False otherwise
    """
    code = str(code).strip()
    return code in NAICS_CODE_TO_DESCRIPTION


def add_naics_descriptions(df, code_column: str = "code"):
    """
    Add NAICS descriptions to a DataFrame.

    Args:
        df: pandas DataFrame with NAICS codes
        code_column: Name of the column containing NAICS codes

    Returns:
        DataFrame with added 'naics_description' column
    """
    import pandas as pd

    df = df.copy()
    df["naics_description"] = df[code_column].astype(str).map(NAICS_CODE_TO_DESCRIPTION)

    # Report unmapped codes
    unmapped = df["naics_description"].isna().sum()
    if unmapped > 0:
        unmapped_codes = df[df["naics_description"].isna()][code_column].unique()
        print(f"Warning: {unmapped} rows have unmapped NAICS codes: {unmapped_codes}")

    return df
