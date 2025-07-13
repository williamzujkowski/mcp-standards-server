"""
Privacy filtering and PII detection for MCP server.

Provides comprehensive PII detection and redaction capabilities.
"""

import hashlib
import logging
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class PIIType(str, Enum):
    """Types of PII that can be detected."""

    EMAIL = "email"
    PHONE = "phone"
    SSN = "ssn"
    CREDIT_CARD = "credit_card"
    IP_ADDRESS = "ip_address"
    AWS_KEY = "aws_key"
    API_KEY = "api_key"
    JWT_TOKEN = "jwt_token"  # nosec B105
    PASSWORD = "password"  # nosec B105
    PERSON_NAME = "person_name"
    ADDRESS = "address"
    DATE_OF_BIRTH = "date_of_birth"
    MEDICAL_ID = "medical_id"
    BANK_ACCOUNT = "bank_account"
    DRIVER_LICENSE = "driver_license"


@dataclass
class PIIMatch:
    """Represents a detected PII match."""

    pii_type: PIIType
    value: str
    start_pos: int
    end_pos: int
    confidence: float


@dataclass
class PrivacyConfig:
    """Configuration for privacy filtering."""

    detect_pii: bool = True
    redact_pii: bool = True
    hash_pii: bool = False
    pii_types: set[PIIType] | None = None
    custom_patterns: dict[str, str] | None = None
    redaction_char: str = "█"
    min_confidence: float = 0.8

    def __post_init__(self) -> None:
        """Set default PII types if not provided."""
        if self.pii_types is None:
            self.pii_types = {
                PIIType.EMAIL,
                PIIType.PHONE,
                PIIType.SSN,
                PIIType.CREDIT_CARD,
                PIIType.IP_ADDRESS,
                PIIType.AWS_KEY,
                PIIType.API_KEY,
                PIIType.JWT_TOKEN,
                PIIType.PASSWORD,
            }


class PIIDetector:
    """Detects PII in text using pattern matching and heuristics."""

    # Regex patterns for common PII types
    PATTERNS = {
        PIIType.EMAIL: r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
        PIIType.PHONE: r"(?:\+?1[-.\s]?)?\(?([0-9]{3})\)?[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})",
        PIIType.SSN: r"\b(?!000|666|9\d{2})\d{3}[-\s]?(?!00)\d{2}[-\s]?(?!0000)\d{4}\b",
        PIIType.CREDIT_CARD: r"\b(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|3[47][0-9]{13}|3(?:0[0-5]|[68][0-9])[0-9]{11}|6(?:011|5[0-9]{2})[0-9]{12}|(?:2131|1800|35\d{3})\d{11})\b",
        PIIType.IP_ADDRESS: r"\b(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b",
        PIIType.AWS_KEY: r"(?:AKIA|ABIA|ACCA|ASIA)[0-9A-Z]{16}",
        PIIType.API_KEY: r'(?:api[_-]?key|apikey|api_token)[\s]*[:=][\s]*["\']?([a-zA-Z0-9_\-]{20,})["\']?',
        PIIType.JWT_TOKEN: r"eyJ[a-zA-Z0-9_-]*\.eyJ[a-zA-Z0-9_-]*\.[a-zA-Z0-9_-]*",
        PIIType.PASSWORD: r'(?:password|passwd|pwd)[\s]*[:=][\s]*["\']?([^\s"\']{8,})["\']?',
        PIIType.BANK_ACCOUNT: r"\b[0-9]{8,17}\b",  # Simplified pattern
        PIIType.DRIVER_LICENSE: r"\b[A-Z]{1,2}[0-9]{5,8}\b",  # Simplified pattern
    }

    def __init__(self, config: PrivacyConfig | None = None) -> None:
        """Initialize PII detector with configuration."""
        self.config = config or PrivacyConfig()
        self._compiled_patterns: dict[PIIType | str, re.Pattern[str]] = {}
        self._compile_patterns()

    def _compile_patterns(self) -> None:
        """Compile regex patterns for efficiency."""
        if self.config.pii_types is None:
            return
        for pii_type in self.config.pii_types:
            if pii_type in self.PATTERNS:
                self._compiled_patterns[pii_type] = re.compile(
                    self.PATTERNS[pii_type],
                    re.IGNORECASE if pii_type != PIIType.AWS_KEY else 0,
                )

        # Add custom patterns
        if self.config.custom_patterns:
            for name, pattern in self.config.custom_patterns.items():
                self._compiled_patterns[name] = re.compile(pattern, re.IGNORECASE)

    def detect(self, text: str) -> list[PIIMatch]:
        """
        Detect PII in text.

        Args:
            text: Text to scan for PII

        Returns:
            List of PII matches found
        """
        if not self.config.detect_pii or not text:
            return []

        matches = []

        # Pattern-based detection
        for pii_type, pattern in self._compiled_patterns.items():
            for match in pattern.finditer(text):
                # Calculate confidence based on pattern match
                confidence = self._calculate_confidence(pii_type, match.group())

                if confidence >= self.config.min_confidence:
                    matches.append(
                        PIIMatch(
                            pii_type=(
                                pii_type
                                if isinstance(pii_type, PIIType)
                                else PIIType.API_KEY
                            ),
                            value=match.group(),
                            start_pos=match.start(),
                            end_pos=match.end(),
                            confidence=confidence,
                        )
                    )

        # Additional heuristic detection
        matches.extend(self._detect_person_names(text))
        matches.extend(self._detect_addresses(text))

        # Remove duplicates and overlaps
        return self._remove_overlaps(matches)

    def _calculate_confidence(self, pii_type: PIIType | str, value: str) -> float:
        """Calculate confidence score for a PII match."""
        base_confidence = 0.9

        # Adjust confidence based on context and validation
        if pii_type == PIIType.CREDIT_CARD:
            if self._validate_credit_card(value):
                return 0.95
            else:
                return 0.6

        elif pii_type == PIIType.EMAIL:
            if "@" in value and "." in value.split("@")[1]:
                return 0.95
            else:
                return 0.7

        elif pii_type == PIIType.SSN:
            if len(value.replace("-", "").replace(" ", "")) == 9:
                return 0.9
            else:
                return 0.5

        return base_confidence

    def _validate_credit_card(self, number: str) -> bool:
        """Validate credit card number using Luhn algorithm."""
        number = re.sub(r"\D", "", number)
        if not number:
            return False

        total = 0
        reverse_digits = number[::-1]

        for i, digit in enumerate(reverse_digits):
            n = int(digit)
            if i % 2 == 1:
                n *= 2
                if n > 9:
                    n -= 9
            total += n

        return total % 10 == 0

    def _detect_person_names(self, text: str) -> list[PIIMatch]:
        """Detect potential person names using heuristics."""
        matches = []

        # Simple heuristic: Capitalized words that might be names
        # This is a simplified approach - in production, use NLP libraries

        # Common name prefixes
        prefixes = ["Mr.", "Mrs.", "Ms.", "Dr.", "Prof."]
        for prefix in prefixes:
            pattern = f"{re.escape(prefix)}\\s+[A-Z][a-z]+ [A-Z][a-z]+"
            for match in re.finditer(pattern, text):
                matches.append(
                    PIIMatch(
                        pii_type=PIIType.PERSON_NAME,
                        value=match.group(),
                        start_pos=match.start(),
                        end_pos=match.end(),
                        confidence=0.85,
                    )
                )

        return matches

    def _detect_addresses(self, text: str) -> list[PIIMatch]:
        """Detect potential addresses using heuristics."""
        matches = []

        # Simple address pattern
        address_pattern = r"\b\d+\s+[A-Za-z\s]+(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Lane|Ln|Drive|Dr|Court|Ct|Place|Pl)\b"

        for match in re.finditer(address_pattern, text, re.IGNORECASE):
            matches.append(
                PIIMatch(
                    pii_type=PIIType.ADDRESS,
                    value=match.group(),
                    start_pos=match.start(),
                    end_pos=match.end(),
                    confidence=0.75,
                )
            )

        return matches

    def _remove_overlaps(self, matches: list[PIIMatch]) -> list[PIIMatch]:
        """Remove overlapping matches, keeping highest confidence."""
        if not matches:
            return []

        # Sort by start position and confidence
        sorted_matches = sorted(matches, key=lambda m: (m.start_pos, -m.confidence))

        result = []
        last_end = -1

        for match in sorted_matches:
            if match.start_pos >= last_end:
                result.append(match)
                last_end = match.end_pos

        return result


class PrivacyFilter:
    """Filters and redacts PII from data."""

    def __init__(self, config: PrivacyConfig | None = None) -> None:
        """Initialize privacy filter with configuration."""
        self.config = config or PrivacyConfig()
        self.detector = PIIDetector(config)
        self._redaction_cache: dict[str, tuple[str, list[PIIMatch]]] = {}

    def filter_text(self, text: str) -> tuple[str, list[PIIMatch]]:
        """
        Filter PII from text.

        Args:
            text: Text to filter

        Returns:
            Tuple of (filtered_text, matches)
        """
        if not self.config.detect_pii:
            return text, []

        # Detect PII
        matches = self.detector.detect(text)

        if not matches or not self.config.redact_pii:
            return text, matches

        # Redact PII
        filtered_text = self._redact_text(text, matches)

        return filtered_text, matches

    def filter_dict(
        self, data: dict[str, Any]
    ) -> tuple[dict[str, Any], dict[str, list[PIIMatch]]]:
        """
        Filter PII from dictionary recursively.

        Args:
            data: Dictionary to filter

        Returns:
            Tuple of (filtered_dict, matches_by_path)
        """
        filtered_data: dict[str, Any] = {}
        all_matches: dict[str, list[PIIMatch]] = {}

        for key, value in data.items():
            filtered_key, key_matches = self.filter_text(str(key))

            if key_matches:
                all_matches[f"key:{key}"] = key_matches

            if isinstance(value, str):
                filtered_value, value_matches = self.filter_text(value)
                if value_matches:
                    all_matches[key] = value_matches
                filtered_data[filtered_key] = filtered_value

            elif isinstance(value, dict):
                filtered_dict, nested_matches = self.filter_dict(value)
                filtered_data[filtered_key] = filtered_dict
                for nested_key, matches in nested_matches.items():
                    all_matches[f"{key}.{nested_key}"] = matches

            elif isinstance(value, list):
                filtered_list: list[Any] = []
                for i, item in enumerate(value):
                    if isinstance(item, str):
                        filtered_item, item_matches = self.filter_text(item)
                        if item_matches:
                            all_matches[f"{key}[{i}]"] = item_matches
                        filtered_list.append(filtered_item)
                    elif isinstance(item, dict):
                        filtered_dict_item, dict_item_matches = self.filter_dict(item)
                        filtered_list.append(filtered_dict_item)
                        for nested_key, matches in dict_item_matches.items():
                            all_matches[f"{key}[{i}].{nested_key}"] = matches
                    else:
                        filtered_list.append(item)
                filtered_data[filtered_key] = filtered_list

            else:
                filtered_data[filtered_key] = value

        return filtered_data, all_matches

    def _redact_text(self, text: str, matches: list[PIIMatch]) -> str:
        """Redact PII matches from text."""
        if self.config.hash_pii:
            return self._hash_redact(text, matches)
        else:
            return self._mask_redact(text, matches)

    def _mask_redact(self, text: str, matches: list[PIIMatch]) -> str:
        """Redact by masking with redaction character."""
        # Sort matches by position (reverse) to maintain positions
        sorted_matches = sorted(matches, key=lambda m: m.start_pos, reverse=True)

        result = text
        for match in sorted_matches:
            # Create redaction of same length
            redaction = self.config.redaction_char * (match.end_pos - match.start_pos)
            result = result[: match.start_pos] + redaction + result[match.end_pos :]

        return result

    def _hash_redact(self, text: str, matches: list[PIIMatch]) -> str:
        """Redact by replacing with hash."""
        sorted_matches = sorted(matches, key=lambda m: m.start_pos, reverse=True)

        result = text
        for match in sorted_matches:
            # Create consistent hash for the value
            hash_value = hashlib.sha256(match.value.encode()).hexdigest()[:8]
            redaction = f"[{match.pii_type.value}:{hash_value}]"
            result = result[: match.start_pos] + redaction + result[match.end_pos :]

        return result

    def get_privacy_report(self, data: Any) -> dict[str, Any]:
        """Generate privacy report for data."""
        report: dict[str, Any] = {
            "has_pii": False,
            "pii_types_found": set(),
            "pii_count": 0,
            "details": [],
        }

        if isinstance(data, str):
            _, matches = self.filter_text(data)
            if matches:
                report["has_pii"] = True
                report["pii_count"] = len(matches)
                for match in matches:
                    report["pii_types_found"].add(match.pii_type.value)
                    report["details"].append(
                        {"type": match.pii_type.value, "confidence": match.confidence}
                    )

        elif isinstance(data, dict):
            _, all_matches = self.filter_dict(data)
            if all_matches:
                report["has_pii"] = True
                for path, matches in all_matches.items():
                    report["pii_count"] += len(matches)
                    for match in matches:
                        report["pii_types_found"].add(match.pii_type.value)
                        report["details"].append(
                            {
                                "path": path,
                                "type": match.pii_type.value,
                                "confidence": match.confidence,
                            }
                        )

        pii_types_found = report["pii_types_found"]
        if isinstance(pii_types_found, set):
            report["pii_types_found"] = list(pii_types_found)
        return report


# Singleton instance
_privacy_filter: PrivacyFilter | None = None


def get_privacy_filter() -> PrivacyFilter:
    """Get the singleton privacy filter instance."""
    global _privacy_filter
    if _privacy_filter is None:
        _privacy_filter = PrivacyFilter()
    return _privacy_filter
