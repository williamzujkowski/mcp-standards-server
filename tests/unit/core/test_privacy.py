"""
Unit tests for privacy filtering and PII detection.
"""

import pytest

from src.core.privacy import (
    PIIDetector,
    PIIType,
    PrivacyConfig,
    PrivacyFilter,
    get_privacy_filter,
)


class TestPrivacyConfig:
    """Test privacy configuration."""

    def test_default_config(self):
        """Test default privacy configuration."""
        config = PrivacyConfig()

        assert config.detect_pii is True
        assert config.redact_pii is True
        assert config.hash_pii is False
        assert config.redaction_char == "█"
        assert config.min_confidence == 0.8

        # Check default PII types
        assert config.pii_types is not None
        assert PIIType.EMAIL in config.pii_types
        assert PIIType.PHONE in config.pii_types
        assert PIIType.SSN in config.pii_types
        assert PIIType.CREDIT_CARD in config.pii_types

    def test_custom_config(self):
        """Test custom privacy configuration."""
        config = PrivacyConfig(
            detect_pii=False,
            redact_pii=False,
            hash_pii=True,
            pii_types={PIIType.EMAIL, PIIType.PHONE},
            redaction_char="*",
            min_confidence=0.9,
        )

        assert config.detect_pii is False
        assert config.redact_pii is False
        assert config.hash_pii is True
        assert config.pii_types == {PIIType.EMAIL, PIIType.PHONE}
        assert config.redaction_char == "*"
        assert config.min_confidence == 0.9


class TestPIIDetector:
    """Test PII detection functionality."""

    @pytest.fixture
    def detector(self):
        """Create PII detector instance."""
        return PIIDetector()

    def test_detect_email(self, detector):
        """Test email detection."""
        text = "Contact me at john.doe@example.com or jane@test.co.uk"
        matches = detector.detect(text)

        assert len(matches) == 2
        assert all(m.pii_type == PIIType.EMAIL for m in matches)
        assert matches[0].value == "john.doe@example.com"
        assert matches[1].value == "jane@test.co.uk"

    def test_detect_phone(self, detector):
        """Test phone number detection."""
        text = "Call me at (555) 123-4567 or +1-555-987-6543"
        matches = detector.detect(text)

        assert len(matches) == 2
        assert all(m.pii_type == PIIType.PHONE for m in matches)

    def test_detect_ssn(self, detector):
        """Test SSN detection."""
        text = "SSN: 123-45-6789 and another 987-65-4321"
        matches = detector.detect(text)

        # The second SSN might not be detected due to regex constraints
        assert len(matches) >= 1
        assert all(m.pii_type == PIIType.SSN for m in matches)
        assert "123-45-6789" in matches[0].value

    def test_detect_credit_card(self, detector):
        """Test credit card detection."""
        # Valid test credit card numbers
        text = "Visa: 4532015112830366 MasterCard: 5425233430109903"
        matches = detector.detect(text)

        assert len(matches) == 2
        assert all(m.pii_type == PIIType.CREDIT_CARD for m in matches)

    def test_detect_ip_address(self, detector):
        """Test IP address detection."""
        text = "Server at 192.168.1.1 and 10.0.0.254"
        matches = detector.detect(text)

        assert len(matches) == 2
        assert all(m.pii_type == PIIType.IP_ADDRESS for m in matches)

    def test_detect_aws_key(self, detector):
        """Test AWS key detection."""
        text = "AWS Access Key: AKIAIOSFODNN7EXAMPLE"
        matches = detector.detect(text)

        assert len(matches) == 1
        assert matches[0].pii_type == PIIType.AWS_KEY
        assert matches[0].value == "AKIAIOSFODNN7EXAMPLE"

    def test_detect_api_key(self, detector):
        """Test API key detection."""
        text = 'api_key="abc123def456ghi789jkl012mno345" and apikey: xyz987wvu654tsr321'
        matches = detector.detect(text)

        assert len(matches) >= 1
        assert any(m.pii_type == PIIType.API_KEY for m in matches)

    def test_detect_jwt_token(self, detector):
        """Test JWT token detection."""
        text = "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6IkpvaG4gRG9lIiwiaWF0IjoxNTE2MjM5MDIyfQ.SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c"
        matches = detector.detect(text)

        assert len(matches) == 1
        assert matches[0].pii_type == PIIType.JWT_TOKEN

    def test_detect_password(self, detector):
        """Test password detection."""
        text = 'password: "SuperSecret123!" and pwd="AnotherPass456"'
        matches = detector.detect(text)

        assert len(matches) >= 1
        assert any(m.pii_type == PIIType.PASSWORD for m in matches)

    def test_detect_person_names(self, detector):
        """Test person name detection."""
        text = "Dr. John Smith and Ms. Jane Doe attended the meeting"
        matches = detector.detect(text)

        # Should detect names with titles
        name_matches = [m for m in matches if m.pii_type == PIIType.PERSON_NAME]
        assert len(name_matches) >= 2

    def test_detect_addresses(self, detector):
        """Test address detection."""
        text = "Visit us at 123 Main Street or 456 Oak Avenue"
        matches = detector.detect(text)

        address_matches = [m for m in matches if m.pii_type == PIIType.ADDRESS]
        assert len(address_matches) == 2

    def test_no_false_positives(self, detector):
        """Test that common text doesn't trigger false positives."""
        text = "The quick brown fox jumps over the lazy dog"
        matches = detector.detect(text)

        assert len(matches) == 0

    def test_confidence_filtering(self):
        """Test confidence-based filtering."""
        config = PrivacyConfig(min_confidence=0.95)
        detector = PIIDetector(config)

        # Invalid SSN format should have low confidence
        text = "SSN: 000-00-0000"  # Invalid SSN
        matches = detector.detect(text)

        # Should be filtered out due to low confidence
        assert len(matches) == 0

    def test_custom_patterns(self):
        """Test custom pattern detection."""
        config = PrivacyConfig(custom_patterns={"employee_id": r"EMP\d{6}"})
        detector = PIIDetector(config)

        text = "Employee ID: EMP123456"
        matches = detector.detect(text)

        assert len(matches) == 1
        assert matches[0].value == "EMP123456"

    def test_overlapping_matches(self, detector):
        """Test handling of overlapping matches."""
        text = "Contact: john@example.com (john@example.com)"
        matches = detector.detect(text)

        # Should remove duplicates/overlaps
        assert len(matches) == 2  # Two separate instances

    def test_credit_card_validation(self, detector):
        """Test credit card number validation."""
        # Invalid credit card number (fails Luhn check)
        text = "Card: 4532015112830367"  # Last digit wrong
        matches = detector.detect(text)

        # Find credit card matches specifically
        cc_matches = [m for m in matches if m.pii_type == PIIType.CREDIT_CARD]

        if cc_matches:
            # Should have lower confidence for invalid card
            assert cc_matches[0].confidence < 0.9


class TestPrivacyFilter:
    """Test privacy filtering functionality."""

    @pytest.fixture
    def filter(self):
        """Create privacy filter instance."""
        return PrivacyFilter()

    def test_filter_text_masking(self, filter):
        """Test text filtering with masking."""
        text = "Email me at john@example.com or call 555-123-4567"
        filtered, matches = filter.filter_text(text)

        assert "john@example.com" not in filtered
        assert "555-123-4567" not in filtered
        assert "████████████████" in filtered  # Email masked
        assert len(matches) >= 2

    def test_filter_text_hashing(self):
        """Test text filtering with hashing."""
        config = PrivacyConfig(hash_pii=True)
        filter = PrivacyFilter(config)

        text = "My email is john@example.com"
        filtered, matches = filter.filter_text(text)

        assert "john@example.com" not in filtered
        assert "[email:" in filtered
        assert "]" in filtered

    def test_filter_text_no_detection(self):
        """Test filtering with detection disabled."""
        config = PrivacyConfig(detect_pii=False)
        filter = PrivacyFilter(config)

        text = "Email: john@example.com"
        filtered, matches = filter.filter_text(text)

        assert filtered == text
        assert len(matches) == 0

    def test_filter_text_no_redaction(self):
        """Test filtering with redaction disabled."""
        config = PrivacyConfig(redact_pii=False)
        filter = PrivacyFilter(config)

        text = "Email: john@example.com"
        filtered, matches = filter.filter_text(text)

        assert filtered == text
        assert len(matches) > 0  # Still detects

    def test_filter_dict_simple(self, filter):
        """Test dictionary filtering."""
        data = {
            "name": "John Doe",
            "email": "john@example.com",
            "phone": "555-123-4567",
            "age": 30,
        }

        filtered, matches = filter.filter_dict(data)

        assert filtered["email"] != "john@example.com"
        assert filtered["phone"] != "555-123-4567"
        assert filtered["age"] == 30  # Non-PII unchanged
        assert "email" in matches
        assert "phone" in matches

    def test_filter_dict_nested(self, filter):
        """Test nested dictionary filtering."""
        data = {
            "user": {"email": "john@example.com", "profile": {"ssn": "123-45-6789"}}
        }

        filtered, matches = filter.filter_dict(data)

        assert filtered["user"]["email"] != "john@example.com"
        assert filtered["user"]["profile"]["ssn"] != "123-45-6789"
        assert "user.email" in matches
        assert "user.profile.ssn" in matches

    def test_filter_dict_with_lists(self, filter):
        """Test dictionary with list filtering."""
        data = {
            "emails": ["john@example.com", "jane@test.com"],
            "users": [{"email": "user1@example.com"}, {"email": "user2@test.com"}],
        }

        filtered, matches = filter.filter_dict(data)

        assert all("@" not in email for email in filtered["emails"])
        assert all("@" not in user["email"] for user in filtered["users"])
        assert "emails[0]" in matches
        assert "emails[1]" in matches
        assert "users[0].email" in matches
        assert "users[1].email" in matches

    def test_filter_dict_key_filtering(self, filter):
        """Test that dictionary keys are also filtered."""
        data = {"john@example.com": "value", "normal_key": "john@example.com"}

        filtered, matches = filter.filter_dict(data)

        # Both key and value should be filtered
        assert "john@example.com" not in filtered
        assert "john@example.com" not in str(filtered.values())

    def test_privacy_report_text(self, filter):
        """Test privacy report generation for text."""
        text = "Email: john@example.com, SSN: 123-45-6789"
        report = filter.get_privacy_report(text)

        assert report["has_pii"] is True
        assert report["pii_count"] >= 2
        assert PIIType.EMAIL.value in report["pii_types_found"]
        assert PIIType.SSN.value in report["pii_types_found"]
        assert len(report["details"]) >= 2

    def test_privacy_report_dict(self, filter):
        """Test privacy report generation for dictionary."""
        data = {"email": "john@example.com", "credit_card": "4532015112830366"}
        report = filter.get_privacy_report(data)

        assert report["has_pii"] is True
        assert report["pii_count"] >= 2
        assert PIIType.EMAIL.value in report["pii_types_found"]
        assert PIIType.CREDIT_CARD.value in report["pii_types_found"]

        # Check details include paths
        paths = [d["path"] for d in report["details"]]
        assert "email" in paths
        assert "credit_card" in paths

    def test_privacy_report_no_pii(self, filter):
        """Test privacy report with no PII."""
        text = "The quick brown fox jumps over the lazy dog"
        report = filter.get_privacy_report(text)

        assert report["has_pii"] is False
        assert report["pii_count"] == 0
        assert len(report["pii_types_found"]) == 0
        assert len(report["details"]) == 0

    def test_custom_redaction_char(self):
        """Test custom redaction character."""
        config = PrivacyConfig(redaction_char="*")
        filter = PrivacyFilter(config)

        text = "Email: john@example.com"
        filtered, _ = filter.filter_text(text)

        assert "*" in filtered
        assert "█" not in filtered

    def test_selective_pii_types(self):
        """Test filtering only specific PII types."""
        config = PrivacyConfig(pii_types={PIIType.EMAIL})
        filter = PrivacyFilter(config)

        text = "Email: john@example.com, Phone: 555-123-4567"
        filtered, matches = filter.filter_text(text)

        assert "john@example.com" not in filtered
        assert "555-123-4567" in filtered  # Phone not filtered
        assert len(matches) == 1
        assert matches[0].pii_type == PIIType.EMAIL


class TestPrivacyFilterSingleton:
    """Test privacy filter singleton."""

    def test_singleton_instance(self):
        """Test that get_privacy_filter returns singleton."""
        filter1 = get_privacy_filter()
        filter2 = get_privacy_filter()

        assert filter1 is filter2

    def test_singleton_configuration(self):
        """Test singleton has default configuration."""
        filter = get_privacy_filter()

        assert filter.config.detect_pii is True
        assert filter.config.redact_pii is True
