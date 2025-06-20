"""
Additional Model Coverage Tests
@nist-controls: SA-11, CA-7
@evidence: Coverage tests for model edge cases
"""

from src.core.standards.models import NaturalLanguageMapping


class TestNaturalLanguageMappingCoverage:
    """Test NaturalLanguageMapping coverage"""

    def test_matches_exact_pattern(self):
        """Test exact pattern matching"""
        mapping = NaturalLanguageMapping(
            query_pattern="secure api",
            standard_refs=["SEC.api", "CS.api"],
            confidence=0.9,
            keywords=["secure", "api", "authentication"]
        )

        # Exact match
        assert mapping.matches("I need secure api guidelines")
        assert mapping.matches("SECURE API best practices")

    def test_matches_keyword_threshold(self):
        """Test keyword threshold matching"""
        mapping = NaturalLanguageMapping(
            query_pattern="authentication system",
            standard_refs=["SEC.auth"],
            confidence=0.8,
            keywords=["auth", "login", "user", "password", "security"]
        )

        # Has 3 out of 5 keywords (60%)
        assert mapping.matches("user login with password")

        # Has only 2 out of 5 keywords (40% - below threshold)
        assert not mapping.matches("user security")

    def test_matches_case_insensitive(self):
        """Test case insensitive matching"""
        mapping = NaturalLanguageMapping(
            query_pattern="data encryption",
            standard_refs=["SEC.crypto"],
            confidence=0.95,
            keywords=["encrypt", "decrypt", "cipher"]
        )

        assert mapping.matches("DATA ENCRYPTION methods")
        assert mapping.matches("Encrypt and Decrypt data")
