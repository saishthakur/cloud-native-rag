class AuthError(Exception):
    """Missing, malformed, expired or forged credentials (HTTP 401)."""


class ForbiddenError(Exception):
    """Authenticated but not allowed (HTTP 403)."""


class BlockedInputError(Exception):
    """Input rejected by the adversarial-input guard (HTTP 422)."""

    def __init__(self, message: str, verdict: dict | None = None):
        super().__init__(message)
        self.verdict = verdict or {}


class RateLimitedError(Exception):
    """Tenant exceeded its request budget (HTTP 429)."""
