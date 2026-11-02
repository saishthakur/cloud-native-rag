"""HS256 JSON Web Tokens carrying tenant, roles and clearance, plus per-tenant rate limiting.

Every service verifies the caller's token itself (zero-trust): the indexing service derives
the tenant namespace from the verified token, never from the request body.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import threading
import time

from ..core.models import Principal
from . import AuthError, ForbiddenError, RateLimitedError


def _b64e(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode()


def _b64d(s: str) -> bytes:
    return base64.urlsafe_b64decode(s + "=" * (-len(s) % 4))


class TokenService:
    def __init__(self, secret: str, issuer: str = "cloudrag", ttl_s: int = 3600):
        self._key = secret.encode()
        self.issuer = issuer
        self.ttl_s = ttl_s

    def issue(self, principal: Principal, ttl_s: int | None = None) -> str:
        now = int(time.time())
        header = {"alg": "HS256", "typ": "JWT"}
        claims = {
            "iss": self.issuer,
            "sub": principal.user_id,
            "tid": principal.tenant_id,
            "roles": principal.roles,
            "clr": principal.clearance,
            "iat": now,
            "exp": now + (ttl_s or self.ttl_s),
        }
        signing_input = f"{_b64e(json.dumps(header, separators=(',', ':')).encode())}." \
                        f"{_b64e(json.dumps(claims, separators=(',', ':')).encode())}"
        sig = hmac.new(self._key, signing_input.encode(), hashlib.sha256).digest()
        return f"{signing_input}.{_b64e(sig)}"

    def verify(self, token: str) -> Principal:
        try:
            h, c, s = token.split(".")
            header = json.loads(_b64d(h))
            claims = json.loads(_b64d(c))
            sig = _b64d(s)
        except Exception as e:  # noqa: BLE001 - any parse failure is an auth failure
            raise AuthError("malformed token") from e
        if header.get("alg") != "HS256":
            raise AuthError("unsupported token algorithm")
        expected = hmac.new(self._key, f"{h}.{c}".encode(), hashlib.sha256).digest()
        if not hmac.compare_digest(sig, expected):
            raise AuthError("invalid token signature")
        if claims.get("iss") != self.issuer:
            raise AuthError("invalid token issuer")
        if int(claims.get("exp", 0)) < time.time():
            raise AuthError("token expired")
        return Principal(
            user_id=claims["sub"],
            tenant_id=claims["tid"],
            roles=list(claims.get("roles", [])),
            clearance=int(claims.get("clr", 0)),
        )

    def from_header(self, authorization: str | None) -> tuple[Principal, str]:
        if not authorization or not authorization.lower().startswith("bearer "):
            raise AuthError("missing bearer token")
        token = authorization.split(" ", 1)[1].strip()
        return self.verify(token), token

    def service_token(self, service_name: str) -> str:
        """Machine identity for service-to-service calls that act on no user's behalf."""
        return self.issue(Principal(user_id=f"svc:{service_name}", tenant_id="*", roles=["service"]))


def require_roles(p: Principal, *roles: str) -> None:
    if not p.has_role("admin", *roles):
        raise ForbiddenError(f"requires one of roles: {', '.join(('admin',) + roles)}")


def require_service(p: Principal) -> None:
    if not p.is_service:
        raise ForbiddenError("internal endpoint: service identity required")


class TenantRateLimiter:
    """Token bucket per tenant so one noisy tenant cannot starve the others."""

    def __init__(self, rate_per_s: float, burst: int):
        self.rate = rate_per_s
        self.burst = burst
        self._buckets: dict[str, tuple[float, float]] = {}
        self._lock = threading.Lock()

    def check(self, tenant_id: str, cost: float = 1.0) -> None:
        if self.rate <= 0:
            return
        now = time.monotonic()
        with self._lock:
            tokens, last = self._buckets.get(tenant_id, (float(self.burst), now))
            tokens = min(self.burst, tokens + (now - last) * self.rate)
            if tokens < cost:
                self._buckets[tenant_id] = (tokens, now)
                raise RateLimitedError(f"rate limit exceeded for tenant {tenant_id}")
            self._buckets[tenant_id] = (tokens - cost, now)
