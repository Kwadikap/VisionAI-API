# auth.py
import os
import time
# import jwt
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Callable, Tuple
# from jwt import PyJWTError
import httpx
from fastapi import Depends, HTTPException, status, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt
from pydantic import BaseModel

from apps.vision.shared.types import Tier

JWT_SECRET = os.getenv("SESSION_TICKET_SECRET")
JWT_ALG = "HS256"
TICKET_TTL_MIN = 15
SESSION_TTL = 60 * 10 # 10 min

TENANT_SUB = os.getenv("TENANT_SUB", "")
TENANT_DOMAIN = os.getenv("TENANT_DOMAIN", "")
TENANT_ID = os.getenv("TENANT_ID", "")
API_AUDIENCE_ENV = os.getenv("API_AUDIENCE", "")
ROLE_CLAIM = os.getenv("ROLE_CLAIM", "roles")
EXT_ROLE_CLAIM = os.getenv("EXT_ROLE_CLAIM", "extension_appRole")

# Parse multiple audiences if provided as CSV  # >>> ADDED
AUDIENCES: List[str] = [a.strip() for a in API_AUDIENCE_ENV.split(",") if a.strip()]
# Fallback to the single value for jose if user only provided one
PRIMARY_AUDIENCE: Optional[str] = AUDIENCES[0] if AUDIENCES else None

# Use GUID form to avoid domain drift
OPENID_CONFIG_URL = (
    f"https://{TENANT_SUB}.ciamlogin.com/{TENANT_ID}/v2.0/.well-known/openid-configuration"
)
ISSUER = f"https://{TENANT_SUB}.ciamlogin.com/{TENANT_ID}/v2.0/"

bearer = HTTPBearer()

_openid_cache: Dict[str, Any] = {}
_jwks_cache: Dict[str, Any] = {}
_jwks_fetched_at = 0
_JWKS_TTL = 60 * 60  # 1h


class TokenPayload(BaseModel):
    iss: Optional[str] = None
    aud: Optional[Any] = None      # aud can be str or list in some IdPs  # >>> CHANGED
    exp: Optional[int] = None
    iat: Optional[int] = None
    nbf: Optional[int] = None
    scp: Optional[str] = None
    roles: Optional[List[str]] = None

    # allow any extra claims (including extension_* and emails)
    class Config:
        extra = "allow"


async def _get_openid() -> Dict[str, Any]:
    global _openid_cache
    if _openid_cache:
        return _openid_cache
    async with httpx.AsyncClient(timeout=10) as client:
        r = await client.get(OPENID_CONFIG_URL)
        r.raise_for_status()
        _openid_cache = r.json()
        return _openid_cache


async def _get_jwks() -> Dict[str, Any]:
    global _jwks_cache, _jwks_fetched_at
    now = time.time()
    if _jwks_cache and (now - _jwks_fetched_at) < _JWKS_TTL:
        return _jwks_cache
    conf = await _get_openid()
    jwks_uri = conf["jwks_uri"]
    async with httpx.AsyncClient(timeout=10) as client:
        r = await client.get(jwks_uri)
        r.raise_for_status()
        _jwks_cache = r.json()
        _jwks_fetched_at = now
        return _jwks_cache


def _get_key_for_kid(kid: str, jwks: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    for k in jwks.get("keys", []):
        if k.get("kid") == kid:
            return k
    return None


def extract_roles_from_claims(claims: Dict[str, Any]) -> List[str]:
    """Normalize roles from ROLE_CLAIM and EXT_ROLE_CLAIM into a lowercased, deduped list."""
    roles = claims.get(ROLE_CLAIM) or []
    if isinstance(roles, str):
        roles = [roles]
    elif not isinstance(roles, list):
        roles = []

    ext = claims.get(EXT_ROLE_CLAIM)
    if isinstance(ext, str) and ext:
        roles.append(ext)

    return sorted({r.lower() for r in roles})  # dedupe + lower  # >>> ADDED


def extract_subject_ids(claims: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
    """Return (tenant_id, object_id_or_sub)."""
    tid = claims.get("tid")
    oid = claims.get("oid") or claims.get("sub")
    return tid, oid  # >>> ADDED


async def validate_access_token_raw(token: str) -> TokenPayload:
    """
    Validate a Bearer token WITHOUT using FastAPI's HTTPBearer.
    Use this in /session/init where Authorization header is optional.
    """
    # Header sanity  # >>> ADDED
    try:
        unverified_header = jwt.get_unverified_header(token)
        if unverified_header.get("alg") != "RS256":
            raise ValueError("Unexpected alg")
    except Exception:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token header")

    # Fetch signing key (with one-time refresh on miss)  # >>> ADDED
    jwks = await _get_jwks()
    kid = unverified_header.get("kid", "")
    key = _get_key_for_kid(kid, jwks)
    if not key:
        # key rotation safe-guard
        _jwks_cache.clear()
        jwks = await _get_jwks()
        key = _get_key_for_kid(kid, jwks)
        if not key:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Unknown signing key")

    # Verify signature & claims  # >>> CHANGED (audience handling + leeway)
    try:
        decode_kwargs = dict(
            token=token,
            key=key,
            algorithms=["RS256"],
            issuer=ISSUER,
            options={"leeway": 90},  # small clock skew
        )
        if AUDIENCES:
            decode_kwargs["audience"] = AUDIENCES
        elif PRIMARY_AUDIENCE:
            decode_kwargs["audience"] = PRIMARY_AUDIENCE

        payload = jwt.decode(**decode_kwargs)
    except Exception:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")

    return TokenPayload(**payload)


# Keep the original dependency for protected routes, but route through the raw validator  # >>> CHANGED
async def verify_access_token(credentials: HTTPAuthorizationCredentials) -> TokenPayload:
    return await validate_access_token_raw(credentials.credentials)


def require_roles(*allowed: str) -> Callable[[TokenPayload], TokenPayload]:
    """
    Usage: Depends(require_roles("basic", "pro", "admin"))
    If no roles are provided, it only enforces "valid token required".
    """
    allowed_norm = {a.lower() for a in allowed}  # >>> CHANGED

    async def _dep(payload: TokenPayload = Depends(lambda cred=Depends(bearer): verify_access_token(cred))):
        claims = payload.model_dump()
        roles = extract_roles_from_claims(claims)  # >>> CHANGED (consistent role parsing)

        if allowed_norm:
            if not (set(roles) & allowed_norm):
                raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Forbidden")
        return payload

    return _dep


# Optional helper for “trying” auth in routes where it’s not required 
async def get_optional_claims_from_request(request: Request) -> Optional[TokenPayload]:
    auth = request.headers.get("Authorization")
    if not auth or not auth.lower().startswith("bearer "):
        return None
    try:
        token = auth.split(" ", 1)[1]
        return await validate_access_token_raw(token)
    except HTTPException:
        return None

def create_token(*, sid: str, tier: str, sub: Optional[str]) -> str:
    now = datetime.now(timezone.utc)
    payload = {
        "sid": sid, "tier": tier,
        "iat": int(now.timestamp()),
        "exp": int((now + timedelta(minutes=TICKET_TTL_MIN)).timestamp()),
    }
    # Only include 'sub' if it's not None
    if sub is not None:
        payload["sub"] = sub

    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALG)

def extract_token(request: Request) -> Optional[str]:
    token = request.cookies.get("token")
    if not token:
        return None
    return token

def validate_token(token: str) -> dict:
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALG])
        print(f"Decoded token")  # Debug log
        return payload
    except JWTError as e:
        print(f"JWT decode error: {e}")  # Debug log
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    except Exception as e:
        print(f"Unexpected error decoding ticket: {e}")  # Debug log
        raise HTTPException(status_code=401, detail="Error decoding token")

def get_agent_tier_from_roles(roles: list[str]) -> str:
    roles = {x.lower() for x in roles or []}
    if "admin" in roles or "advanced" in roles: return Tier.PRO.value
    if "pro" in roles: return Tier.PRO.value
    return Tier.BASIC.value