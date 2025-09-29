# auth.py
import os
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Callable, Tuple

import httpx, time
from fastapi import Depends, HTTPException, status, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt
from pydantic import BaseModel

from apps.vision.shared.types import Tier

# ---- Config -----------------------------------------------------------------

JWT_SECRET = os.getenv("SESSION_TICKET_SECRET")
JWT_ALG = "HS256"
TICKET_TTL_MIN = 15
SESSION_TTL = 60 * 10  # 10 min

TENANT_ID = os.getenv("TENANT_ID", "").strip()
API_AUDIENCE_ENV = os.getenv("API_AUDIENCE", "")
ROLE_CLAIM = os.getenv("ROLE_CLAIM", "roles")
EXT_ROLE_CLAIM = os.getenv("EXT_ROLE_CLAIM", "extension_appRole")

# Accept comma-separated audiences
AUDIENCES: List[str] = [a.strip() for a in API_AUDIENCE_ENV.split(",") if a.strip()]
PRIMARY_AUDIENCE: Optional[str] = AUDIENCES[0] if AUDIENCES else None

if not TENANT_ID:
    raise RuntimeError("TENANT_ID must be set")
if not JWT_SECRET:
    raise RuntimeError("SESSION_TICKET_SECRET must be set")

OPENID_CONFIG_URL = (
    f"https://login.microsoftonline.com/{TENANT_ID}/v2.0/.well-known/openid-configuration"
)

bearer = HTTPBearer()

# ---- Caches -----------------------------------------------------------------

_openid_cache: Dict[str, Any] = {}
_jwks_cache: Dict[str, Any] = {}
_jwks_fetched_at = 0
_JWKS_TTL = 60 * 60  # 1h

# ---- Models -----------------------------------------------------------------

class TokenPayload(BaseModel):
    iss: Optional[str] = None
    aud: Optional[Any] = None    # may be str or list
    exp: Optional[int] = None
    iat: Optional[int] = None
    nbf: Optional[int] = None
    scp: Optional[str] = None
    roles: Optional[List[str]] = None
    tid: Optional[str] = None
    ver: Optional[str] = None

    class Config:
        extra = "allow"           # keep all extra claims available


# ---- OIDC & JWKS ------------------------------------------------------------

async def _get_openid() -> Dict[str, Any]:
    if _openid_cache:
        return _openid_cache
    async with httpx.AsyncClient(timeout=10) as client:
        r = await client.get(OPENID_CONFIG_URL)
        r.raise_for_status()
        _openid_cache.update(r.json())
        return _openid_cache
    
async def _fetch_openid_from_issuer(issuer: str) -> dict:
    # Try issuer/.well-known/openid-configuration, then issuer/v2.0/...
    urls = [
        issuer.rstrip("/") + "/.well-known/openid-configuration",
        issuer.rstrip("/") + "/v2.0/.well-known/openid-configuration",
    ]
    last = None
    async with httpx.AsyncClient(timeout=10) as client:
        for u in urls:
            try:
                r = await client.get(u)
                r.raise_for_status()
                return r.json()
            except Exception as e:
                last = e
    raise HTTPException(500, f"OIDC discovery failed for {issuer}: {last}")


def _get_key_for_kid(kid: str, jwks: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    for k in jwks.get("keys", []):
        if k.get("kid") == kid:
            return k
    return None

# ---- Claim utilities ---------------------------------------------------------

def extract_roles_from_claims(claims: Dict[str, Any]) -> List[str]:
    roles = claims.get(ROLE_CLAIM) or []
    if isinstance(roles, str):
        roles = [roles]
    elif not isinstance(roles, list):
        roles = []
    ext = claims.get(EXT_ROLE_CLAIM)
    if isinstance(ext, str) and ext:
        roles.append(ext)
    return sorted({r.lower() for r in roles})

def extract_subject_ids(claims: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
    tid = claims.get("tid")
    oid = claims.get("oid") or claims.get("sub")
    return tid, oid

# ---- Access token validation -------------------------------------------------

async def validate_access_token_raw(token: str) -> TokenPayload:
    # 0) Unverified header/claims so we can pick the right issuer
    try:
        hdr = jwt.get_unverified_header(token)
        claims_unverified = jwt.get_unverified_claims(token)
    except Exception:
        raise HTTPException(401, "Invalid token header/payload")

    if hdr.get("alg") != "RS256":
        raise HTTPException(401, f"Unexpected alg {hdr.get('alg')}")
    kid = hdr.get("kid")
    if not kid:
        raise HTTPException(401, "Missing kid")
    iss = claims_unverified.get("iss")
    if not iss:
        raise HTTPException(401, "Missing iss")

    # 1) Discover from the token's issuer (works for ciamlogin.com, b2clogin.com, gov clouds, v1/v2)
    conf = await _fetch_openid_from_issuer(iss)
    issuer = conf.get("issuer")
    jwks_uri = conf.get("jwks_uri")
    if not issuer or not jwks_uri:
        raise HTTPException(500, "OIDC discovery missing issuer/jwks_uri")

    # 2) Fetch JWKS and match kid (with rotation retry)
    global _jwks_cache, _jwks_fetched_at
    now = time.time()
    jwks = _jwks_cache if (_jwks_cache and now - _jwks_fetched_at < _JWKS_TTL) else None
    if not jwks:
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.get(jwks_uri)
            r.raise_for_status()
            jwks = r.json()
            _jwks_cache, _jwks_fetched_at = jwks, now

    key = _get_key_for_kid(kid, jwks)
    if not key:
        # rotation: force refresh once
        _jwks_cache = {}
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.get(jwks_uri)
            r.raise_for_status()
            jwks = r.json()
            _jwks_cache, _jwks_fetched_at = jwks, time.time()
        key = _get_key_for_kid(kid, jwks)
        if not key:
            raise HTTPException(401, "Unknown signing key")

    # 3) Decode + verify (single audience to jose, then optional allowlist)
    try:
        options = {"leeway": 90, "verify_aud": False}
        kwargs = dict(token=token, key=key, algorithms=["RS256"], issuer=issuer, options=options)
        if PRIMARY_AUDIENCE:
            kwargs["audience"] = PRIMARY_AUDIENCE
        payload = jwt.decode(**kwargs)
        aud_claim = payload.get("aud")
        aud_values = set(aud_claim if isinstance(aud_claim, list) else [aud_claim])
        allowed = set(AUDIENCES)  # parsed from API_AUDIENCE
        if allowed and aud_values.isdisjoint(allowed):
            raise HTTPException(401, "Invalid audience")
    except JWTError as e:
        raise HTTPException(401, f"Invalid token: {e}")

    if AUDIENCES:
        aud = payload.get("aud")
        aud_values = set(aud if isinstance(aud, list) else [aud])
        if aud_values.isdisjoint(set(AUDIENCES)):
            raise HTTPException(401, "Invalid audience")

    # Optional tenant pin: only keep if you are *not* multi-tenant
    tid = payload.get("tid")
    if tid and TENANT_ID and tid != TENANT_ID:
        raise HTTPException(401, "Wrong tenant")

    return TokenPayload(**payload)

# FastAPI dependency (uses the raw validator)
async def verify_access_token(credentials: HTTPAuthorizationCredentials) -> TokenPayload:
    return await validate_access_token_raw(credentials.credentials)

def require_roles(*allowed: str) -> Callable[[TokenPayload], TokenPayload]:
    allowed_norm = {a.lower() for a in allowed}
    async def _dep(payload: TokenPayload = Depends(lambda cred=Depends(bearer): verify_access_token(cred))):
        claims = payload.model_dump()
        roles = extract_roles_from_claims(claims)
        if allowed_norm and not (set(roles) & allowed_norm):
            raise HTTPException(status_code=403, detail="Forbidden")
        return payload
    return _dep

# ---- Optional "try auth" helper ---------------------------------------------

async def get_optional_claims_from_request(request: Request) -> Optional[TokenPayload]:
    auth = request.headers.get("Authorization")
    if not auth or not auth.lower().startswith("bearer "):
        return None
    token = auth.split(" ", 1)[1]
    try:
        return await validate_access_token_raw(token)
    except HTTPException:
        return None

# ---- Your app's internal HS256 ticket (unchanged) ---------------------------

def create_token(*, sid: str, tier: str, sub: Optional[str]) -> str:
    now = datetime.now(timezone.utc)
    payload = {
        "sid": sid,
        "tier": tier,
        "iat": int(now.timestamp()),
        "exp": int((now + timedelta(minutes=TICKET_TTL_MIN)).timestamp()),
    }
    if sub is not None:
        payload["sub"] = sub
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALG)

def extract_token(request: Request) -> Optional[str]:
    return request.cookies.get("token") or None

def validate_token(token: str) -> dict:
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALG])
        return payload
    except JWTError as e:
        raise HTTPException(status_code=401, detail=f"Invalid or expired token: {e}")
    except Exception:
        raise HTTPException(status_code=401, detail="Error decoding token")


def get_agent_tier_from_roles(roles: List[str]) -> str:
    rs = {x.lower() for x in roles or []}
    if "admin" in rs or "advanced" in rs:
        return Tier.PRO.value
    if "pro" in rs:
        return Tier.PRO.value
    return Tier.BASIC.value
