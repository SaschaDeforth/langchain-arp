"""
Agentic Reasoning Protocol (ARP) — Ed25519 Signature Verification
=================================================================

Verifies the cryptographic signature of a reasoning.json file by:
1. Fetching the public key from DNS TXT record
2. Reconstructing the canonical payload (JCS RFC 8785)
3. Validating the Ed25519 signature

This completes the cryptographic trust loop between the browser-based
ARP Signer and downstream AI agent ingestion.

Requirements:
    pip install cryptography dnspython jcs

Usage:
    from langchain_arp.verify import verify_arp_signature

    import json
    payload = json.load(open("reasoning.json"))
    result = verify_arp_signature(payload)
    print(result)  # {"verified": True, "dns_record": "arp._arp.example.com", ...}

License: MIT
Author: Sascha Deforth / TrueSource
Spec: https://arp-protocol.org
"""

from __future__ import annotations

import base64
import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)


def verify_arp_signature(payload: dict) -> Dict[str, Any]:
    """
    Verify the Ed25519 cryptographic signature of a reasoning.json payload.

    Uses the enveloped signature pattern: the _arp_signature metadata
    (with signature="") is included in the signed canonical bytes.

    Args:
        payload: Parsed reasoning.json as a Python dict

    Returns:
        Dict with verification result:
        {
            "verified": bool,
            "dns_record": str,
            "signed_at": str,
            "expires_at": str,
            "error": str | None
        }
    """
    sig_data = payload.get("_arp_signature")
    if not sig_data:
        return {
            "verified": False,
            "dns_record": "none",
            "signed_at": "none",
            "expires_at": "none",
            "error": "No _arp_signature block found",
        }

    dns_record = sig_data.get("dns_record", "unknown")
    signed_at = sig_data.get("signed_at", "unknown")
    expires_at = sig_data.get("expires_at", "unknown")

    try:
        # Lazy imports — these are optional dependencies
        try:
            import jcs  # RFC 8785 JSON Canonicalization Scheme
        except ImportError:
            raise ImportError(
                "The 'jcs' package is required for signature verification. "
                "Install it with: pip install jcs"
            )

        try:
            import dns.resolver
        except ImportError:
            raise ImportError(
                "The 'dnspython' package is required for DNS key lookup. "
                "Install it with: pip install dnspython"
            )

        try:
            from cryptography.hazmat.primitives.asymmetric.ed25519 import (
                Ed25519PublicKey,
            )
        except ImportError:
            raise ImportError(
                "The 'cryptography' package is required for signature verification. "
                "Install it with: pip install cryptography"
            )

        # 1. Fetch Public Key from DNS TXT record
        logger.info(f"Resolving DNS TXT record: {dns_record}")
        answers = dns.resolver.resolve(dns_record, "TXT")
        txt_records = [b.decode("utf-8") for r in answers for b in r.strings]
        arp_txt = next(
            (t for t in txt_records if t.startswith("v=ARP1")), None
        )

        if not arp_txt:
            return {
                "verified": False,
                "dns_record": dns_record,
                "signed_at": signed_at,
                "expires_at": expires_at,
                "error": f"No ARP TXT record found at {dns_record}",
            }

        # Extract Base64 public key from "v=ARP1; k=ed25519; p=..."
        parts_dict = {}
        for item in arp_txt.split(";"):
            item = item.strip()
            if "=" in item:
                k, v = item.split("=", 1)
                parts_dict[k.strip()] = v.strip()

        pubkey_b64 = parts_dict.get("p")
        if not pubkey_b64:
            return {
                "verified": False,
                "dns_record": dns_record,
                "signed_at": signed_at,
                "expires_at": expires_at,
                "error": "No public key (p=) found in DNS TXT record",
            }

        public_key = Ed25519PublicKey.from_public_bytes(
            base64.b64decode(pubkey_b64)
        )

        # 2. Reconstruct canonical payload (enveloped signature pattern)
        import copy

        payload_copy = copy.deepcopy(payload)
        payload_copy["_arp_signature"]["signature"] = ""  # Empty for canonicalization
        canonical_bytes = jcs.canonicalize(payload_copy)

        # 3. Decode and verify the Ed25519 signature
        sig_b64url = sig_data["signature"]
        # Fix Base64URL padding if necessary
        sig_b64url += "=" * ((4 - len(sig_b64url) % 4) % 4)
        signature_bytes = base64.urlsafe_b64decode(sig_b64url)

        public_key.verify(signature_bytes, canonical_bytes)

        logger.info(f"✅ Signature verified for {dns_record}")
        return {
            "verified": True,
            "dns_record": dns_record,
            "signed_at": signed_at,
            "expires_at": expires_at,
            "error": None,
        }

    except ImportError:
        raise  # Re-raise import errors so developers know what to install

    except Exception as e:
        logger.warning(f"Signature verification failed: {e}")
        return {
            "verified": False,
            "dns_record": dns_record,
            "signed_at": signed_at,
            "expires_at": expires_at,
            "error": str(e),
        }


# ─── CLI ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import json
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m langchain_arp.verify <reasoning.json>")
        print("Example: python -m langchain_arp.verify ./reasoning.json")
        sys.exit(1)

    file_path = sys.argv[1]
    print(f"\n🔐 Verifying signature for {file_path}...\n")

    try:
        with open(file_path, "r") as f:
            data = json.load(f)

        result = verify_arp_signature(data)

        if result["verified"]:
            print(f"✅ VERIFIED — Signature is valid")
        else:
            print(f"❌ FAILED — {result['error']}")

        print(f"   DNS Record: {result['dns_record']}")
        print(f"   Signed At:  {result['signed_at']}")
        print(f"   Expires At: {result['expires_at']}")

    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)
