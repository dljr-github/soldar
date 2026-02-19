"""Solana wallet management using solders."""

from __future__ import annotations

import time
import logging

import base58
import requests
from solders.keypair import Keypair
from solders.pubkey import Pubkey
from solders.transaction import VersionedTransaction
from dotenv import load_dotenv
import os

load_dotenv()

log = logging.getLogger("execution.wallet")


class SolanaWallet:
    """Manages a Solana keypair and provides signing/sending helpers."""

    def __init__(self) -> None:
        pk_b58 = os.getenv("WALLET_PRIVATE_KEY", "")
        if not pk_b58 or pk_b58 == "your_base58_private_key_here":
            log.warning("WALLET_PRIVATE_KEY not set — wallet in read-only mode")
            self.keypair: Keypair | None = None
        else:
            secret = base58.b58decode(pk_b58)
            self.keypair = Keypair.from_bytes(secret)
            log.info("Wallet loaded: %s", self.pubkey)

        self.rpc_url = os.getenv(
            "HELIUS_RPC_URL", "https://api.mainnet-beta.solana.com"
        )

    @property
    def pubkey(self) -> Pubkey | None:
        return self.keypair.pubkey() if self.keypair else None

    def _rpc_call(self, method: str, params: list) -> dict:
        """Make a JSON-RPC call to the Solana RPC endpoint."""
        payload = {"jsonrpc": "2.0", "id": 1, "method": method, "params": params}
        resp = requests.post(self.rpc_url, json=payload, timeout=30)
        resp.raise_for_status()
        return resp.json()

    def get_balance_sol(self, rpc_url: str | None = None) -> float:
        """Fetch SOL balance for the wallet."""
        if self.pubkey is None:
            return 0.0
        url = rpc_url or self.rpc_url
        payload = {
            "jsonrpc": "2.0", "id": 1,
            "method": "getBalance",
            "params": [str(self.pubkey)],
        }
        resp = requests.post(url, json=payload, timeout=30)
        resp.raise_for_status()
        result = resp.json().get("result", {})
        lamports = result.get("value", 0)
        return lamports / 1e9

    def get_token_balance(
        self, rpc_url: str | None, token_mint: str, wallet_pubkey: str | None = None
    ) -> float:
        """Fetch SPL token balance for a given mint."""
        url = rpc_url or self.rpc_url
        owner = wallet_pubkey or (str(self.pubkey) if self.pubkey else None)
        if not owner:
            return 0.0
        payload = {
            "jsonrpc": "2.0", "id": 1,
            "method": "getTokenAccountsByOwner",
            "params": [
                owner,
                {"mint": token_mint},
                {"encoding": "jsonParsed"},
            ],
        }
        resp = requests.post(url, json=payload, timeout=30)
        resp.raise_for_status()
        result = resp.json().get("result", {})
        accounts = result.get("value", [])
        total = 0.0
        for acct in accounts:
            info = (
                acct.get("account", {})
                .get("data", {})
                .get("parsed", {})
                .get("info", {})
                .get("tokenAmount", {})
            )
            total += float(info.get("uiAmount", 0) or 0)
        return total

    def sign_transaction(self, tx_bytes: bytes) -> bytes:
        """Sign a serialized versioned transaction."""
        if self.keypair is None:
            raise RuntimeError("Wallet not loaded — cannot sign")
        tx = VersionedTransaction.from_bytes(tx_bytes)
        signed = VersionedTransaction(tx.message, [self.keypair])
        return bytes(signed)

    def send_transaction(self, rpc_url: str | None, signed_tx_bytes: bytes) -> str:
        """Submit a signed transaction to the Solana RPC and return the signature."""
        import base64

        url = rpc_url or self.rpc_url
        tx_b64 = base64.b64encode(signed_tx_bytes).decode()
        payload = {
            "jsonrpc": "2.0", "id": 1,
            "method": "sendTransaction",
            "params": [
                tx_b64,
                {"encoding": "base64", "skipPreflight": False, "maxRetries": 3},
            ],
        }
        resp = requests.post(url, json=payload, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        if "error" in data:
            raise RuntimeError(f"RPC error: {data['error']}")
        return data["result"]

    def confirm_transaction(
        self, rpc_url: str | None, signature: str, timeout: int = 60
    ) -> bool:
        """Poll until the transaction is confirmed or timeout."""
        url = rpc_url or self.rpc_url
        deadline = time.time() + timeout
        while time.time() < deadline:
            payload = {
                "jsonrpc": "2.0", "id": 1,
                "method": "getSignatureStatuses",
                "params": [[signature], {"searchTransactionHistory": True}],
            }
            resp = requests.post(url, json=payload, timeout=30)
            resp.raise_for_status()
            statuses = resp.json().get("result", {}).get("value", [])
            if statuses and statuses[0]:
                status = statuses[0]
                if status.get("err"):
                    log.error("Transaction %s failed: %s", signature, status["err"])
                    return False
                conf = status.get("confirmationStatus")
                if conf in ("confirmed", "finalized"):
                    log.info("Transaction %s confirmed (%s)", signature, conf)
                    return True
            time.sleep(2)
        log.error("Transaction %s timed out after %ds", signature, timeout)
        return False
