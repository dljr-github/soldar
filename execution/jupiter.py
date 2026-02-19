"""Jupiter V6 API integration for Solana token swaps."""

from __future__ import annotations

import base64
import logging
from dataclasses import dataclass

import requests

from execution.wallet import SolanaWallet

log = logging.getLogger("execution.jupiter")

BASE_URL = "https://quote-api.jup.ag/v6"

SOL_MINT = "So11111111111111111111111111111111111111112"
USDC_MINT = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"


@dataclass
class SwapResult:
    success: bool
    input_amount: float
    output_amount: float
    price: float
    signature: str
    error: str


def get_quote(
    input_mint: str,
    output_mint: str,
    amount_lamports: int,
    slippage_bps: int = 1000,
) -> dict | None:
    """Get a swap quote from Jupiter V6."""
    params = {
        "inputMint": input_mint,
        "outputMint": output_mint,
        "amount": str(amount_lamports),
        "slippageBps": slippage_bps,
        "onlyDirectRoutes": "false",
    }
    try:
        resp = requests.get(f"{BASE_URL}/quote", params=params, timeout=15)
        resp.raise_for_status()
        quote = resp.json()
        log.info(
            "Quote: %s → %s | in=%s out=%s impact=%s",
            input_mint[:8], output_mint[:8],
            quote.get("inAmount"), quote.get("outAmount"),
            quote.get("priceImpactPct"),
        )
        return quote
    except requests.RequestException as exc:
        log.error("Jupiter quote failed: %s", exc)
        return None


def get_swap_transaction(
    quote: dict, user_pubkey: str, wrap_unwrap_sol: bool = True
) -> bytes | None:
    """Get a serialized swap transaction from Jupiter."""
    body = {
        "quoteResponse": quote,
        "userPublicKey": user_pubkey,
        "wrapAndUnwrapSol": wrap_unwrap_sol,
        "dynamicComputeUnitLimit": True,
        "prioritizationFeeLamports": "auto",
    }
    try:
        resp = requests.post(f"{BASE_URL}/swap", json=body, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        tx_b64 = data.get("swapTransaction")
        if not tx_b64:
            log.error("No swapTransaction in response")
            return None
        return base64.b64decode(tx_b64)
    except requests.RequestException as exc:
        log.error("Jupiter swap tx failed: %s", exc)
        return None


def execute_swap(
    wallet: SolanaWallet,
    rpc_url: str | None,
    input_mint: str,
    output_mint: str,
    amount_lamports: int,
    slippage_bps: int = 1000,
    dry_run: bool = False,
) -> SwapResult:
    """Full swap flow: quote → transaction → sign → send → confirm."""
    # Get quote
    quote = get_quote(input_mint, output_mint, amount_lamports, slippage_bps)
    if not quote:
        return SwapResult(False, 0, 0, 0, "", "Failed to get quote")

    in_amount = int(quote.get("inAmount", 0))
    out_amount = int(quote.get("outAmount", 0))
    price = out_amount / in_amount if in_amount else 0

    if dry_run:
        log.info(
            "[DRY RUN] Would swap %s %s → %s %s (price=%s)",
            in_amount, input_mint[:8], out_amount, output_mint[:8], price,
        )
        return SwapResult(True, in_amount, out_amount, price, "dry_run", "")

    if wallet.pubkey is None:
        return SwapResult(False, 0, 0, 0, "", "Wallet not loaded")

    # Get transaction
    tx_bytes = get_swap_transaction(quote, str(wallet.pubkey))
    if not tx_bytes:
        return SwapResult(False, in_amount, out_amount, price, "", "Failed to get swap tx")

    # Sign
    try:
        signed = wallet.sign_transaction(tx_bytes)
    except Exception as exc:
        return SwapResult(False, in_amount, out_amount, price, "", f"Sign failed: {exc}")

    # Send
    try:
        sig = wallet.send_transaction(rpc_url, signed)
    except Exception as exc:
        return SwapResult(False, in_amount, out_amount, price, "", f"Send failed: {exc}")

    # Confirm
    confirmed = wallet.confirm_transaction(rpc_url, sig)
    if not confirmed:
        return SwapResult(False, in_amount, out_amount, price, sig, "Confirmation timeout")

    log.info("Swap confirmed: %s", sig)
    return SwapResult(True, in_amount, out_amount, price, sig, "")
