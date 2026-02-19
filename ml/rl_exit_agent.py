"""
PPO RL exit agent trainer and evaluator.
Usage:
  python ml/rl_exit_agent.py --train              # train from scratch
  python ml/rl_exit_agent.py --train --resume     # resume training
  python ml/rl_exit_agent.py --eval               # evaluate saved policy
  python ml/rl_exit_agent.py --demo               # run demo episode
  python ml/rl_exit_agent.py --demo --n 5         # run 5 demo episodes
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys

import numpy as np

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path for ml.* imports
# ---------------------------------------------------------------------------
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# Models directory
MODELS_DIR = os.path.join(_ROOT, "ml", "models")
MODEL_PATH = os.path.join(MODELS_DIR, "ppo_exit_agent")
LOGS_DIR = os.path.join(MODELS_DIR, "rl_logs")
CHECKPOINTS_DIR = os.path.join(MODELS_DIR, "checkpoints")
TB_DIR = os.path.join(MODELS_DIR, "rl_tensorboard")


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(resume: bool = False, total_timesteps: int = 500_000) -> None:
    """Train the PPO exit agent."""
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
    from stable_baselines3.common.env_util import make_vec_env

    from ml.augment import SlippageAugmenter
    from ml.rl_data import load_sequences
    from ml.rl_env import MemeExitEnv

    sequences = load_sequences()
    if not sequences:
        print("ERROR: No sequences loaded. Cannot train.")
        sys.exit(1)

    print(f"Loaded {len(sequences)} sequences (avg length: {np.mean([len(s) for s in sequences]):.0f} steps)")

    augmenter = SlippageAugmenter()

    # Train/eval split
    split = int(0.85 * len(sequences))
    train_seqs = sequences[:split]
    eval_seqs = sequences[split:]
    print(f"Split: {len(train_seqs)} train / {len(eval_seqs)} eval")

    def make_train_env():
        return MemeExitEnv(train_seqs, augmenter=augmenter)

    env = make_vec_env(make_train_env, n_envs=4)
    eval_env = MemeExitEnv(eval_seqs)

    # Ensure directories exist
    for d in [MODELS_DIR, LOGS_DIR, CHECKPOINTS_DIR, TB_DIR]:
        os.makedirs(d, exist_ok=True)

    callbacks = [
        EvalCallback(
            eval_env,
            best_model_save_path=MODELS_DIR,
            log_path=LOGS_DIR,
            eval_freq=10_000,
            n_eval_episodes=50,
            deterministic=True,
            verbose=1,
        ),
        CheckpointCallback(
            save_freq=50_000,
            save_path=CHECKPOINTS_DIR,
            name_prefix="ppo_exit",
        ),
    ]

    if resume and os.path.exists(MODEL_PATH + ".zip"):
        print(f"Resuming from {MODEL_PATH}.zip")
        model = PPO.load(MODEL_PATH, env=env)
        model.set_env(env)
    else:
        if resume:
            print("WARNING: --resume specified but no saved model found. Training from scratch.")
        import torch as _torch
        _device = "cuda" if _torch.cuda.is_available() else "cpu"
        print(f"[RL] Training on device: {_device}")
        model = PPO(
            "MlpPolicy",
            env,
            policy_kwargs=dict(net_arch=[128, 128]),
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            verbose=1,
            tensorboard_log=TB_DIR,
            device=_device,
        )

    print(f"Training for {total_timesteps:,} timesteps...")
    model.learn(
        total_timesteps=total_timesteps,
        callback=callbacks,
        reset_num_timesteps=not resume,
    )
    model.save(MODEL_PATH)
    print(f"Training complete. Model saved to {MODEL_PATH}.zip")


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(n_episodes: int = 200) -> dict:
    """Compare RL policy vs baselines on held-out sequences."""
    from stable_baselines3 import PPO

    from ml.rl_data import load_sequences
    from ml.rl_env import MemeExitEnv

    if not os.path.exists(MODEL_PATH + ".zip"):
        print(f"ERROR: No trained model at {MODEL_PATH}.zip — run --train first.")
        sys.exit(1)

    model = PPO.load(MODEL_PATH)
    sequences = load_sequences()
    eval_seqs = sequences[int(0.85 * len(sequences)):]

    if not eval_seqs:
        print("ERROR: No eval sequences available.")
        sys.exit(1)

    env = MemeExitEnv(eval_seqs)

    strategies: dict[str, object] = {
        "rl_agent": lambda obs, e: int(model.predict(obs, deterministic=True)[0]),
        "hold_120m": lambda obs, e: 0,
        "exit_at_2x": lambda obs, e: 3 if obs[0] >= 2.0 else 0,
        "exit_at_30m": lambda obs, e: 3 if e.time_held >= 30 else 0,
        "panic_sell": lambda obs, e: 3 if obs[0] < 0.85 else 0,
    }

    results = {}
    for name, policy_fn in strategies.items():
        pnls = []
        for _ in range(n_episodes):
            obs, _ = env.reset()
            done = False
            while not done:
                action = policy_fn(obs, env)
                obs, _, done, _, info = env.step(action)
            pnls.append(info["realized_pnl"])

        results[name] = {
            "mean_pnl": float(np.mean(pnls)),
            "median_pnl": float(np.median(pnls)),
            "win_rate": float(np.mean([p > 0 for p in pnls])),
            "pnl_std": float(np.std(pnls)),
        }

    # Save results
    os.makedirs(MODELS_DIR, exist_ok=True)
    results_path = os.path.join(MODELS_DIR, "ppo_eval_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*50}")
    print("  RL Agent Evaluation — Strategy Comparison")
    print(f"{'='*50}")
    print(f"{'Strategy':15s}  {'Mean PnL':>10s}  {'Win Rate':>10s}  {'Std':>8s}  {'Median':>10s}")
    print(f"{'-'*55}")
    for name, r in results.items():
        print(
            f"{name:15s}  {r['mean_pnl']:+10.4f}  {r['win_rate']:9.1%}  "
            f"{r['pnl_std']:8.4f}  {r['median_pnl']:+10.4f}"
        )
    print(f"\nResults saved to {results_path}")
    return results


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def demo(n_episodes: int = 3) -> None:
    """Run demo episodes showing step-by-step decisions."""
    from stable_baselines3 import PPO

    from ml.rl_data import load_sequences
    from ml.rl_env import MemeExitEnv

    if not os.path.exists(MODEL_PATH + ".zip"):
        print(f"ERROR: No trained model at {MODEL_PATH}.zip — run --train first.")
        sys.exit(1)

    model = PPO.load(MODEL_PATH)
    sequences = load_sequences()
    env = MemeExitEnv(sequences, max_hold_minutes=120)

    action_names = {0: "HOLD", 1: "SELL 25%", 2: "SELL 50%", 3: "SELL 100%"}

    for ep in range(n_episodes):
        obs, _ = env.reset()
        entry_price = env.entry_price
        print(f"\n{'='*60}")
        print(f"  Episode {ep + 1}/{n_episodes} | Entry: {entry_price:.8f}")
        print(f"{'='*60}")
        done = False
        step = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            action = int(action)

            price = env.seq[min(env.step_idx, len(env.seq) - 1)].get("price", entry_price)
            pct = (price / entry_price - 1) * 100

            obs, reward, done, _, info = env.step(action)

            # Print sell actions always; hold actions every 5 steps
            if action != 0 or step % 5 == 0:
                pos_bar = "#" * int(env.position_pct * 20)
                print(
                    f"  Min {step:3d}: price {pct:+7.1f}% | "
                    f"pos [{pos_bar:<20s}] {env.position_pct:.0%} | "
                    f"action={action_names[action]}"
                )
            step += 1

        pnl = info["realized_pnl"]
        emoji = "+" if pnl > 0 else ""
        print(f"\n  >> Final PnL: {emoji}{pnl:.4f} ({emoji}{pnl * 100:.1f}%)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="PPO RL exit agent — train, evaluate, or demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--train", action="store_true", help="Train the PPO agent")
    parser.add_argument("--resume", action="store_true", help="Resume training from saved model")
    parser.add_argument("--eval", action="store_true", help="Evaluate saved policy vs baselines")
    parser.add_argument("--demo", action="store_true", help="Run demo episodes")
    parser.add_argument("--n", type=int, default=None, help="Number of episodes (eval: default 200, demo: default 3)")
    parser.add_argument("--timesteps", type=int, default=500_000, help="Total training timesteps (default: 500k)")

    args = parser.parse_args()

    if not any([args.train, args.eval, args.demo]):
        parser.print_help()
        sys.exit(1)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    if args.train:
        train(resume=args.resume, total_timesteps=args.timesteps)

    if args.eval:
        n = args.n if args.n is not None else 200
        evaluate(n_episodes=n)

    if args.demo:
        n = args.n if args.n is not None else 3
        demo(n_episodes=n)


if __name__ == "__main__":
    main()
