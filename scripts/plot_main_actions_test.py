#!/usr/bin/env python3
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from config.hyperparameters import HyperParameters
from models.main_dqn.enviroment import MainDQNEnvironment
from models.main_dqn.model import build_main_dqn


def build_pg_model(input_dim: int = 2, action_dim: int = 3) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(input_dim,))
    x = tf.keras.layers.Dense(64, activation="relu")(inputs)
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    logits = tf.keras.layers.Dense(action_dim, activation="linear", name="policy_logits")(x)
    value = tf.keras.layers.Dense(1, activation="linear", name="state_value")(x)
    return tf.keras.Model(inputs=inputs, outputs=[logits, value])


def infer_actions_test(dataset_path: str, weights_path: str, alpha: float, omega: int, variant_id: str):
    hp = HyperParameters()
    df = pd.read_csv(dataset_path, parse_dates=["timestamp"])
    test_df = df[df["split"] == "test"].copy().sort_values("timestamp")
    if test_df.empty:
        raise RuntimeError("No hay filas con split=test en el dataset.")

    env = MainDQNEnvironment(
        x1=test_df["x1"].values,
        x2=test_df["x2"].values,
        prices=test_df["ap_t"].values,
        initial_investment=hp.INITIAL_INVESTMENT,
        transaction_fee=hp.TRANSACTION_FEE,
        alpha=alpha,
        omega=omega,
        reward_mode="paper",
    )

    if variant_id == "policy_gradient":
        model = build_pg_model(input_dim=2, action_dim=3)
        model.load_weights(weights_path)

        def choose_action(state: np.ndarray) -> int:
            logits, _ = model(state[None, :], training=False)
            return int(tf.argmax(logits[0]).numpy())
    else:
        model = build_main_dqn(variant_id=variant_id)
        model.load_weights(weights_path)

        def choose_action(state: np.ndarray) -> int:
            q = model(state[None, :], training=False).numpy()[0]
            return int(np.argmax(q))

    state = env.reset()
    done = False
    actions = []
    portfolio = []
    rewards = []

    while not done:
        action = choose_action(state)
        actions.append(action)
        next_state, reward, done, info = env.step(action)
        portfolio.append(float(info["portfolio_value"]))
        rewards.append(float(reward))
        state = next_state

    # Align timestamps/prices to action count (env stops at len-1)
    n = len(actions)
    out = test_df.iloc[:n].copy()
    out["main_action"] = actions
    out["portfolio_value"] = portfolio
    out["reward"] = rewards
    return out


def plot_actions(df: pd.DataFrame, out_png: str, title: str):
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)

    action_label = {0: "SELL", 1: "HOLD", 2: "BUY"}
    action_color = {0: "#2ca02c", 1: "#7f7f7f", 2: "#d62728"}

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True, gridspec_kw={"height_ratios": [2, 1]})

    axes[0].plot(df["timestamp"], df["ap_t"], color="royalblue", linewidth=1.0, label="BTC Price (test)")
    for a in [0, 1, 2]:
        sub = df[df["main_action"] == a]
        if not sub.empty:
            axes[0].scatter(sub["timestamp"], sub["ap_t"], s=8, alpha=0.7, c=action_color[a], label=action_label[a])
    axes[0].set_ylabel("Price")
    axes[0].set_title(title)
    axes[0].legend(loc="upper left")

    axes[1].plot(df["timestamp"], df["main_action"], color="black", linewidth=0.8)
    axes[1].set_yticks([0, 1, 2])
    axes[1].set_yticklabels(["SELL", "HOLD", "BUY"])
    axes[1].set_ylabel("Main action")
    axes[1].set_xlabel("Time (test)")

    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, help="Ruta a integrated_dataset*.csv")
    p.add_argument("--weights", required=True, help="Ruta a checkpoint .weights.h5 del Main")
    p.add_argument("--alpha", type=float, required=True)
    p.add_argument("--omega", type=int, required=True)
    p.add_argument("--variant-id", default="paper", help="paper | policy_gradient | dueling_double")
    p.add_argument("--out-csv", required=True)
    p.add_argument("--out-png", required=True)
    p.add_argument("--title", default="Main Actions Over Test Time")
    args = p.parse_args()

    out_df = infer_actions_test(
        dataset_path=args.dataset,
        weights_path=args.weights,
        alpha=args.alpha,
        omega=args.omega,
        variant_id=args.variant_id,
    )
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out_csv, index=False)
    plot_actions(out_df, args.out_png, args.title)

    counts = out_df["main_action"].value_counts().sort_index()
    print(f"saved_csv={args.out_csv}")
    print(f"saved_png={args.out_png}")
    print("action_counts:", {int(k): int(v) for k, v in counts.items()})


if __name__ == "__main__":
    main()
