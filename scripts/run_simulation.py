"""CLI entrypoint for running Active Learning simulations."""

import argparse
import sys
import os
import torch

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.loop.state import LoopConfig
from src.loop.orchestrator import ActiveLearningOrchestrator
from src.data.cifar10 import CIFAR10Adapter
from src.data.agnews import AGNewsAdapter
from src.models.cnn import CIFARConvNet, ResNet18TransferNet
from src.models.text_net import TextMLP
from src.models.wrapper import ActiveLearningClassifier
from src.strategies.registry import get_strategy
from src.labeling.oracle import SimulatedOracle


def parse_args():
    parser = argparse.ArgumentParser(description="Active Learning Simulation Runner")
    parser.add_argument("--config", type=str, default="configs/cifar10_simulation.yaml", help="Path to config YAML")
    parser.add_argument("--strategy", type=str, default=None, help="Override query strategy")
    parser.add_argument("--rounds", type=int, default=None, help="Override max rounds")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("--seed", type=int, default=None, help="Override random seed")
    parser.add_argument("--synthetic", action="store_true", help="Force synthetic data for instant CPU test")
    return parser.parse_args()


def main():
    args = parse_args()
    if os.path.exists(args.config):
        config = LoopConfig.from_yaml(args.config)
    else:
        config = LoopConfig()

    # Overrides
    if args.strategy:
        config.strategy = args.strategy
    if args.rounds:
        config.max_rounds = args.rounds
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.seed:
        config.seed = args.seed
    if args.synthetic:
        config.use_synthetic = True

    # Deterministic seeding
    torch.manual_seed(config.seed)

    # 1. Dataset Adapter
    if config.dataset.lower() == "cifar10":
        dataset = CIFAR10Adapter(
            max_samples=config.max_samples,
            use_synthetic=config.use_synthetic,
            seed=config.seed,
        )
        if config.model_type == "resnet18":
            module_cls = lambda: ResNet18TransferNet(num_classes=dataset.num_classes)
        else:
            module_cls = lambda: CIFARConvNet(num_classes=dataset.num_classes)
    elif config.dataset.lower() == "agnews":
        dataset = AGNewsAdapter(
            max_samples=config.max_samples,
            use_synthetic=config.use_synthetic,
            seed=config.seed,
        )
        module_cls = lambda: TextMLP(in_features=64, num_classes=dataset.num_classes)
    else:
        raise ValueError(f"Unknown dataset: {config.dataset}")

    # 2. Model Wrapper
    model = ActiveLearningClassifier(
        module=module_cls,
        max_epochs=config.max_epochs,
        lr=config.lr,
        warm_start=config.warm_start,
        device="cpu",
        batch_size=32,
        verbose=0,
    )

    # 3. Query Strategy
    strategy = get_strategy(config.strategy, seed=config.seed)

    # 4. Simulated Oracle
    oracle = SimulatedOracle(dataset=dataset, noise_rate=config.label_noise_rate, seed=config.seed)

    # 5. Orchestrator
    orchestrator = ActiveLearningOrchestrator(
        config=config,
        dataset=dataset,
        model=model,
        strategy=strategy,
        labeler=oracle,
    )

    # Run Simulation
    tracker = orchestrator.run_simulation()

    print("\n--- FINAL SIMULATION LEARNING CURVE ---")
    print(tracker.to_markdown())
    print("\nSimulation complete!")


if __name__ == "__main__":
    main()
