"""Learning curve recording and formatting utilities."""

from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional
import pandas as pd


@dataclass
class RoundMetric:
    round_num: int
    strategy: str
    n_labeled: int
    n_pool: int
    test_acc: float
    test_f1: float
    ece: Optional[float] = None
    temperature: Optional[float] = None


class LearningCurveTracker:
    """Tracks learning curve metrics across active learning rounds."""

    def __init__(self, strategy_name: str):
        self.strategy_name = strategy_name
        self.history: List[RoundMetric] = []

    def log_round(
        self,
        round_num: int,
        n_labeled: int,
        n_pool: int,
        test_acc: float,
        test_f1: float,
        ece: Optional[float] = None,
        temperature: Optional[float] = None,
    ) -> RoundMetric:
        metric = RoundMetric(
            round_num=round_num,
            strategy=self.strategy_name,
            n_labeled=n_labeled,
            n_pool=n_pool,
            test_acc=test_acc,
            test_f1=test_f1,
            ece=ece,
            temperature=temperature,
        )
        self.history.append(metric)
        return metric

    def to_dataframe(self) -> pd.DataFrame:
        """Convert history into pandas DataFrame."""
        return pd.DataFrame([asdict(m) for m in self.history])

    def to_markdown(self) -> str:
        """Render markdown table of learning curve history."""
        df = self.to_dataframe()
        return df.to_markdown(index=False)
