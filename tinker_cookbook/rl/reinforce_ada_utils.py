"""
Utility function for Reinforce-Ada training.
"""
from collections import defaultdict
from typing import Dict, List


class RewardHistory:
    """Tracks rewards for each prompt to compute global advantages."""
    
    def __init__(self):
        # Maps prompt hash to list of rewards
        self.prompt_rewards: Dict[str, List[float]] = defaultdict(list)
    
    def add_rewards(self, prompt_id: str, rewards: List[float]):
        """Add new rewards for a prompt."""
        self.prompt_rewards[prompt_id].extend(rewards)
    
    def get_mean(self, prompt_id: str) -> float:
        """Get the global mean reward for a prompt."""
        rewards = self.prompt_rewards[prompt_id]
        if not rewards:
            return 0.0
        return sum(rewards) / len(rewards)
    
    def get_count(self, prompt_id: str) -> int:
        """Get the number of rewards collected for a prompt."""
        return len(self.prompt_rewards[prompt_id])
    
    def clear(self):
        """Clear all history."""
        self.prompt_rewards.clear()