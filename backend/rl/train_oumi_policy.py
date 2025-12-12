from datasets import Dataset
from oumi import train
from oumi.core.configs import TrainingConfig

# Tiny synthetic dataset based on your incident logic
data = [
    {"prompt": "fight confidence:0.9 status:high_risk", "action": "ALERT_POLICE"},
    {"prompt": "theft confidence:0.85 status:high_risk", "action": "ALERT_POLICE"},
    {"prompt": "normal confidence:0.7 status:low_risk", "action": "MONITOR"},
    {"prompt": "normal confidence:0.4 status:low_risk", "action": "MONITOR"},
]

dataset = Dataset.from_list(data)

config = TrainingConfig(
    method="grpo",
    dataset=dataset,
    # keep other fields default for this demo
)

train(config)
print("✅ Oumi GRPO training demo completed.")
