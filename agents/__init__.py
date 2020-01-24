from .dqn import DQNAgent
from .future_reality import FutureRealityAgent
from .base import BaseAgent


agent_hub = {
    'dqn': DQNAgent,
    'future_reality': FutureRealityAgent,
}
