from .dqn import DQNAgent
from .future_reality import FutureRealityAgent


agent_hub = {
    'dqn': DQNAgent,
    'future_reality': FutureRealityAgent,
}
