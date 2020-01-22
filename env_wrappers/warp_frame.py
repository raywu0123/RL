import gym
from gym import spaces
import cv2


class WarpFrameWrapper(gym.ObservationWrapper):

    def __init__(self, env):
        """Warp frames to 84x84 as done in the Nature paper and later work."""
        super().__init__(env)
        self.width = 84
        self.height = 84
        self.observation_space = spaces.Box(low=0, high=255, shape=(self.height, self.width, 1))

    def observation(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
        return frame[:, :, None]
