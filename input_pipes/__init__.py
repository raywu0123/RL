from functools import partial

from .consecutive_frames import ConsecutiveFramesPipe
from .identity import IdentityPipe


input_pipe_hub = {
    'identity': IdentityPipe,
    'consecutive_frames_3': partial(ConsecutiveFramesPipe, n=3),
}
