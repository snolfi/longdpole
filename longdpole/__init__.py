import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)

register(
    id='LongdpoleEnv-v0',
    entry_point='longdpole.envs:LongdpoleEnv',
)

register(
    id='LongdpoleEnv-v1',
    entry_point='longdpole.envs:LongdpoleEnv1',
)

register(
    id='LongdpoleEnv-v2',
    entry_point='longdpole.envs:LongdpoleEnv2',
)

