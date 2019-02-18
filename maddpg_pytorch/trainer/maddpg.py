
from maddpg_pytorch import AgentTrainer



class MADDPGAgentTrainer(AgentTrainer):

    def __init__(self, name, model, obs_shape_n, act_space_n, agent_index, args, local_q_func=False):

        AgentTrainer.__init__(self, name, model, obs_shape_n, act_space_n)

        raise NotImplementedError