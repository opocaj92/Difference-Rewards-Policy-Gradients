import torch as th
import torch.nn as nn
import torch.nn.functional as F


class IndependentRewardNetwork(nn.Module):
    def __init__(self, scheme, args):
        super(IndependentRewardNetwork, self).__init__()

        self.args = args
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents

        input_shape = self._get_input_shape(scheme)
        self.output_type = "r"

        # Set up network layers
        self.fc1 = nn.Linear(input_shape, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, batch, t=None):
        bs = batch.batch_size
        max_t = batch.max_seq_length if t is None else 1

        inputs = self._build_inputs(batch, t=t)
        x = F.relu(self.fc1(inputs))
        x = F.relu(self.fc2(x))
        q = self.fc3(x)
        return q

    def _build_inputs(self, batch, t=None):
        bs = batch.batch_size
        max_t = batch.max_seq_length if t is None else 1
        ts = slice(None) if t is None else slice(t, t+1)
        inputs = []

        # observation
        inputs.append(batch["obs"][:, ts])

        # actions
        inputs.append(batch["actions_onehot"][:, ts])

        inputs.append(th.eye(self.n_agents, device=batch.device).unsqueeze(0).unsqueeze(0).expand(bs, max_t, -1, -1))

        inputs = th.cat([x.reshape(bs, max_t, self.n_agents, -1) for x in inputs], dim=-1)
        
        if self.args.reward_baseline_fn == "au":
            inputs = inputs.repeat(1, 1, self.n_actions, 1)

            # actions (masked out by agent)
            inputs[:, :, :, -2 * self.n_actions:-self.n_actions] = th.eye(self.n_actions, device=batch.device).repeat(self.n_agents, 1).unsqueeze(0).unsqueeze(0)
        return inputs

    def _get_input_shape(self, scheme):
        # observation
        input_shape = scheme["obs"]["vshape"]
        # actions
        input_shape += scheme["actions_onehot"]["vshape"][0]
        # agent id
        input_shape += self.n_agents
        return input_shape
