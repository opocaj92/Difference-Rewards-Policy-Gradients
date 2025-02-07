from components.episode_buffer import EpisodeBatch
from modules.rewards.centralized_r import CentralizedRewardNetwork
from utils.rl_utils import discount_rewards
import torch as th
from torch.optim import RMSprop


class DrReinforceRLearner:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.n_agents = args.n_agents
        self.n_actions = args.n_actions
        self.mac = mac
        self.logger = logger

        self.log_stats_t = -self.args.learner_log_interval - 1

        if self.args.reward_fn == "centralized_r":
            self.reward = CentralizedRewardNetwork(scheme, args)
            self.reward_params = list(self.reward.parameters())
            self.reward_optimiser = RMSprop(params=self.reward_params, lr=args.reward_lr, alpha=args.optim_alpha, eps=args.optim_eps)
        elif self.args.reward_fn == "true":
            self.reward_params = []
        else:
            raise ValueError("Reward {} not recognised.".format(args.reward_fn))
        
        self.agent_params = list(mac.parameters())
        self.params = self.agent_params + self.reward_params

        self.agent_optimiser = RMSprop(params=self.agent_params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps)

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        # Get the relevant quantities
        bs = batch.batch_size
        max_t = batch.max_seq_length
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"][:, :-1]

        reward_mask = mask.clone()

        mask = mask.repeat(1, 1, self.n_agents).view(-1)

        if self.args.reward_fn != "true":
            r_vals, reward_train_stats = self._train_reward(batch, rewards, terminated, actions, avail_actions,
                                                        reward_mask, bs, max_t)
        else:
            r_vals = batch["all_rewards"][:, :-1]

        actions = actions[:, :-1]

        mac_out = []
        self.mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length - 1):
            agent_outs = self.mac.forward(batch, t=t)
            mac_out.append(agent_outs)
        mac_out = th.stack(mac_out, dim=1)  # Concat over time

        # Mask out unavailable actions, renormalise (as in action selection)
        mac_out[avail_actions == 0] = 0
        mac_out = mac_out/mac_out.sum(dim=-1, keepdim=True)
        mac_out[avail_actions == 0] = 0

        # Compute difference returns
        r_vals = r_vals.reshape(-1, self.n_actions)
        pi = mac_out.view(-1, self.n_actions)
        baseline = (pi * r_vals).sum(-1).reshape(bs, max_t - 1, self.n_agents).detach()
        returns = discount_rewards((rewards - baseline).detach(), terminated, reward_mask, self.n_agents, self.args.gamma).reshape(-1)

        # Calculate policy grad with mask
        pi_taken = th.gather(pi, dim=1, index=actions.reshape(-1, 1)).squeeze(1)
        pi_taken[mask == 0] = 1.0
        log_pi_taken = th.log(pi_taken)

        drpg_loss = - ((returns * log_pi_taken) * mask).sum() / mask.sum()

        # Optimise agents
        self.agent_optimiser.zero_grad()
        drpg_loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.agent_params, self.args.grad_norm_clip)
        self.agent_optimiser.step()


        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            if self.args.reward_fn != "true":
                ts_logged = len(reward_train_stats["reward_loss"])
                for key in ["reward_loss", "reward_grad_norm", "pred_error_abs"]:
                    self.logger.log_stat(key, sum(reward_train_stats[key])/ts_logged, t_env)

            self.logger.log_stat("difference_return_mean", (returns * mask).sum().item() / mask.sum().item(), t_env)
            self.logger.log_stat("drpg_loss", drpg_loss.item(), t_env)
            self.logger.log_stat("agent_grad_norm", grad_norm, t_env)
            self.logger.log_stat("pi_max", (pi.max(dim=1)[0] * mask).sum().item() / mask.sum().item(), t_env)
            self.log_stats_t = t_env

    def _train_reward(self, batch, rewards, terminated, actions, avail_actions, mask, bs, max_t):
        r_vals = th.zeros_like(rewards)[:, :].unsqueeze(2).repeat(1, 1, self.n_agents, self.n_actions)

        running_log = {
            "reward_loss": [],
            "reward_grad_norm": [],
            "pred_error_abs": [],
        }

        for t in reversed(range(rewards.size(1))):
            mask_t = mask[:, t]
            if mask_t.sum() == 0:
                continue

            r_t = self.reward(batch, t).reshape(bs, 1, self.n_agents, self.n_actions)
            r_vals[:, t] = r_t.view(bs, self.n_agents, self.n_actions)
            r_taken = th.gather(r_t, dim=3, index=actions[:, t:t+1]).squeeze(3)
            targets_t = rewards[:, t]

            pred_error = (r_taken.squeeze(1)[:, 0] - targets_t.detach())

            # 0-out the targets that came from padded data
            masked_pred_error = pred_error * mask_t

            # Normal MSE loss
            loss = (masked_pred_error ** 2).sum() / mask_t.sum()
            self.reward_optimiser.zero_grad()
            loss.backward()
            grad_norm = th.nn.utils.clip_grad_norm_(self.reward_params, self.args.grad_norm_clip)
            self.reward_optimiser.step()

            running_log["reward_loss"].append(loss.item())
            running_log["reward_grad_norm"].append(grad_norm)
            mask_elems = mask_t.sum().item()
            running_log["pred_error_abs"].append(masked_pred_error.abs().sum().item() / mask_elems)

        return r_vals, running_log

    def cuda(self):
        self.mac.cuda()
        if self.args.reward_fn != "true":
            self.reward.cuda()

    def save_models(self, path):
        self.mac.save_models(path)
        if self.args.reward_fn != "true":
            th.save(self.reward.state_dict(), "{}/reward.th".format(path))
        th.save(self.agent_optimiser.state_dict(), "{}/agent_opt.th".format(path))
        if self.args.reward_fn != "true":
            th.save(self.reward_optimiser.state_dict(), "{}/reward_opt.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        if self.args.reward_fn != "true":
            self.reward.load_state_dict(th.load("{}/reward.th".format(path), map_location=lambda storage, loc: storage))
        self.agent_optimiser.load_state_dict(th.load("{}/agent_opt.th".format(path), map_location=lambda storage, loc: storage))
        if self.args.reward_fn == "true":
            self.reward_optimiser.load_state_dict(th.load("{}/reward_opt.th".format(path), map_location=lambda storage, loc: storage))
