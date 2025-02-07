import torch as th


def build_td_lambda_targets(rewards, terminated, mask, target_qs, n_agents, gamma, td_lambda):
    # Assumes  <target_qs > in B*T*A and <reward >, <terminated >, <mask > in (at least) B*T-1*1
    # Initialise  last  lambda -return  for  not  terminated  episodes
    ret = target_qs.new_zeros(*target_qs.shape)
    ret[:, -1] = target_qs[:, -1] * (1 - th.sum(terminated, dim=1))
    # Backwards  recursive  update  of the "forward  view"
    for t in range(ret.shape[1] - 2, -1,  -1):
        ret[:, t] = td_lambda * gamma * ret[:, t + 1] + mask[:, t] \
                    * (rewards[:, t] + (1 - td_lambda) * gamma * target_qs[:, t + 1] * (1 - terminated[:, t]))
    # Returns lambda-return from t=0 to t=T-1, i.e. in B*T-1*A
    return ret[:, 0:-1]


def discount_rewards(rewards, terminated, mask, n_agents, gamma):
    # Assumes  <reward >, <terminated > in (at least) B*T-1*1
    # Initialise  last  return  for  not  terminated  episodes
    ret = rewards.new_zeros(*rewards.shape)
    ret[:, -1] = rewards[:, -1]
    # Backwards  recursive  update  of the "forward  view"
    for t in range(ret.shape[1] - 2, -1,  -1):
        ret[:, t] = mask[:, t] * rewards[:, t] + gamma * ret[:, t + 1] * (1 - terminated[:, t])
    # Returns discounted rewards from t=0 to t=T-1, i.e. in B*T-1*A
    return ret