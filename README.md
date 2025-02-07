# Difference Rewards Policy Gradients
This codebase accompanies paper "**Difference Rewards Policy Gradients**" ([Link](https://link.springer.com/article/10.1007/s00521-022-07960-5)) and the two proposed algorithms, Dr.Reinforce and Dr.ReinforceR. The implementation is based on [PyMARL](https://github.com/oxwhirl/pymarl) and [SMAC](https://github.com/oxwhirl/smac) codebases which are open-sourced.

PyMARL is [WhiRL](http://whirl.cs.ox.ac.uk)'s framework for deep multi-agent reinforcement learning and includes implementations of the following algorithms:
- [**QMIX**: QMIX: Monotonic Value Function Factorisation for Deep Multi-Agent Reinforcement Learning](https://arxiv.org/abs/1803.11485)
- [**COMA**: Counterfactual Multi-Agent Policy Gradients](https://arxiv.org/abs/1705.08926)
- [**VDN**: Value-Decomposition Networks For Cooperative Multi-Agent Learning](https://arxiv.org/abs/1706.05296) 
- [**IQL**: Independent Q-Learning](https://arxiv.org/abs/1511.08779)
- [**QTRAN**: QTRAN: Learning to Factorize with Transformation for Cooperative Multi-Agent Reinforcement Learning](https://arxiv.org/abs/1905.05408)

PyMARL is written in PyTorch and uses [SMAC](https://github.com/oxwhirl/smac) as its environment.

#### Additional Algorithms

- [**PG**: Learning to Cooperate via Policy Search](https://arxiv.org/abs/cs/0105032)
- **CentralQ**: A2C w/ Centralized Q-Function Critic
- [**Colby**: Approximating Difference Evaluations with Local Knowledge](https://dl.acm.org/doi/abs/10.5555/2615731.2616070)
- [**Dr.Reinforce**: Difference Rewards Policy Gradients w/ True Rewards (Our)](https://link.springer.com/article/10.1007/s00521-022-07960-5)
- [**Dr.ReinforceR**: Difference Rewards Policy Gradients w/ Centralized Reward Network (Our)](https://link.springer.com/article/10.1007/s00521-022-07960-5)

## Installation instructions

Build the Dockerfile using 
```shell
cd docker
bash build.sh
```

Set up StarCraft II and SMAC:
```shell
bash install_sc2.sh
```

This will download SC2 into the 3rdparty folder and copy the maps necessary to run over.

The requirements.txt file can be used to install the necessary packages into a virtual environment (not recomended).

## Run an experiment 

```shell
python3 src/main.py --config=qmix --env-config=sc2 with env_args.map_name=2s3z
```

The config files act as defaults for an algorithm or environment. 

They are all located in `src/config`.
`--config` refers to the config files in `src/config/algs`
`--env-config` refers to the config files in `src/config/envs`

To run experiments using the Docker container:
```shell
bash run.sh $GPU python3 src/main.py --config=qmix --env-config=sc2 with env_args.map_name=2s3z
```

All results will be stored in the `Results` folder.

The previous config files used for the SMAC Beta have the suffix `_beta`.

## Saving and loading learnt models

### Saving models

You can save the learnt models to disk by setting `save_model = True`, which is set to `False` by default. The frequency of saving models can be adjusted using `save_model_interval` configuration. Models will be saved in the result directory, under the folder called *models*. The directory corresponding each run will contain models saved throughout the experiment, each within a folder corresponding to the number of timesteps passed since starting the learning process.

### Loading models

Learnt models can be loaded using the `checkpoint_path` parameter, after which the learning will proceed from the corresponding timestep. 

## Watching StarCraft II replays

`save_replay` option allows saving replays of models which are loaded using `checkpoint_path`. Once the model is successfully loaded, `test_nepisode` number of episodes are run on the test mode and a .SC2Replay file is saved in the Replay directory of StarCraft II. Please make sure to use the episode runner if you wish to save a replay, i.e., `runner=episode`. The name of the saved replay file starts with the given `env_args.save_replay_prefix` (map_name if empty), followed by the current timestamp. 

The saved replays can be watched by double-clicking on them or using the following command:

```shell
python -m pysc2.bin.play --norender --rgb_minimap_size 0 --replay NAME.SC2Replay
```

**Note:** Replays cannot be watched using the Linux version of StarCraft II. Please use either the Mac or Windows version of the StarCraft II client.

## Documentation/Support

Please raise an issue in this repo, or email [Jacopo](J.Castellini@liverpool.ac.uk)

## Citing this repository

If you use this repository in your research, please cite the [Difference Rewards Policy Gradients paper](https://link.springer.com/article/10.1007/s00521-022-07960-5).

*J. Castellini, S. Devlin, F. A. Oliehoek, and R. Savani. Difference Rewards Policy Gradients. Neural Computing & Applications (Special Issue on Adaptive and Learning Agents 2021), 2022. https://doi.org/10.1007/s00521-022-07960-5*

In BibTeX format:

```tex
@article{castellini2022drpg,
  author = {Castellini, Jacopo and Devlin, Sam and Oliehoek, Frans A. and Savani, Rahul},
  title = {Difference Rewards Policy Gradients},
  journal = {Neural Computing and Applications},
  volume = {Special Issue on Adaptive and Learning Agents 2021},
  numpages = {24},
  publisher = {Springer Nature},
  year = {2022}
}
```

Please consider also citing the [PyMARL](https://github.com/oxwhirl/pymarl) repository, on which the current one is heavily based upon.

## License

Code licensed under the Apache License v2.0
