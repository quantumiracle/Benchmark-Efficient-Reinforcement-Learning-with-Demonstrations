# PPO (Tensorflow)

## Description

Implementations of PPO algorithm with Reacher environment, including PPO for Reacher of 2/3 joints, PPO with initialized policy, PPO+Reptile, PPO+MAML, PPO+FOMAML (first-order MAML), etc.

## Prerequisites

* Python 3.5
* Tensorflow

##  To Use:

`python ***.py --train/test`

## Contents

**Env:**

* `env.py`: Reacher environment with 3 joints.
* `env_2.py`: Reacher environment with 2 joints.

**Multi-thread PPO:**

* `ppo0.py`: original PPO code for Openai gym environments.
* `ppo.py`: PPO for Reacher environment.

**Single-thread PPO:**

* `ppo_single0.py`: original PPO code for Openai gym environments.
* `ppo_single.py`: PPO for Reacher environment. (3 joints)
* `ppo_single_2.py`: PPO for Reacher environment. (2 joints)

**Meta-learning PPO:**

* `ppo_single_ini.py`: single-thread PPO for Reacher environment with **initialized policy** in `./ini`.
* `ppo_reptile.py`: **PPO+Reptile** for Reacher environment (3 joints).
* `ppo_reptile_2.py`: **PPO+Reptile** for Reacher environment (2 joints).
* `ppo_fomaml_2.py`: **PPO+FOMAML** for Reacher environment (2 joints).
* `ppo_maml_2.py`: **PPO+MAML** for Reacher environment (2 joints).