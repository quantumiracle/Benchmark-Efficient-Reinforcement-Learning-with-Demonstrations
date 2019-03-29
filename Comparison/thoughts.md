# How to make Robotics work with Reinforcement Learning

The key problem to make robotics in real world work with reinforcement learning (RL) is: the learning efficiency.

In order to improve that, several strategies are proposed: (1). better initialisation with supervised learning from demonstrations, (2). learning with demonstrations in sampling buffer, (3). meta-learning across tasks in task domain, (4). residual policy learning, etc.



When we have expert demonstrations, we can not only use them for training a supervised learning policy for initialising the actor policy in RL, but can also feed them into the sampling buffer. Actually we have done this when we pre-train the critic part (value evaluation network) in RL. So the first two strategies can work together when we have expert demonstrations: pre-train the actor policy with demonstrations with supervised learning (which can also be called imitation learning) as a better initialisation (and pre-train the critic using the demonstrations with fixed pre-trained actor policy); then keep sampling from the demonstrations and explored samples during RL training process, using prioritized experience replay to balance two sources of samples.

The intuition of above process can be thought of as: we first learn heavily from a teacher to gain some basic experiences (imitate the teacher's demonstrations); then we start our own explorations (reinforcement learning with noisy actions) to try to find some better policies, but meanwhile keep reviewing what we've learned (still sampling from expert demonstrations).  And we compare what we've explored with those expert demonstrations we've learned to determine which are better and therefore supposed to be learned at next step, so as to make sure we keep improving and even better than expert demonstrations.



Above strategies are based on that we have expert demonstrations, which are real world samples for robotics in real world, like we manually move a robot arm to reach some goals and generate expert trajectories with that. The amount of this kind of demonstrations is supposed to be small as it needs human labour (the 'expert'), and our goal to apply robotics is to replace or reduce human's labour. So what if we don't have expert demonstrations in reality?

We have to learn well in simulation, and transfer the learned policy into reality and fine tune it a little bit. There comes the problem of sim-to-real. We need to make sure our policy can learn fast in reality, at least for a task domain (we don't need a robot to do all the things, but a specific set of things). A task domain is a domain of different tasks but with similar basic settings (like same environment with different goals, etc). In order to learn fast in reality on this task domain, we'd better learn across different tasks in this domain, which is multi-task learning. 

Algorithms like modal-agnostic meta-learning (MAML) is one of this kind, for efficient learning in multi-task settings. What MAML learns is one policy that can generalise to all tasks in this domain, as an initialisation which can learn faster for each specific task in this domain than a policy learned from scratch. So MAML's meta policy actually grasps some common knowledge in this task domain, which people believe could help with accelerating the learning process. 

A more efficient idea is the multi-modal modal-agnostic meta-learning (MuMoMAML) or knowledge transfer learning, which learns the adaptation of a basic policy across tasks to a specific policy for one task, or learn a set of policies which could be mapped to each task. Through RL training process the learned policies not only grasp the common knowledge of the task domain, but also the transformation from the policy based on common knowledge to a more specific initialisation of policy for each task.

As it needs to train across tasks to grasp this kind of across-task knowledge, above process usually needs to be conducted in simulation. Therefore it is a good initialisation of policy in simulation. I still haven't figure it out whether we can apply both initialisation strategies with demonstrations and with meta-learning at the same time in simulation, but they can be applied sequentially from simulation to reality.



Another problem in making robotics work in real world lies in the training process in real world: it is not only with high consumption but also unsafe sometimes. The robot cannot move arbitrarily in real world. The strategy called residual policy learning can alleviate this problem. This strategy tells us how to utilise the initialisation policy in a better way. Instead of directly initialise the actor policy with the pre-trained policy with demonstrations or some ways of deriving the initialisation policies, it factorized the actor policy into two parts: the initialised policy and the residual policy. The initialised policy is kept fixed and determined, and the RL only learns the residual policy during training. People believe this strategy effectively reduce the search space of actions and therefore promote or accelerate the learning process. Moreover, as the initialised policy are remained during training, it could help with keeping the robot motions in a safe and sensible range during training in reality. Actually, as long as the output of a task can be factorized, the idea of residual learning can be applied. Although the residual learning method is not very orthogonal to initialisation with demonstrations, we can still combine these two methods.



So a possible workflow of making a real robot work with reinforcement learning could be:

* Build simulations to mimic robotics in real world;

* In simulation, train the policy with MAML/MuMoMAML across tasks;
* In reality, obtain some expert demonstrations and train an initialisation policy using demonstrations via supervised learning;
* Use the trained policy as initialisation, learn a residual policy or learn the initialised policy itself;
* Sample from expert demonstrations (also workable when learning a residual) and explorations, use prioritized experience replay to balance it.