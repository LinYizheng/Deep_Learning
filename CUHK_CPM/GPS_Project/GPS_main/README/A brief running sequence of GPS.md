Feb 22, 2017
Wenfei Zhu

1. 初始化GPSMain对象，其中**主要**包括初始化：
 - 一个agent对象（×）
 - 一个algorithm对象，其中又会初始化：
    - 对每个condition各初始化一个cost对象（×）
    - 对每个condition各初始化一个dynamics对象
    - 对每个condition，通过init_traj_distr方法各初始化一个LinearGaussianPolicy对象（×）
    - 一个traj_opt对象
    - 如果algorithm选择的是BADMM，那么还会初始化一个policy_opt对象，以及对每个condition初始化一个policy_prior对象
 - 一个gui对象

# zx, begin

training phase

1. run()
  1.1 self._take_sample()       # Collect a sample from the agent.
    1.1.1 self.agent.sample()   # Invoke PI agent's sample function
      1.1.1.1 policy.act()      # LinearGaussianPolicy

# zx, end

2. call GPSMain.run, begin outer loop iteration
 1. 利用LinearGaussianPolicy采样，call agent.sample方法
 2. call algorithm.iteration, begin inner loop iteration # zx, iteration()@algorithm_traj_opt.py,iteration of LQR
 	1. 一些初始化
 	2. 根据采样结果更新dynamics, # zx, self._update_dynamics
 	3. 训练NN policy, call policy_opt.update
 	4. 更新policy prior
 	5. 更新LinearGaussianPolicy, call traj_opt.update
 3. 利用NN policy采样以便检验其学习效果（optional,关闭此功能可节省训练时间）
 4. log data


- - -

**关于Agent类**
Agent类是一个超类，对于每一种机器人，或者仿真环境，我们都需要定义一个新的Agent子类。Agent类原先设计的目的是用于和机器人通信，所以类方法都是通信方法。这要求机器人那边也实现相应的通信接口才行。对于PI的机器人，我们并没有这么做，而是直接把机器人的控制接口暴露给了Agent，在Agent.sample方法中直接控制机器人采样。

**关于Cost类**
在Cost类下面有用于计算不同种类的cost的子类。
- cost_action类用于处理针对action(即u)的惩罚
- cost_state类用于处理针对state(即x)的惩罚
- cost_fk类比较特殊，用于处理针对torque的惩罚（不适用于我们现在的情况）
- cost_sum类用于求多种类的加权平均
- cost_coord6d类是我们自己定义的，其糅合了cost_action, cost_state, cost_sum的功能，便于调试(implemented by PIE)

traj_opt对象会利用cost类中的方法来计算cost用于trajectory optimization，即更新LinearGaussianPolicy。通过调整cost类中的cost计算方法及参数，可以改变最后优化出来的轨迹。

**关于init_traj_dist方法**
这个方法用于初始化LinearGaussianPolicy。原作者实现了2种方法，一种是init_lqr,一种是init_pd，我们只用过前者。对于这个方法有2点需要注意：第一，方法默认的初始化条件（或者说是trajecotry optimization的目标）是把机器人保持在初始状态(hold the initial state)。这个条件可以改。第二，trajectory optimization需要用到dynamics，因为一开始我们并不知道dynamics，所以需要去猜。所以原作者写了一个叫guess_dynamics的函数。因为原作者使用torque去控制机器人，所以原本的guess_dynamics只适用于有torque的情况，我们把它改成了适用于利用joint angle控制的情况。
