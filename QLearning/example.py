# 这是一个简单的 Q-Learning 算法示例：
# 环境是一个一维世界, 只有在世界的最右边有宝藏, 探索者只要得到宝藏尝到了甜头,
#  然后以后就记住了得到宝藏的方法, 这就是他用强化学习所学习到的行为。
# 即-o---T      T 就是宝藏的位置, o 是探索者的位置

import torch
import pandas as pd
import numpy as np
import time

EPSILON = 0.9
ACTIONS = ['left', 'right']
N_STATES = 6 #这个一维世界的长度
FRESH_TIME = 2
MAX_EPISODES = 10
GAMMA = 0.9
ALPHA = 0.1

# 第一步该干啥？超参数应该定义哪些？不知道[尬笑]
# 先写写神经网络吧
#class Qnet(nn.Module):

#看了一眼示例：
#为什么没写class? 只有一些def？

#AI告诉我是因为这个例子太简单了，用pop就行

#那我看看需要写什么函数，求出Q值、选择动作、优化更新，目前知道要写这三个


# 建立Q表
# 需要传入 状态的数量、有哪些动作
def Q_value(n_states,actions):
    Q_table = pd.DataFrame(
        np.zeros((n_states,len(actions))),  # n_states行len(actions)列 的0矩阵
        columns=actions     # 列名是动作,都是列表
    )
    return Q_table


# 选择动作
# 想想需要什么，Q值、state(知道是在哪个状态下的动作选择)、
# 然后用epsilon-greedy选出动作
def choose_action(Q_table,state):
    state_actions = Q_table.iloc[state, :]  # .iloc[a, b]是pandas的方法，用于选中a行b列的元素
    # ：表示所有，即选中第state行，所有列，也就是该状态下的所有动作的Q值
    if (np.random.uniform() > EPSILON) or (state_actions == 0).all():
        # .all()属于panda方法，用于判断series中是否全部相同
        # 若相同则返回True
        action_name = np.random.choice(ACTIONS)    # choice是random库中的的方法，用于从一维数组中随机选择一个元素
    else:
        action_name = state_actions.idxmax()# 返回最大值的索引
    return action_name

# 还要干什么？
# 哦，开始更新，怎么更新
# 要开始实现算法的核心部分了    不对，不是核心部分

# 不会写啊…………抄一抄吧先

"""
大体思路：

因为只有在最右边才有宝藏，向右走状态（就相当于格子的序号数）加一，向左走减一
对于奖励就简化成得到宝藏奖励1，其余奖励一律为零
单独考虑两种特殊情况，在原点（无法向左走），在倒数第二个点（如果向右则已到终点）

(1)向左走
    ①不在0处，位置减一
    ②在0处，位置不变
(2)向右走
    ①在倒数第二格，获得宝藏（省去了判断是否到达终点这一步）
    ②不在，位置加一
"""

# 智能体和环境的交互
def get_env_feedback(S,A):      # S:
    # This is how agent will interact with the environment
    if A == 'right':
        if S == N_STATES - 2:
            S_ = 'terminal'
            R = 1
        else:
            S_ = S + 1
            R = 0
    else:
        R = 0
        if S == 0:
            S_ = S  # reach the wall
        else:
            S_ = S - 1
    return S_, R

"""
先初始化一个环境，如果状态到达宝藏的位置，则本轮结束，输出本轮的训练结果；
如果状态没有到达，则用o代表agent的位置进行环境更新并输出；
"""

# 更新环境，就是做一个简单的可视化
# 要有状态s、
def update_env(S,episode,step_counter):
    env_list = ['-']*(N_STATES - 1) + ['T']   # 初始化一个可视的一维世界
    
    if S =='terminal':
        interaction = f"Episode {episode+1}:total_steps = {step_counter}"
        print(f"\r{interaction}",end = '') # end = ''表示不换行，下一次打印继续在当前行的起始位置输出，与\r配合实现动态展示
        time.sleep(2) # 停两秒
        print(f'\r{" " * 32}',end = '') # 用空格覆盖

    else:
        env_list[S] = 'o' # 目前agent所处的位置
        interaction = ''.join(env_list)
        # ''.join(env_list) 调用 str.join() 方法，把列表中的每个元素用空字符串 '' 作为“连接符”拼接起来，变成一个完整的字符串
        print(f"\r{interaction}",end = '')
        time.sleep(FRESH_TIME)

# 下面才是核心部分
# 是主要循环部分，在不断交互中学习最佳策略的过程
# 上面的封装都是为下面的循环做的准备

def rl():
    q_table = Q_value(N_STATES,ACTIONS)

    # 开始训练
    for episode in range(MAX_EPISODES):
        # 初始化
        S = 0
        step_counter = 0
        is_terminated = False

        # 每回合结束后更新
        update_env(S, episode, step_counter)

        # 开始回合更新
        while not is_terminated:
            A = choose_action(q_table,S)
            S_,R = get_env_feedback(S,A)
            q_predict = q_table.loc[S, A]   # 这是估算的值
            # .loc[行标签，列标签]用于选取数据，是panda里的用法

            if S_ != 'terminal':
                q_target = R + GAMMA * q_table.iloc[S_, :].max()   #  实际的值 (回合没结束)
            else:
                q_target = R     #  实际的(状态-行为)值 (回合结束)
                is_terminated = True    # terminate this episode

            #  q_table 更新
            q_table.loc[S, A] += ALPHA * (q_target - q_predict)  
            S = S_  # 探索者移动到下一个 state

            update_env(S, episode, step_counter+1)  # 环境更新

            step_counter += 1
    return q_table


if __name__ == "__main__":
    Q_table = rl()
print('\r\nQ-table:\n')
print(Q_table)
            