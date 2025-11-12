# 这是一个简单的 Q-Learning 算法示例：
# 环境是一个一维世界, 在世界的右边有宝藏, 探索者只要得到宝藏尝到了甜头,
#  然后以后就记住了得到宝藏的方法, 这就是他用强化学习所学习到的行为。
# 即-o---T      T 就是宝藏的位置, o 是探索者的位置

import torch
import pandas as pd
import numpy as np

# 第一步该干啥？超参数应该定义哪些？不知道[尬笑]
# 先写写神经网络吧
#class Qnet(nn.Module):

#看了一眼示例：
#为什么没写class? 只有一些def？

#AI告诉我是因为这个例子太简单了，用pop就行

#那我看看需要写什么函数，求出Q值、选择动作、优化更新，目前知道要写这三个


# 建立Q表
# 需要传入 状态的数量，以及 有哪些动作
def Q_value(n_states,actions):
    Q_table = pd.DataFrame(
        np.zeros((n_states,len(actions))),
        columns=actions     # 列名是动作,都是列表
    )