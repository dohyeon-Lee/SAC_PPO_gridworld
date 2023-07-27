# import torch
# from torch.distributions import Categorical

# # 확률 분포를 나타내는 tensor를 정의합니다.
# probs1 = torch.tensor([[0.1, 0.3, 0.6], [0.1, 0.2, 0.3], [0.1, 0.3, 0.6]])

# # Categorical 분포를 생성합니다.
# dist = Categorical(probs1)

# # 각 행별로 샘플링을 수행합니다. 결과는 랜덤으로 선택된 인덱스들로 이루어진 tensor입니다.
# action = dist.sample()
# action = action.unsqueeze(1)
# prob = probs1[action]
# #prob = prob.unsqueeze(1)
# print("Sampled indices:")
# print(action)
# print(prob)
import torch
from torch.distributions import Categorical

# 확률 분포를 나타내는 tensor를 정의합니다.
probs1 = torch.tensor([[0.1, 0.3, 0.6], [0.1, 0.2, 0.3], [0.1, 0.3, 0.6]])

# Categorical 분포를 생성합니다.
dist = Categorical(probs1)

# 각 행별로 샘플링을 수행합니다. 결과는 랜덤으로 선택된 인덱스들로 이루어진 tensor입니다.
action = dist.sample()
action = action.unsqueeze(1)

# 행 인덱스를 생성합니다.
rows = torch.arange(0, probs1.size(0)).unsqueeze(1)

# 행 인덱스와 열 인덱스를 합쳐서 최종 인덱스를 만듭니다.
indices = torch.cat((rows, action), dim=1)

# 각 샘플에 해당하는 확률을 가져옵니다.
prob = probs1[indices[:, 0], indices[:, 1]].unsqueeze(1)

print("Sampled indices:")
print(indices)
print(prob)