import torch
model = torch.load('C:\\Users\\73559\\Desktop\\ml\\net1.pkl')
state_dict = model.state_dict()
print(state_dict.keys())
print(state_dict['conv1.0.weight'].size())
