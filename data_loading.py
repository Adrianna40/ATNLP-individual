import re 
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
import torch 


def read_file(file_path):
    commands = []
    actions = []
    with open(file_path, 'r') as f:
        for line in f.readlines():
            command_str = re.findall('(?<=IN: )(.*)(?= OUT:)', line)[0]   # match with all characters after IN: and before OUT: 
            actions_str = re.findall('(?<=OUT: )(.*)', line)[0]  # match with all characters after OUT: 
            commands.append(command_str)
            actions.append(actions_str)
 
    assert len(commands) == len(actions), 'Commands and actions lists do not match in size'
    return commands, actions

def get_dataloader(input_ids, output_ids, batch_size, device):
    data = TensorDataset(input_ids.to(device),
                         output_ids.to(device))
    sampler = RandomSampler(data)
    dataloader = DataLoader(data, sampler=sampler, batch_size=batch_size)
    return dataloader





