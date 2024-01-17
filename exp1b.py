import yaml
import os 
from data_loading import read_file, get_dataloader
from transformers import T5Tokenizer, T5ForConditionalGeneration, RobertaTokenizer
from transformers.optimization import Adafactor
import torch 
import wandb
from nn_utils import device, cpu_device, evaluate
from tqdm import tqdm
import sys 


# set wandb parameters out of local_config.yml
with open('local_config.yml', 'r') as file:
    local_user_config = yaml.safe_load(file)

project = local_user_config["project"]
entity = local_user_config["entity"]
wandb.init()
cwd = os.getcwd()
parent = os.path.dirname(cwd)
dataset_path = os.path.join(parent, 'SCAN') 
size_variation = sys.argv[1]
train_file_path = os.path.join(dataset_path, f'simple_split/size_variations/tasks_train_simple_p{size_variation}.txt')
test_file_path = os.path.join(dataset_path, f'simple_split/size_variations/tasks_test_simple_p{size_variation}.txt')
train_commands, train_actions = read_file(train_file_path)
test_commands, test_actions = read_file(test_file_path)

config = {
                    'lr': 0.0001,
                    "batch_size": 8,
                    'n_steps': 100000,
                }

if 't5' in sys.argv:
    model_checkpoint = 't5-small'
    tokenizer = T5Tokenizer.from_pretrained(model_checkpoint)
else: 
    model_checkpoint = 'Salesforce/codet5-small'
    tokenizer = RobertaTokenizer.from_pretrained(model_checkpoint)

model = T5ForConditionalGeneration.from_pretrained(model_checkpoint)
model.to(device)

train_in_tensor = tokenizer(train_commands, padding='max_length',  return_tensors="pt").input_ids
train_out_tensor = tokenizer(train_actions, padding='max_length', return_tensors="pt").input_ids
test_in_tensor = tokenizer(test_commands, padding='max_length',  return_tensors="pt").input_ids
test_out_tensor = tokenizer(test_actions, padding='max_length', return_tensors="pt").input_ids


train_dataloader = get_dataloader(train_in_tensor, train_out_tensor, config["batch_size"], cpu_device)
test_dataloader = get_dataloader(test_in_tensor, test_out_tensor, config["batch_size"], cpu_device)

# optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'])
optimizer = Adafactor(model.parameters(), lr=config['lr'], scale_parameter=False, relative_step=False)
step = 0 

print('size variation', size_variation)
print('model', model_checkpoint)
num_epochs = config['n_steps']//len(train_actions) + 1 
evaluate_x_times = 5 
evaluation_frequency = num_epochs // evaluate_x_times
for epoch in range(num_epochs):
    model.train()
    total_loss = 0

    tqdm_iterator = tqdm(train_dataloader)
    for i, (input_ids_batch, labels_batch) in enumerate(tqdm_iterator):
        input_ids_batch = input_ids_batch.to(device)
        labels_batch = labels_batch.to(device)
        optimizer.zero_grad()

        # Forward pass
        loss = model(input_ids=input_ids_batch, labels=labels_batch).loss
        wandb.log({'step':step, 'train_loss':loss})
        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        
        step += len(input_ids_batch)
        del input_ids_batch, labels_batch
        torch.cuda.empty_cache()
        
        if step + 1 >= config['n_steps']: 
            acc = evaluate(model, test_dataloader)
            print('final accuracy: ', acc)
            wandb.log({'step':step, 'accuracy':acc})
            model.save_pretrained(f'./model_e{epoch}.bin')
            break
    if step + 1 >= config['n_steps']:
        break 
    if epoch % evaluation_frequency == 0 and epoch != 0:
        acc = evaluate(model, test_dataloader)
        wandb.log({'epoch':epoch, 'accuracy':acc})
        model.save_pretrained(f'./model_e{epoch}.bin')
        avg_loss = total_loss / len(train_dataloader)
        print(f"Epoch {epoch + 1}/{num_epochs}, Average Loss: {avg_loss}, Accuracy: {acc}")
        wandb.log({'epoch':epoch, 'avg_loss':avg_loss})









