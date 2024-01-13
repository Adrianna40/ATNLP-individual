import yaml
import os 
from data_loading import read_file, get_dataloader
from transformers import RobertaTokenizer, T5ForConditionalGeneration
import torch 
import wandb
from nn_utils import device, cpu_device, evaluate


# set wandb parameters out of local_config.yml
with open('local_config.yml', 'r') as file:
    local_user_config = yaml.safe_load(file)

project = local_user_config["project"]
entity = local_user_config["entity"]
wandb.init()
cwd = os.getcwd()
parent = os.path.dirname(cwd)
dataset_path = os.path.join(parent, 'SCAN') 
size_variation = '32'
train_file_path = os.path.join(dataset_path, f'simple_split/size_variations/tasks_train_simple_p{size_variation}.txt')
test_file_path = os.path.join(dataset_path, f'simple_split/size_variations/tasks_test_simple_p{size_variation}.txt')
train_commands, train_actions = read_file(train_file_path)
test_commands, test_actions = read_file(test_file_path)

config = {
                    'epochs': 30,
                    'lr': 1e-3,
                    "batch_size": 8
                }

model_checkpoint = 'Salesforce/codet5-small'
tokenizer = RobertaTokenizer.from_pretrained(model_checkpoint)
model = T5ForConditionalGeneration.from_pretrained(model_checkpoint)

# model = T5ForConditionalGeneration.from_pretrained("./model_e9.bin")
model.to(device)

train_in_tensor = tokenizer(train_commands, padding='max_length',  return_tensors="pt").input_ids
train_out_tensor = tokenizer(train_actions, padding='max_length', return_tensors="pt").input_ids
test_in_tensor = tokenizer(test_commands, padding='max_length',  return_tensors="pt").input_ids
test_out_tensor = tokenizer(test_actions, padding='max_length', return_tensors="pt").input_ids


train_dataloader = get_dataloader(train_in_tensor, train_out_tensor, config["batch_size"], cpu_device)
test_dataloader = get_dataloader(test_in_tensor, test_out_tensor, config["batch_size"], cpu_device)

optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'])
step = 0 

print('size variation', size_variation)
print('model', model_checkpoint)
# print(evaluate(model, test_dataloader))

for epoch in range(config['epochs']):
    model.train()
    total_loss = 0

    for batch in train_dataloader:
        input_ids_batch, labels_batch = batch
        input_ids_batch = input_ids_batch.to(device)
        labels_batch = labels_batch.to(device)
        optimizer.zero_grad()

        # Forward pass
        loss = model(input_ids=input_ids_batch, labels=labels_batch).loss
        wandb.log({'step':step, 'train_loss':loss})
        step += 1 
        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        del input_ids_batch, labels_batch
        torch.cuda.empty_cache()
        

    avg_loss = total_loss / len(train_dataloader)
    acc = evaluate(model, test_dataloader)
    print(f"Epoch {epoch + 1}/{config['epochs']}, Average Loss: {avg_loss}, Accuracy: {acc}")
    wandb.log({'epoch':epoch, 'avg_loss':avg_loss, 'accuracy':acc})
    model.save_pretrained(f'./model_e{epoch}.bin')
    wandb.save(f'./model_e{epoch}.bin')









