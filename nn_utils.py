import torch 
from tqdm import tqdm
from torch.nn.functional import pad 

device = torch.device('cuda')
# device = torch.device('mps')
cpu_device = torch.device('cpu')

def remove_paddding(input_tensor, padding_label=0):
    mask = input_tensor != padding_label
    return input_tensor[mask]

def sentence_level_accuracy(predictions, labels):
    predictions = remove_paddding(predictions.to(cpu_device))
    labels = remove_paddding(labels)
    return int(torch.equal(predictions, labels))

def count_matching_rows(tensor1, tensor2):
    row_matches = torch.all(tensor1 == tensor2, dim=1)
    matching_rows_count = torch.sum(row_matches).item()
    
    return matching_rows_count

def evaluate(model, test_dataloader):
    correct = 0
    all = 0 
    for _, (batch_x, batch_y) in enumerate(tqdm(test_dataloader)):
        logits = model(input_ids=batch_x.to(device), labels=batch_y.to(device)).logits
        preds = torch.argmax(logits, dim=-1)
        all += len(preds)
        correct += count_matching_rows(preds.to(cpu_device), batch_y)
    # print(correct)
    return correct/all 

def get_lengths(encoded_sequence, tokenizer):
    return len(tokenizer.decode(encoded_sequence, skip_special_tokens=True).split(' ')) 

def evaluate_per_lenght(model, test_dataloader, tokenizer):
    correct_per_x_len = {}
    correct_per_y_len = {}
    cnt_per_x_len = {}
    cnt_per_y_len = {}
    acc_per_x_len = {}
    acc_per_y_len = {}
    all_cnt = 0 
    all_correct = 0
    
    for _, (batch_x, batch_y) in enumerate(tqdm(test_dataloader)):
        logits = model(input_ids=batch_x.to(device), labels=batch_y.to(device)).logits
        preds = torch.argmax(logits, dim=-1)
        x_lens = [get_lengths(x, tokenizer) for x in batch_x]
        y_lens = [get_lengths(y, tokenizer) for y in batch_y]
        matches = torch.all(preds.to(cpu_device) == batch_y, dim=1).to(int)
        for x_len, y_len, match in zip(x_lens, y_lens, matches):
            all_cnt += 1 
            all_correct += int(match)
            cnt_per_x_len[x_len] = cnt_per_x_len.get(x_len, 0) + 1 
            cnt_per_y_len[y_len] = cnt_per_y_len.get(y_len, 0) + 1 
            correct_per_x_len[x_len] = correct_per_x_len.get(x_len, 0) + int(match)
            correct_per_y_len[y_len] = correct_per_y_len.get(x_len, 0) + int(match)
    print('correct count per commands length', correct_per_x_len)
    print('correct count per actions lentgh', correct_per_y_len)
    print('all count per commands length', cnt_per_x_len)
    print('all count per actions lentgh', cnt_per_y_len)
    print('general accuracy', all_correct/ all_cnt)
    for x_len in cnt_per_x_len.keys():
        acc_per_x_len[x_len] = correct_per_x_len[x_len] / cnt_per_x_len[x_len]
    for y_len in cnt_per_y_len.keys():
        acc_per_y_len[y_len] = correct_per_y_len[y_len] / cnt_per_y_len[y_len]
   
    return acc_per_x_len, acc_per_y_len, all_correct/ all_cnt
