import torch 

device = torch.device('cuda')
cpu_device = torch.device('cpu')

def remove_paddding(input_tensor, padding_label=0):
    mask = input_tensor != padding_label
    return input_tensor[mask]

def sentence_level_accuracy(predictions, labels):
    predictions = remove_paddding(predictions.to(cpu_device))
    labels = remove_paddding(labels)
    return int(torch.equal(predictions, labels))

def evaluate(model, test_dataloader):
    cnt = 0 
    correct = 0
    for batch_x, batch_y in test_dataloader:
        preds = model.generate(batch_x.to(device), max_length=480)
        for i in range(len(preds)):
            correct += sentence_level_accuracy(preds[i], batch_y[i])
            cnt += 1 
    return correct/cnt
