
import torch 
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score


def train(args, model, device, train_loader, val_loader, optimizer, scheduler):
    model.train()
    for epoch in range(args.epochs):
        epoch_loss = 0
        batch_idx = 0

        for X,y in train_loader:
            batch_idx+=1
            X,y = X.to(device), y.to(device)
            
            ## Write code here
            y_pred = model(X)
            loss = model.loss(y_pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

            if batch_idx % args.log_interval == 0:
                print(f"Batch {batch_idx}:\t Avg Train Loss: {round(epoch_loss/batch_idx,5)}")

        val_loss = round(compute_loss(model, device, val_loader), 5)
        print(f"Epoch {epoch+1}:\t Avg Train Loss: {round(epoch_loss/batch_idx,5)}\t Avg Val Loss: {val_loss}")
        scheduler.step()

@torch.no_grad()
def compute_loss(model, device, data_loader):
    total_loss = 0
    for i, (X,y) in enumerate(data_loader):
        X,y = X.to(device), y.to(device)
        y_pred = model(X)

        loss = model.loss(y_pred, y)
        total_loss += loss.item()

    return total_loss/(i+1)


def evaluate(model, device, args, test_loader):
    y_true = []
    for _, y in test_loader:
    # showed error TypeError: can't convert mps:0 device type tensor to numpy. Use Tensor.cpu() to copy the tensor to host memory first.
        y_true.extend(y.cpu())
    
    y_pred = predict(model, test_loader, device)
    
    
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred,average='macro')
    recall = recall_score(y_true, y_pred,average='macro')
    
    
    print("\nEvaluation Results on Test Data:")
    print(f"Accuracy:  {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall:    {recall}")

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall
    }
    
    

def predict(model, data_loader, device):
    all_prediction = []
    for X, _ in data_loader:
        X = X.to(device)
        y_pred = model(X)  # Run forward pass to get predictions
        pred_labels = torch.argmax(y_pred, dim=1)  # Choose class with highest score
        # https://discuss.pytorch.org/t/what-is-the-cpu-in-pytorch/15007
        # Some operations on tensors cannot be performed on cuda tensors so you need to move them to cpu first.
        all_prediction.extend(pred_labels.cpu())  # Save predictions
    return all_prediction