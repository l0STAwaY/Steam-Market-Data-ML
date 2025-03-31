import torch 
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score



def batch_train(model, train_loader, valid_loader, num_epochs, lr=0.0001):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    #  Learning Rate Scheduler


    for epoch in range(num_epochs):
        epoch_loss = 0
        num_batches = 0

        for X,y in train_loader:
            ## Complete this loop
            num_batches += 1

            ## Hint: You might have to reshape y_pred and flatten y. https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
            hidden = model.init_hidden(X.size(0))  # X.size(0) is batch size
            # Forward pass
            y_pred, _  = model(X, hidden)
            # Calculate loss
            # print(y)
            # print("oredict",y_pred)
            # print("y_pred shape",y_pred.shape)
            # print("y_target shape",y.shape)
            # print("y_pred shape after",y_pred.view(-1, y_pred.size(-1)).shape)
            # print("y_target shape after",y.view(-1).shape)

            # Compute the loss
                        
            loss = model.loss(y_pred.reshape(-1, y_pred.size(-1)), y.reshape(-1)) # Flatten y to match predictions
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25) #helps with exploding gradient
            optimizer.step()
            epoch_loss += loss.item()

        val_loss = round(compute_loss(model, valid_loader), 5)
        # if epoch%5 == 0:
        #     print(f"Epoch {epoch+1}:\t Avg Train Loss: {round(epoch_loss/num_batches,5)}\t Avg Val Loss: {val_loss}")



@torch.no_grad()  ## ensures gradient not computed for this function. 
def compute_loss(model, data_loader):
    """
    Returns avg loss of the model on the data
    """
    total_loss = 0
    for i, datapoint in enumerate(data_loader):
        ## Implement this loop
        X, y = datapoint
        hidden = model.init_hidden(X.size(0))  # Initialize hidden state for validation

        # Forward pass
        y_pred, _ = model(X, hidden) # we ignore variable

        # Calculate loss
        loss = model.loss(y_pred.reshape(-1, y_pred.size(-1)), y.reshape(-1))
        total_loss += loss.item()

    return total_loss/(i+1)


def evaluate(model, test_loader):
    y_true = []
    for _, y in test_loader:
    # showed error TypeError: can't convert mps:0 device type tensor to numpy. Use Tensor.cpu() to copy the tensor to host memory first.
        y_true.extend(y.cpu())
    
    y_pred = predict(model, test_loader)
    print(y_true)
    
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
    
    

def predict(model, data_loader):
    all_prediction = []
    for X, _ in data_loader:
        hidden = model.init_hidden(X.size(0))
        y_pred , _  = model(X,hidden)  # Run forward pass to get predictions
        pred_labels = torch.argmax(y_pred, dim=1)  # Choose class with highest score
        # https://discuss.pytorch.org/t/what-is-the-cpu-in-pytorch/15007
        # Some operations on tensors cannot be performed on cuda tensors so you need to move them to cpu first.
        # print(torch.softmax(y_pred,dim=1))
        all_prediction.extend(pred_labels)  # Save predictions
    return all_prediction