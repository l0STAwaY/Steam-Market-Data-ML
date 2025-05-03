import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.feature_extraction.text import TfidfVectorizer
import torch
from datasets import Dataset
from tqdm import tqdm
from transformers import DataCollatorWithPadding
import subprocess



def get_predictions(model, data, batch_size=64):
    """ Get predictions on data for a classification model m returning predictions and true labels"""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model.to(device)
    model.eval()

    all_predictions = []
    all_confidences = []
    
    if len(data['input_ids']) % batch_size != 0:
        num_batches = len(data['input_ids']) // batch_size + 1
    else:
        num_batches = len(data['input_ids']) // batch_size

    for i in tqdm(range(num_batches),desc= "Prediction Progress"):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(data['input_ids'])) # prevent going out of bount input_ids has same size as attention mask should
        input_ids_batch = torch.tensor(data['input_ids'][start_idx:end_idx]).to(device)
        attention_mask_batch = torch.tensor(data['attention_mask'][start_idx:end_idx]).to(device)
        with torch.no_grad():  # Disable gradient computation for inference
            predictions = model(input_ids_batch, attention_mask=attention_mask_batch)
            logits = predictions.logits
            predictions = torch.argmax(logits, dim=-1) # make sure it is a ser
            probs = torch.nn.functional.softmax(logits, dim=-1) # return a proabiblity distribution
            max_probs, _ = torch.max(probs, dim=-1)
            
        all_confidences.append(max_probs)  # this tells us how confident we are with an value
        all_predictions.append(predictions)       
        # print(predictions)
        # print("--------")
    # print(all_predictions)
    
    
    #Concatenate all batch predictions into one tensor since we currently have a list of lists
    all_predictions = torch.cat(all_predictions, dim=0)
    all_confidences = torch.cat(all_confidences, dim=0)
    
    return all_predictions , all_confidences



def self_training_loop(model, train_dataset,tokenizer,confidence_threshold=0.9, num_epochs=1, mode='unlabeled'):
    
    # still need to better develop still bad code since exit really randomly depend on the confidence_threshold which is not good for big data     waiting for the final data point to set in
    data_collator = DataCollatorWithPadding(tokenizer) 


    training_args = TrainingArguments(
        num_train_epochs=num_epochs,      # Number of training epochs
    )

    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,    # Training dataset   
        data_collator=data_collator,
    )
    
    #https://github.com/tqdm/tqdm#usage
    unlabeled_mask =  train_dataset["label"] == -1
    total_unlabeled = sum(label == -1 for label in train_dataset["label"])
    pbar = tqdm(total=total_unlabeled, desc="Unlabeled Samples Remaining")

    labeled_train_dataset = train_dataset.filter(lambda x: x["label"] != -1)
    trainer.train_dataset = labeled_train_dataset  # Update trainer with labeled dataset
    print(f"data: { labeled_train_dataset =}")
    # https://peps.python.org/pep-0289/
    print(sum(label == -1 for label in labeled_train_dataset["label"]))
    print("hello")
    trainer.train()
    
    while -1 in train_dataset["label"]:
            
            predictions, confidences = get_predictions(model, train_dataset)
            print(confidences)
            print(predictions)
            print(len(train_dataset))
            print("------------------------")
            if mode == 'unlabeled':
            # we might want to make distinction here is the train_dataset label only or etc
            #  It allows you to apply a processing function to each example in a dataset, independently or in batches. This function               can even create new rows and columns.
            # https://huggingface.co/docs/datasets/en/process
                condition_met_indices = [] # this is just here to keep track of the pesudo labeled added per iteration if needed
                # example here is each row
                def update_labels(example, idx):
                    # Update labels for unlabeled data only, if confidence is above the threshold
                    if example['label'] == -1 and confidences[idx] >= confidence_threshold:
                        example['label'] = predictions[idx]
                        condition_met_indices.append(idx)
                    return example
    
                # Use .map() to apply the updates to all examples where the condition is met
                train_dataset = train_dataset.map(update_labels, with_indices=True)
                print(f"Indices where the condition was met: {condition_met_indices}")
                      
    
            # untested yet !!!!!!!
            elif mode == "all":
                def update_labels_all(example, idx):
                    if confidences[idx] >= confidence_threshold:
                        example['label'] = predictions[idx]
                        condition_met_indices.append(idx)
                    return example
    
                train_dataset = train_dataset.map(update_labels_all, with_indices=True)
                print(f"Indices where the condition was met: {condition_met_indices}")
    
         # Update trainer with the latest training data
            pbar.update(len(condition_met_indices))
            trainer.train_dataset = train_dataset
            trainer.train()
                
            if -1 not in train_dataset["label"]:
              print("All labels are filled. Stopping training.")
              break     
    return model, train_dataset

def main():

    
    
    # We are still in datafre
    
    review_data = pd.read_csv("Machine_Learning_final/unlabeled_reviews.csv")
    print(review_data.shape)
    
    print(review_data["review_comment"].isna().sum())
    review_data["review_comment"] = review_data["review_comment"].fillna("") 
    review_data["recommend"] = review_data["recommend"].map({"Recommended": 1, "Not Recommended": 0})
    
    
    
    # vectorizer = TfidfVectorizer()
    
    # review_X = vectorizer.fit_transform(review_data["review_comment"])  # Convert text to TF-IDF features
    
    # # Array representing the labels. Unlabeled samples should have the label -1.
    
    review_data["recommend"] = review_data["recommend"].fillna(-1)
    
    
    print("wow")
    print(torch.cuda.get_device_name(0))

    # Run nvidia-smi and capture the output
    try:
        result = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
        print("nvidia-smi output:\n", result.stdout)
    except subprocess.CalledProcessError as e:
        print("Error running nvidia-smi:", e.stderr)
    print(review_data[review_data["recommend"]==-1])
    
    
    # for some reason the tyepe matter https://discuss.huggingface.co/t/valueerror-target-size-torch-size-8-must-be-the-same-as-input-size-torch-size-8-8/12133/9
    train_df = pd.DataFrame({
    "text": review_data["review_comment"], 
    "label": review_data["recommend"].apply(lambda x: int(x) if x in [0, 1, -1] else -1)
})
    
    
    
    # Now we are working with huggingface daset 
    
    train_dataset = Dataset.from_pandas(train_df)
    
    # print(sum(label == -1 for label in train_dataset["label"]))
    # print(train_dataset["label"])
    unlabeled_mask =  train_dataset["label"] == -1
    print(unlabeled_mask)
    # Load tokenizer
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    
    # Bert model accepts max 512
    # Tokenize datasets
    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, padding='max_length', max_length=512)
    
    #------------------------------------
    
    
    #------------------------------------
    
    # In Hugging Face models like BERT, the features come from the tokenized input data that is fed into the model.
    # the map applie the tokenize function to each batch of the function 
    train_dataset = train_dataset.map(tokenize, batched=True)
    
    # print(train_dataset)
    # Load model
    # define model
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model.to(device)
    
    model, train_dataset = self_training_loop(model, train_dataset,tokenizer)
    
    # Save the trained model and tokenizer
    model.save_pretrained("self_trained_bert_model")
    tokenizer.save_pretrained("self_trained_tokenizer")
    
    # Save the final labeled dataset
    train_dataset_df = train_dataset.to_pandas()  # Convert dataset back to pandas DataFrame
    train_dataset_df = train_dataset_df[["label","text"]]
    train_dataset_df.to_csv("final_labeled_reviews.csv", index=False)
    
    
    #---------Evaluate-----------
    
    
    
    train_dataset_df = train_dataset.to_pandas()  # Convert dataset back to pandas DataFrame
    train_dataset_df = train_dataset_df[["label","text"]]
    train_dataset_df.to_csv("final_labeled_reviews.csv", index=False)
    review_data_test = pd.read_csv("Machine_Learning_final/cleaned_reviews.csv")
    
    
    
    print(review_data_test ["review_comment"].isna().sum())
    review_data_test ["review_comment"] = review_data_test ["review_comment"].fillna("") 
    review_data_test ["recommend"] = review_data_test ["recommend"].map({"Recommended": 1, "Not Recommended": 0})
    review_data_test ["recommend"] = review_data_test ["recommend"].fillna(-1)
    
    
    
    review_y_test = review_data_test["recommend"]
    
    train_dataset_frame =  train_dataset.to_pandas()
    # only the unlabled data
    mask = review_data["recommend"] == -1 
    review_y_test =  review_y_test[mask]
    
    final_data_set = train_dataset_frame[mask]
    print(review_y_test)
    print(final_data_set)
    
    # print(review_X_train.shape[0])
    # This metaestimator allows a given supervised classifier to function as a semi-supervised classifier, allowing it to learn from unlabeled data. 
    # It does this by iteratively predicting pseudo-labels for the unlabeled data and adding them to the training set.
    
    accuracy = accuracy_score(review_y_test, final_data_set["label"] )
    precision = precision_score(review_y_test, final_data_set["label"]  ,average='macro')
    recall = recall_score(review_y_test,final_data_set["label"]  ,average='macro')
    
    
    print("\nEvaluation Results on Test Data:")
    print(f"Accuracy:  {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall:    {recall}")
    
    with open('evaluation_results_selftrain_noboot_bert_models.txt', 'w') as f:
        f.write(f"Accuracy after fine-tuning: {accuracy:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall: {recall:.4f}\n")
        
if __name__ == '__main__':  
    main()
