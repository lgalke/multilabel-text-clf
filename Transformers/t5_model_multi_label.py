import warnings
warnings.simplefilter('ignore')
import argparse
import json
import os
import random
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.preprocessing import MultiLabelBinarizer
import transformers
from transformers import T5Tokenizer
import torch
from torch.utils.data import Dataset, DataLoader
import logging
logging.basicConfig(level=logging.ERROR)

DATASET_REGISTRY = {
    # key          : (folder,       num_labels, default_epochs)
    "reuters"      : ("reuters",     90,         15),
    "rcv1-v2"      : ("rcv1-v2",     103,        15),
    "econbiz"      : ("econbiz",     5661,       15),
    "amazon"       : ("amazon",      531,        15),
    "dbpedia"      : ("dbpedia",     298,        5),
    "nyt"          : ("nyt",         166,        15),
    "goemotions"   : ("goemotions",  28,         5),
}

MODEL_NAME = "t5"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class CustomDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.text = dataframe.text
        self.targets = self.data.labels
        self.max_len = max_len

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        text = str(self.text[index])
        text = " ".join(text.split())
        inputs = self.tokenizer.encode_plus(
            text, None,
            add_special_tokens=True,
            max_length=self.max_len,
            truncation=True,
            pad_to_max_length=True,
            return_token_type_ids=False
        )
        return {
            'ids':     torch.tensor(inputs['input_ids'],      dtype=torch.long),
            'mask':    torch.tensor(inputs['attention_mask'], dtype=torch.long),
            'targets': torch.tensor(self.targets[index],      dtype=torch.float),
        }


class T5Class(torch.nn.Module):
    def __init__(self, num_labels):
        super(T5Class, self).__init__()
        self.t5 = transformers.T5ForSequenceClassification.from_pretrained(
            'google-t5/t5-base', num_labels=num_labels
        )

    def forward(self, ids, mask):
        return self.t5(ids, attention_mask=mask).logits


def loss_fn(outputs, targets):
    return torch.nn.BCEWithLogitsLoss()(outputs, targets)


def loss_plot(epochs_range, loss, plot_path):
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(epochs_range, loss, color='red', label='loss')
    plt.xlabel("epochs")
    plt.title("validation loss")
    plt.savefig(plot_path)
    plt.close()


def train_model(n_epochs, training_loader, validation_loader, model, optimizer, device):
    loss_vals = []
    for epoch in range(1, n_epochs + 1):
        train_loss = 0
        valid_loss = 0

        model.train()
        print(f'############# Epoch {epoch}: Training Start   #############')
        for batch_idx, data in enumerate(training_loader):
            optimizer.zero_grad()
            ids     = data['ids'].to(device, dtype=torch.long)
            targets = data['targets'].to(device, dtype=torch.float)
            mask    = data['mask'].to(device, dtype=torch.long)
            outputs = model(ids, mask)
            loss    = loss_fn(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += (1 / (batch_idx + 1)) * (loss.item() - train_loss)

        model.eval()
        with torch.no_grad():
            for batch_idx, data in enumerate(validation_loader, 0):
                ids     = data['ids'].to(device, dtype=torch.long)
                targets = data['targets'].to(device, dtype=torch.float)
                mask    = data['mask'].to(device, dtype=torch.long)
                outputs = model(ids, mask)
                loss    = loss_fn(outputs, targets)
                valid_loss += (1 / (batch_idx + 1)) * (loss.item() - valid_loss)

            train_loss /= len(training_loader)
            valid_loss /= len(validation_loader)
            print(f'Epoch: {epoch} \tAvg Training Loss: {train_loss:.6f} \tAvg Validation Loss: {valid_loss:.6f}')
            loss_vals.append(valid_loss)

    return model, loss_vals


def evaluate(model, testing_loader, device, threshold):
    model.eval()
    fin_targets, fin_outputs = [], []
    with torch.no_grad():
        for _, data in enumerate(testing_loader, 0):
            ids     = data['ids'].to(device, dtype=torch.long)
            mask    = data['mask'].to(device, dtype=torch.long)
            targets = data['targets'].to(device, dtype=torch.float)
            outputs = model(ids, mask)
            fin_targets.extend(targets.cpu().detach().numpy().tolist())
            fin_outputs.extend(torch.sigmoid(outputs).cpu().detach().numpy().tolist())
    preds = np.array(fin_outputs) >= threshold
    return preds, fin_targets


def main():
    parser = argparse.ArgumentParser(description=f"{MODEL_NAME} multi-label text classification")
    parser.add_argument("--dataset",    required=True, choices=list(DATASET_REGISTRY.keys()),
                        help="Dataset name (determines num_labels and default epoch count)")
    parser.add_argument("--seed",       type=int,   default=42)
    parser.add_argument("--data-root",  default="../multi_label_data",
                        help="Root directory containing per-dataset folders")
    parser.add_argument("--lr",         type=float, default=5e-5)
    parser.add_argument("--batch-size", type=int,   default=4)
    parser.add_argument("--max-len",    type=int,   default=512)
    parser.add_argument("--epochs",     type=int,   default=None,
                        help="Override default epoch count from DATASET_REGISTRY")
    parser.add_argument("--threshold",  type=float, default=0.5,
                        help="Sigmoid threshold for converting probabilities to binary predictions")
    parser.add_argument("--output-dir", default="results")
    args = parser.parse_args()

    folder, num_labels, default_epochs = DATASET_REGISTRY[args.dataset]
    n_epochs = args.epochs if args.epochs is not None else default_epochs

    set_seed(args.seed)

    data_dir   = os.path.join(args.data_root, folder)
    train_list = json.load(open(os.path.join(data_dir, "train_data.json")))
    test_list  = json.load(open(os.path.join(data_dir, "test_data.json")))

    train_data   = np.array(list(map(lambda x: list(x.values())[:2], train_list)), dtype=object)
    train_labels = np.array(list(map(lambda x: list(x.values())[2],  train_list)), dtype=object)
    test_data    = np.array(list(map(lambda x: list(x.values())[:2], test_list)),  dtype=object)
    test_labels  = np.array(list(map(lambda x: list(x.values())[2],  test_list)),  dtype=object)

    label_encoder    = MultiLabelBinarizer()
    label_encoder.fit([*train_labels, *test_labels])
    train_labels_enc = label_encoder.transform(train_labels)
    test_labels_enc  = label_encoder.transform(test_labels)

    train_df = pd.DataFrame({'text': train_data[:, 1], 'labels': train_labels_enc.tolist()})
    test_df  = pd.DataFrame({'text': test_data[:, 1],  'labels': test_labels_enc.tolist()})

    print(f"Train texts: {len(train_df)},  Test texts: {len(test_df)}")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    tokenizer = T5Tokenizer.from_pretrained('google-t5/t5-base')

    train_split = train_df.sample(frac=0.8, random_state=args.seed)
    valid_split = train_df.drop(train_split.index).reset_index(drop=True)
    train_split = train_split.reset_index(drop=True)
    test_split  = test_df.reset_index(drop=True)

    print(f"TRAIN: {train_split.shape}  VAL: {valid_split.shape}  TEST: {test_split.shape}")

    train_params = {'batch_size': args.batch_size, 'shuffle': True,  'num_workers': 0}
    eval_params  = {'batch_size': args.batch_size, 'shuffle': False, 'num_workers': 0}

    training_loader   = DataLoader(CustomDataset(train_split, tokenizer, args.max_len), **train_params)
    validation_loader = DataLoader(CustomDataset(valid_split, tokenizer, args.max_len), **eval_params)
    testing_loader    = DataLoader(CustomDataset(test_split,  tokenizer, args.max_len), **eval_params)

    model     = T5Class(num_labels).to(device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr)

    model, loss_vals = train_model(n_epochs, training_loader, validation_loader, model, optimizer, device)

    preds, targets = evaluate(model, testing_loader, device, args.threshold)
    accuracy   = metrics.accuracy_score(targets, preds)
    f1_samples = metrics.f1_score(targets, preds, average='samples')
    f1_micro   = metrics.f1_score(targets, preds, average='micro')
    f1_macro   = metrics.f1_score(targets, preds, average='macro')

    print(f"Accuracy Score = {accuracy}")
    print(f"F1 Score (Samples) = {f1_samples}")
    print(f"F1 Score (Micro) = {f1_micro}")
    print(f"F1 Score (Macro) = {f1_macro}")

    os.makedirs(args.output_dir, exist_ok=True)
    stem      = f"{MODEL_NAME}_{args.dataset}_seed{args.seed}"
    plot_path = os.path.join(args.output_dir, f"{stem}_loss.png")
    txt_path  = os.path.join(args.output_dir, f"{stem}.txt")
    json_path = os.path.join(args.output_dir, f"{stem}.json")

    loss_plot(np.linspace(1, n_epochs, n_epochs).astype(int), loss_vals, plot_path)

    with open(txt_path, "w") as f:
        print(
            f"F1 Score (Samples) = {f1_samples}",
            f"Accuracy Score = {accuracy}",
            f"F1 Score (Micro) = {f1_micro}",
            f"F1 Score (Macro) = {f1_macro}",
            file=f
        )

    payload = {
        "model": MODEL_NAME,
        "dataset": args.dataset,
        "seed": args.seed,
        "num_labels": num_labels,
        "hyperparameters": {
            "lr": args.lr,
            "batch_size": args.batch_size,
            "max_len": args.max_len,
            "epochs": n_epochs,
            "threshold": args.threshold,
        },
        "metrics": {
            "accuracy": accuracy,
            "f1_samples": f1_samples,
            "f1_micro": f1_micro,
            "f1_macro": f1_macro,
        },
        "val_loss_per_epoch": loss_vals,
    }
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"Results saved to {txt_path}, {json_path}, {plot_path}")


if __name__ == "__main__":
    main()
