import os
import gc
import torch
import numpy as np
import pandas as pd
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModel
from rdkit import Chem
import selfies as sf
from tqdm import tqdm

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class SELFIESDataset(Dataset):
    def __init__(self, selfies_list):
        self.selfies = selfies_list

    def __len__(self):
        return len(self.selfies)

    def __getitem__(self, idx):
        return self.selfies[idx]


class ModelFineTuner(nn.Module):
    def __init__(self, model_name='ibm/materials.selfies-ted'):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        hidden_size = self.model.config.hidden_size
        self.regressor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Dropout(0.1),
            nn.GELU(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state
        mask = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
        summed = torch.sum(hidden_states * mask, dim=1)
        counts = torch.clamp(mask.sum(1), min=1e-9)
        pooled = summed / counts
        return self.regressor(pooled)


class SELFIESEncoder:
    def __init__(self, model=None):
        self.tokenizer = AutoTokenizer.from_pretrained("ibm/materials.selfies-ted")
        self.model = model or AutoModel.from_pretrained("ibm/materials.selfies-ted")
        self.invalid_indices = []
        self.model.eval()

    @staticmethod
    def smiles_to_selfies(smiles):
        try:
            return sf.encoder(smiles.strip()).replace('][', '] [')
        except:
            try:
                canonical = Chem.MolToSmiles(Chem.MolFromSmiles(smiles.strip()))
                return sf.encoder(canonical).replace('][', '] [')
            except:
                return None

    @torch.no_grad()
    def get_embedding_batch(self, selfies_batch):
        encodings = self.tokenizer(
            selfies_batch,
            return_tensors='pt',
            max_length=128,
            truncation=True,
            padding='max_length'
        )
        encodings = {k: v for k, v in encodings.items()}

        outputs = self.model(
            input_ids=encodings['input_ids'],
            attention_mask=encodings['attention_mask']
        )

        return outputs.squeeze().numpy()

    def convert(self, smiles_input):
        if isinstance(smiles_input, str):
            smiles_list = [smiles_input]
            single_input = True
        else:
            smiles_list = smiles_input
            single_input = False

        selfies = [self.smiles_to_selfies(s) for s in smiles_list]
        self.invalid_indices = [i for i, s in enumerate(selfies) if s is None]
        selfies = [s if s else '[nop]' for s in selfies]

        return selfies if not single_input else selfies[0]

    def encode(self, smiles_input, use_gpu=True, return_tensor=False, batch_size=128, num_workers=0):
        is_single = isinstance(smiles_input, str)
        smiles_list = [smiles_input] if is_single else smiles_input

        selfies = self.convert(smiles_list)
        dataset = SELFIESDataset(selfies)
        device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
        self.model.to(device)

        loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)
        embeddings = []

        for batch in tqdm(loader, desc="Encoding"):
            emb = self.encode_batch(batch)
            embeddings.append(emb)
            del emb
            gc.collect()

        final_emb = np.vstack(embeddings)

        for idx in self.invalid_indices:
            final_emb[idx] = np.nan
            print(f"[Warning] Invalid SELFIES for input: {smiles_list[idx]}")

        result = torch.tensor(final_emb) if return_tensor else pd.DataFrame(final_emb)
        return result.iloc[0] if is_single else result

    def predict(self, smiles_input, use_gpu=True, batch_size=128, num_workers=0):
        is_single = isinstance(smiles_input, str)
        smiles_list = [smiles_input] if is_single else smiles_input

        selfies = self.convert(smiles_list)
        dataset = SELFIESDataset(selfies)

        device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
        self.model.to(device)

        loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)

        output = []
        for batch in tqdm(loader, desc="Encoding"):
            pred = self.get_embedding_batch(batch)
            output.append(pred)

        final_output = np.vstack(output)
        return final_output[0] if is_single else final_output



def load_finetuned_model(ckpt_filename="qm9-homo.pt"):
    model = torch.load(ckpt_filename)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained('ibm/materials.selfies-ted', trust_remote_code=True)
    return model, tokenizer


"""# Example usage
if __name__ == "__main__":
    base_model_path = "qm9-homo.pt"

    model = load_finetuned_model(base_model_path)
    tokenizer = AutoTokenizer.from_pretrained('ibm/materials.selfies-ted', trust_remote_code=True)

    test_smiles = "C1CC1C#N"
    test_selfie = sf.encoder(test_smiles).replace("][", "] [")

    encoded = tokenizer([test_selfie], truncation=True, padding='max_length', max_length=128, return_tensors='pt')
    with torch.no_grad():
        output = model(encoded['input_ids'], encoded['attention_mask']).squeeze()
    print(output.numpy())"""
