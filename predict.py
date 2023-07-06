import torch
import pandas as pd
from urllib.request import urlretrieve
from torch_geometric.data import Dataset
from torch_geometric.nn import GATConv
import os
from torch.nn import functional as F


class HW3Dataset(Dataset):
    url = 'https://technionmail-my.sharepoint.com/:u:/g/personal/ploznik_campus_technion_ac_il/EUHUDSoVnitIrEA6ALsAK1QBpphP5jX3OmGyZAgnbUFo0A?download=1'

    def __init__(self, root, transform=None, pre_transform=None):
        super(HW3Dataset, self).__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
        return ['data.pt']

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        file_url = self.url.replace(' ', '%20')
        raw_path = os.path.join(self.raw_dir, self.raw_file_names[0])
        try:
            urlretrieve(file_url, raw_path)
        except Exception as e:
            raise Exception(f"Failed to download the file due to error: {e}")


        with open(os.path.join(self.raw_dir, self.raw_file_names[0]), 'wb') as f:
            f.write(response.content)

    def process(self):
        raw_path = os.path.join(self.raw_dir, self.raw_file_names[0])
        data = torch.load(raw_path)
        torch.save(data, self.processed_paths[0])

    def len(self):
        return 1

    def get(self, idx):
        return torch.load(self.processed_paths[0])


class SimpleGAT(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_heads):
        super(SimpleGAT, self).__init__()
        self.conv1 = GATConv(in_dim, hidden_dim, heads=num_heads)
        # On the final layer we set concat to False, meaning we just take the average
        self.conv2 = GATConv(hidden_dim * num_heads, out_dim, concat=False)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)


def save_predictions_to_csv(data, model, file_name):
    model.eval()
    with torch.no_grad():
        out = model(data)
        _, preds = out.max(dim=1)
        indexes = torch.arange(data.num_nodes)
        result_df = pd.DataFrame({"idx": indexes.cpu().numpy(), "prediction": preds.cpu().numpy()})
        result_df.to_csv(file_name, index=False)


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset = HW3Dataset(root='data/hw3/')
    data = dataset[0].to(device)

    model = SimpleGAT(in_dim=128, hidden_dim=64, out_dim=dataset.num_classes, num_heads=6)
    model.load_state_dict(torch.load("my_model.pt", map_location=device))
    model = model.to(device)

    save_predictions_to_csv(data, model, "prediction.csv")

main()