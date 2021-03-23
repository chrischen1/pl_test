import argparse

from torch.utils import data

from data import Random_dict_Dataset
from network import graph_model, dict_model


if __name__ == '__main__':

    ap = argparse.ArgumentParser(description='Process data for training')
    ap.add_argument('--network', type=str, required=False, default='dict_model')
    ap.add_argument('--out_path', type=str, required=False,
                    default='output')
    ap.add_argument('--num_gpus', type=int, required=False, default=1)
    ap.add_argument('--num_workers', type=int, required=False, default=4)
    ap.add_argument('--checkpoint_file', type=str, required=False, default='output/last.ckpt')
    ap.add_argument('--test_size', type=int, required=False, default=5)

    args = ap.parse_args()
    network = args.network
    out_path = args.out_path
    num_gpus = args.num_gpus
    num_workers = args.num_workers
    checkpoint_file = args.checkpoint_file
    test_size = args.test_size

    data_loader = data.DataLoader(Random_dict_Dataset(test_size), num_workers=num_workers, batch_size=1)
    if network == 'graph_model':
        model = graph_model.load_from_checkpoint(checkpoint_file)
    elif network == 'dict_model':
        model = dict_model.load_from_checkpoint(checkpoint_file)
    else:
        raise NotImplementedError

    pred = []
    for it, x in enumerate(data_loader):
        out = model(x).squeeze()
        pred.append(out)

