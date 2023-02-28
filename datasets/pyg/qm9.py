from typing import Optional, Callable, List

import sys
import os
import os.path as osp
from tqdm import tqdm
import numpy as np

import torch
import torch.nn.functional as F
from torch_scatter import scatter
from torch_geometric.data import (InMemoryDataset, download_url, extract_zip,
                                  Data)
from torch_geometric.nn import radius_graph


HAR2EV = 27.211386246
KCALMOL2EV = 0.04336414

conversion = torch.tensor([
    1., 1., HAR2EV, HAR2EV, HAR2EV, 1., HAR2EV, HAR2EV, HAR2EV, HAR2EV, HAR2EV,
    1., KCALMOL2EV, KCALMOL2EV, KCALMOL2EV, KCALMOL2EV, 1., 1., 1.
])

atomrefs = {
    6: [0., 0., 0., 0., 0.],
    7: [
        -13.61312172, -1029.86312267, -1485.30251237, -2042.61123593,
        -2713.48485589
    ],
    8: [
        -13.5745904, -1029.82456413, -1485.26398105, -2042.5727046,
        -2713.44632457
    ],
    9: [
        -13.54887564, -1029.79887659, -1485.2382935, -2042.54701705,
        -2713.42063702
    ],
    10: [
        -13.90303183, -1030.25891228, -1485.71166277, -2043.01812778,
        -2713.88796536
    ],
    11: [0., 0., 0., 0., 0.],
}


targets = ['mu', 'alpha', 'homo', 'lumo', 'gap', 'r2', 'zpve', 'U0',
           'U', 'H', 'G', 'Cv', 'U0_atom', 'U_atom', 'H_atom', 'G_atom', 'A', 'B', 'C']


# for pre-processing target based on atom ref
atomrefs_tensor = torch.zeros(5, 19)
atomrefs_tensor[:, 7]  = torch.tensor(atomrefs[7])
atomrefs_tensor[:, 8]  = torch.tensor(atomrefs[8])
atomrefs_tensor[:, 9]  = torch.tensor(atomrefs[9])
atomrefs_tensor[:, 10] = torch.tensor(atomrefs[10])


class QM9(InMemoryDataset):
    """
    1. This is the QM9 dataset, adapted from Pytorch Geometric to incorporate 
    cormorant data split. (Reference: Geometric and Physical Quantities improve 
    E(3) Equivariant Message Passing)
    2. Add pair-wise distance for each graph. """

    raw_url = ('https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/'
               'molnet_publish/qm9.zip')
    raw_url2 = 'https://ndownloader.figshare.com/files/3195404'
    processed_url = 'https://data.pyg.org/datasets/qm9_v3.zip'

    def __init__(self, root, split, feature_type="one_hot", update_atomrefs=True, torchmd_net_split=True):
        assert feature_type in ["one_hot", "cormorant", "gilmer"], "Please use valid features"
        assert split in ["train", "valid", "test"]
        self.split = split
        self.feature_type = feature_type
        self.root = osp.abspath(root)
        self.update_atomrefs = update_atomrefs
        self.torchmd_net_split = torchmd_net_split
        
        super().__init__(self.root)
        self.data, self.slices = torch.load(self.processed_paths[0])


    def calc_stats(self, target):
        y = torch.cat([self.get(i).y for i in range(len(self))], dim=0)
        y = y[:, target]
        mean = float(torch.mean(y))
        mad = float(torch.mean(torch.abs(y - mean)))
        #ys = np.array([data.y.item() for data in self])
        #mean = np.mean(ys)
        #mad = np.mean(np.abs(ys - mean))
        return mean, mad
    

    def mean(self, target: int) -> float:
        y = torch.cat([self.get(i).y for i in range(len(self))], dim=0)
        return float(y[:, target].mean())


    def std(self, target: int) -> float:
        y = torch.cat([self.get(i).y for i in range(len(self))], dim=0)
        return float(y[:, target].std())


    def atomref(self, target) -> Optional[torch.Tensor]:
        if target in atomrefs:
            out = torch.zeros(100)
            out[torch.tensor([1, 6, 7, 8, 9])] = torch.tensor(atomrefs[target])
            return out.view(-1, 1)
        return None


    @property
    def raw_file_names(self) -> List[str]:
        try:
            import rdkit  # noqa
            return ['gdb9.sdf', 'gdb9.sdf.csv', 'uncharacterized.txt']
        except ImportError:
            return ['qm9_v3.pt']


    @property
    def processed_file_names(self) -> str:
        #return "_".join([self.split, str(np.round(self.radius, 2)), self.feature_type]) + '.pt'
        return "_".join([self.split, self.feature_type]) + '.pt'


    def download(self):
        try:
            import rdkit  # noqa
            file_path = download_url(self.raw_url, self.raw_dir)
            extract_zip(file_path, self.raw_dir)
            os.unlink(file_path)

            file_path = download_url(self.raw_url2, self.raw_dir)
            os.rename(osp.join(self.raw_dir, '3195404'),
                      osp.join(self.raw_dir, 'uncharacterized.txt'))
        except ImportError:
            path = download_url(self.processed_url, self.raw_dir)
            extract_zip(path, self.raw_dir)
            os.unlink(path)


    def process(self):
        try:
            import rdkit
            from rdkit import Chem
            from rdkit.Chem.rdchem import HybridizationType
            from rdkit.Chem.rdchem import BondType as BT
            from rdkit import RDLogger
            RDLogger.DisableLog('rdApp.*')
        except ImportError:
            assert False, "Install rdkit-pypi"
            
        types = {'H': 0, 'C': 1, 'N': 2, 'O': 3, 'F': 4}
        bonds = {BT.SINGLE: 0, BT.DOUBLE: 1, BT.TRIPLE: 2, BT.AROMATIC: 3}

        with open(self.raw_paths[1], 'r') as f:
            target = f.read().split('\n')[1:-1]
            target = [[float(x) for x in line.split(',')[1:20]]
                      for line in target]
            target = torch.tensor(target, dtype=torch.float)
            target = torch.cat([target[:, 3:], target[:, :3]], dim=-1)
            target = target * conversion.view(1, -1)

        with open(self.raw_paths[2], 'r') as f:
            skip = [int(x.split()[0]) - 1 for x in f.read().split('\n')[9:-2]]

        suppl = Chem.SDMolSupplier(self.raw_paths[0], removeHs=False,
                                   sanitize=False)
        data_list = []

        Nmols = len(suppl) - len(skip)
        Ntrain = 100000
        Ntest = int(0.1*Nmols)
        Nvalid = Nmols - (Ntrain + Ntest)

        np.random.seed(0)
        data_perm = np.random.permutation(Nmols)
        
        if self.torchmd_net_split:
            Ntrain = 110000
            Nvalid =  10000
            Ntest = Nmols - (Ntrain + Nvalid)
            data_perm = np.random.default_rng(1).permutation(Nmols)

        train, valid, test = np.split(data_perm, [Ntrain, Ntrain+Nvalid])
        indices = {"train": train, "valid": valid, "test": test}

        np.savez(os.path.join(self.root, 'splits.npz'), idx_train=train, idx_valid=valid, idx_test=test)

        # Add a second index to align with cormorant splits.
        j = 0
        for i, mol in enumerate(tqdm(suppl)):
            if i in skip:
                continue
            if j not in indices[self.split]:
                j += 1
                continue
            j += 1

            N = mol.GetNumAtoms()

            pos = suppl.GetItemText(i).split('\n')[4:4 + N]
            pos = [[float(x) for x in line.split()[:3]] for line in pos]
            pos = torch.tensor(pos, dtype=torch.float)

            #edge_index = radius_graph(pos, r=self.radius, loop=False)
            
            # build pair-wise edge graphs
            num_nodes = pos.shape[0]
            node_index = torch.tensor([i for i in range(num_nodes)])
            edge_d_dst_index = torch.repeat_interleave(node_index, repeats=num_nodes)
            edge_d_src_index = node_index.repeat(num_nodes)
            edge_d_attr = pos[edge_d_dst_index] - pos[edge_d_src_index]
            edge_d_attr = edge_d_attr.norm(dim=1, p=2)
            edge_d_dst_index = edge_d_dst_index.view(1, -1)
            edge_d_src_index = edge_d_src_index.view(1, -1)
            edge_d_index = torch.cat((edge_d_dst_index, edge_d_src_index), dim=0)
            
            type_idx = []
            atomic_number = []
            aromatic = []
            sp = []
            sp2 = []
            sp3 = []
            num_hs = []
            for atom in mol.GetAtoms():
                type_idx.append(types[atom.GetSymbol()])
                atomic_number.append(atom.GetAtomicNum())
                aromatic.append(1 if atom.GetIsAromatic() else 0)
                hybridization = atom.GetHybridization()
                sp.append(1 if hybridization == HybridizationType.SP else 0)
                sp2.append(1 if hybridization == HybridizationType.SP2 else 0)
                sp3.append(1 if hybridization == HybridizationType.SP3 else 0)

            z = torch.tensor(atomic_number, dtype=torch.long)

            # from torch geometric
            row, col, edge_type = [], [], []
            for bond in mol.GetBonds():
                start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                row += [start, end]
                col += [end, start]
                edge_type += 2 * [bonds[bond.GetBondType()]]

            edge_index = torch.tensor([row, col], dtype=torch.long)
            edge_type = torch.tensor(edge_type, dtype=torch.long)
            edge_attr = F.one_hot(edge_type,
                                  num_classes=len(bonds)).to(torch.float)

            perm = (edge_index[0] * N + edge_index[1]).argsort()
            edge_index = edge_index[:, perm]
            edge_type = edge_type[perm]
            edge_attr = edge_attr[perm]

            if self.feature_type == "one_hot":
                x = F.one_hot(torch.tensor(type_idx), num_classes=len(types)).float()
            elif self.feature_type == "cormorant":
                one_hot = F.one_hot(torch.tensor(type_idx), num_classes=len(types))
                x = get_cormorant_features(one_hot, z, 2, z.max())
            elif self.feature_type == "gilmer":
                row, col = edge_index
                hs = (z == 1).to(torch.float)
                num_hs = scatter(hs[row], col, dim_size=N).tolist()

                x1 = F.one_hot(torch.tensor(type_idx), num_classes=len(types))
                x2 = torch.tensor([atomic_number, aromatic, sp, sp2, sp3, num_hs],
                                  dtype=torch.float).t().contiguous()
                x = torch.cat([x1.to(torch.float), x2], dim=-1)

            y = target[i].unsqueeze(0)
            name = mol.GetProp('_Name')
            
            if self.update_atomrefs:
                node_atom = z.new_tensor([-1, 0, -1, -1, -1, -1, 1, 2, 3, 4])[z]
                atomrefs_value = atomrefs_tensor[node_atom]
                atomrefs_value = torch.sum(atomrefs_value, dim=0, keepdim=True)
                y = y - atomrefs_value                

            data = Data(x=x, pos=pos, z=z, edge_index=edge_index, 
                edge_attr=edge_attr, y=y, name=name, index=i, 
                edge_d_index=edge_d_index, edge_d_attr=edge_d_attr)
            data_list.append(data)
            
        torch.save(self.collate(data_list), self.processed_paths[0])


def get_cormorant_features(one_hot, charges, charge_power, charge_scale):
    """ Create input features as described in section 7.3 of https://arxiv.org/pdf/1906.04015.pdf """
    charge_tensor = (charges.unsqueeze(-1) / charge_scale).pow(
        torch.arange(charge_power + 1., dtype=torch.float32))
    charge_tensor = charge_tensor.view(charges.shape + (1, charge_power + 1))
    atom_scalars = (one_hot.unsqueeze(-1) * charge_tensor).view(charges.shape[:2] + (-1,))
    return atom_scalars


if __name__ == "__main__":
    
    from torch_geometric.loader import DataLoader
    import matplotlib.pyplot as plt
    
    #dataset = QM9("temp", "valid", feature_type="one_hot")
    #print("length", len(dataset))
    #dataloader = DataLoader(dataset, batch_size=4)
    
    '''
    _target = 1
    
    dataset = QM9("test_atom_ref/with_atomrefs", "test", feature_type="one_hot", update_atomrefs=True)
    mean = dataset.mean(_target)
    _, std = dataset.calc_stats(_target)
    
    dataset_original = QM9("test_atom_ref/without_atomrefs", "test", feature_type="one_hot", update_atomrefs=False)
    
    for i in range(12):
        mean = dataset.mean(i)
        std = dataset.std(i)
        
        mean_original = dataset_original.mean(i)
        std_original = dataset_original.std(i)
        
        print('Target: {}, mean diff = {}, std diff = {}'.format(i, 
            mean - mean_original, std - std_original))
    '''

    dataset = QM9("test_torchmd_net_splits", "train", feature_type="one_hot", update_atomrefs=True, torchmd_net_split=True)