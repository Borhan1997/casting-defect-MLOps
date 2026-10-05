from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image

# For Labels : 1 = def_front (defective), 0 = ok_front (OK)

class CastingDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.samples = []
         
        for cls in ["def_front", "ok_front"]:
            cls_dir = self.root_dir / cls
            for im in cls_dir.glob("*"):
                self.samples.append((im, 1 if cls == "def_front" else 0))
        pass

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # 1. get the (filepath, label) at position idx from self.samples
        path, label = self.samples[idx]
        
        # 2. open the image from filepath
        img = Image.open(path).convert("RGB")

        # 3. apply self.transform if it's set
        if self.transform:
            img = self.transform(img)

        # 4. return (image_tensor, label)
        return (img, label)