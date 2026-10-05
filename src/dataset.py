from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image

# For Labels : 1 = def_front (defective), 0 = ok_front (OK)
CLASSES = {"def_front": 1, "ok_front": 0}
MANIFEST_PATH = Path("data/val_manifest.txt")


def read_manifest(path=MANIFEST_PATH) -> set[str]:
    """Return the validation keys, e.g. {'def_front/xyz.jpg', ...}."""
    lines = Path(path).read_text().splitlines()
    return {line.strip() for line in lines if line.strip()}

class CastingDataset(Dataset):
    def __init__(self, root_dir, transform=None, include=None, exclude=None):
        if not isinstance(root_dir, (list, tuple)):
            root_dir = [root_dir]
        self.root_dirs = [Path(d) for d in root_dir]
        self.include = include
        self.exclude = exclude
        self.transform = transform
        self.samples = []
         
        for folder in self.root_dirs:
            for cls, label in CLASSES.items():
                for im in sorted((folder / cls).glob("*")):
                    if not im.is_file():
                        continue
                    key = f"{cls}/{im.name}"
                    if self.include is not None and key not in self.include:
                        continue
                    if self.exclude is not None and key in self.exclude:
                        continue
                    self.samples.append((im, label))

        if not self.samples:
            raise ValueError(
                f"CastingDataset is empty: folders={[str(d) for d in self.root_dirs]}, "
                f"include={'set' if include is not None else 'None'}, "
                f"exclude={'set' if exclude is not None else 'None'}"
            )
        

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