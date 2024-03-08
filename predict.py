# import argparse
import json
from pathlib import Path

import cv2
import polars as pl
from pandas.core.dtypes.dtypes import np
from torch.utils.data import DataLoader, Dataset
from torchvision.io import read_image
from tqdm import tqdm
from ultralytics import YOLO

# parser = argparse.ArgumentParser()
# parser.add_argument(
#     "-i", "--input", type=Path, help="input filelist parquet", required=True
# )
# parser.add_argument("-o", "--output", type=Path, required=True)
# parser.add_argument("-m", "--model", type=Path, default=Path("./model/best.pt"))


class ImageData(Dataset):
    def __init__(self, files: list[str], transform=None):
        self.files = files
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        if self.transform:
            img = read_image(self.files[index])
            return {"path": self.files[index], "img": self.transform(img)}
        else:
            img = cv2.imread(self.files[index]).astype(np.float32)
            img /= 255
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (640, 640), interpolation=cv2.INTER_AREA)
            img = np.moveaxis(img, 2, 0)
        return {"path": self.files[index], "img": img}


def main():
    # args = parser.parse_args()

    df = pl.read_parquet("/media/sdb/cow-fix/boxlist.parquet")
    df = (
        df.lazy()
        .with_columns(pl.col("cam").cast(pl.Int32))
        .sort("cam", "image")
        .collect()
    )

    output_dir = Path("/media/sda/dataset/fix-0203-track/")
    cam_list = df.group_by("cam").agg().get_column("cam").to_numpy()

    cam = 2
    paths = df.filter(pl.col("cam") == cam)["image"].to_numpy()
    cam_dir = output_dir / f"{cam:02d}"

    # data_set = ImageData(paths)
    # data_loader = DataLoader(dataset=data_set, batch_size=32, num_workers=16)

    model = YOLO("/home/wesley/GitHub/lirm/model/detect_13k.pt")
    for path in paths:
        # results = model(path)
        # img = results[0].plot()
        # cv2.imwrite(str(cam_dir / Path(path).name), img)
        results = model.track(path, persist=True)
        boxes = results[0].boxes.xyxy.cpu()
        track_ids = results[0].boxes.id.int().cpu().tolist()
        img = cv2.imread(path)
        for box, track_id in zip(boxes, track_ids):
            x1, y1, x2, y2 = box
            p = cam_dir / f"{track_id:04d}"
            p.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(
                str(p / Path(path).name),
                img[int(y1) : int(y2), int(x1) : int(x2)],
            )

    # for d in tqdm(data_loader):
    #     yolo_results = model(d["img"], verbose=False)
    #
    #     batch_result = []
    #     for p, r in zip(d["path"], yolo_results):
    #         obj = dict()
    #         obj["path"] = p
    #         obj["prediction"] = json.loads(r.tojson())
    #         batch_result.append(obj)
    #
    #     if df is None:
    #         df = pl.from_dicts(batch_result)
    #     else:
    #         df.extend(pl.from_dicts(batch_result))
    # if df is not None:
    #     df.write_parquet(args.output)


if __name__ == "__main__":
    main()
