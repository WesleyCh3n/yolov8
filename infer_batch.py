from datetime import datetime

import polars as pl
from tqdm import tqdm

import trt

# from ultralytics import YOLO
# ptmodel = YOLO("/home/wesley/GitHub/lirm/model/detect_13k.pt", verbose=False)
# ptresult = (
#     ptmodel(row["image"], verbose=False)[0]
#     .boxes.cpu()
#     .xyxy.numpy()
#     .astype(np.int32)
# )


conf = 0.25
nms_threshold = 0.9
trtmodel = trt.YOLOv8(
    "/home/wesley/GitHub/lirm/model/detect_13k.trt", 64, conf, nms_threshold
)

df = pl.read_parquet("/media/sdb/cow-fix/boxlist_detect.parquet")
df = (
    df.lazy()
    .with_columns(
        pl.col("image").str.split("/").list.get(5).cast(pl.Int32).alias("cam"),
        pl.col("image")
        .str.split("/")
        .list.get(6)
        .str.slice(0, 19)
        .str.to_datetime("%Y%m%d-%H%M%S%.3f", time_unit="ms")
        .alias("date"),
    )
    .filter(pl.col("date") <= datetime(2024, 2, 20, 10, 0, 0))
    .sort("cam", "image", "date")
    .collect()
)
print(df)

cam_list = df.get_column("cam").unique().to_list()
rows = []
for cam in tqdm(cam_list):
    camdf = df.filter(pl.col("cam") == cam)

    for imgs in trt.batch(camdf.get_column("image").to_list(), 512):
        results = trtmodel(imgs)
        for img, result in zip(imgs, results):
            rows.append({"image": img, "bbox": result, "cam": cam, "num": len(result)})
df = pl.from_dicts(rows)
print(df)
df.write_parquet("/media/sdb/cow-fix/boxlist_trt_nms09.parquet")
