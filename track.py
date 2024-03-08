from concurrent.futures import ThreadPoolExecutor, wait
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import polars as pl
from tqdm import tqdm

from sort import Sort


def write_video(frames, name):
    writer = cv2.VideoWriter(
        name,
        cv2.VideoWriter_fourcc(*"mp4v"),  # type: ignore
        5,  # FPS
        (1280, 720),
        True,
    )
    for f in frames:
        writer.write(cv2.resize(f, (1280, 720), interpolation=cv2.INTER_AREA))

    writer.release()


def tracking(df, max_age=5, min_hits=1, iou_threshold=0.2):
    sort_tracker = Sort(max_age=max_age, min_hits=min_hits, iou_threshold=iou_threshold)
    track_results = []
    for row in tqdm(df.iter_rows(named=True), total=len(df)):
        boxes = np.array(row["bbox"]) if len(row["bbox"]) > 0 else np.empty((0, 5))
        tracks = sort_tracker.update(boxes).astype(np.int32)
        img = cv2.imread(row["image"])
        tracks[:, [1, 3]] = np.clip(tracks[:, [1, 3]], 0, img.shape[0])
        tracks[:, [0, 2]] = np.clip(tracks[:, [0, 2]], 0, img.shape[1])
        for track in tracks:
            x1, y1, x2, y2, id = track
            track_results.append(
                {
                    "image": row["image"],
                    "track id": id,
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                }
            )
    track_df = pl.from_dicts(track_results)
    return track_df


def save_cropped(df, cam, output_dir):
    track_df = tracking(df, 5, 1, 0.3)
    trackid_group = track_df.group_by("track id").len()
    filtered_ids = (
        trackid_group.filter(pl.col("len") > 100).get_column("track id").to_list()
    )
    filtered_track_df = (
        track_df.filter(pl.col("track id").is_in(filtered_ids))
        .group_by("image")
        .agg(pl.struct("x1", "y1", "x2", "y2", "track id").alias("bbox"))
        .sort("image")
    )

    for row in tqdm(
        filtered_track_df.iter_rows(named=True), total=len(filtered_track_df)
    ):
        img = cv2.imread(row["image"])
        for track in row["bbox"]:
            filename = Path(row["image"]).name
            p = output_dir / f"{cam:02d}-{int(track['track id']):04d}"
            if not p.is_dir():
                ref_img = np.copy(img)
                cv2.rectangle(
                    ref_img,
                    (track["x1"], track["y1"]),
                    (track["x2"], track["y2"]),
                    (0, 255, 0),
                    2,
                )
                cv2.putText(
                    ref_img,
                    f"{int(track['track id']):04d}",
                    (track["x1"], track["y1"] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    2,
                    (0, 255, 0),
                )
                ref_p = output_dir / "refs"
                ref_p.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(
                    str(ref_p / f"{cam:02d}-{int(track['track id']):04d}.jpg"),
                    ref_img,
                )
            p.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(
                str(p / filename),
                img[track["y1"] : track["y2"], track["x1"] : track["x2"]],
            )


if __name__ == "__main__":
    df = pl.read_parquet("/media/sdb/cow-fix/boxlist_trt_nms09.parquet")
    df = (
        df.lazy()
        .with_columns(
            # pl.col("image").str.split("/").list.get(5).cast(pl.Int32).alias("cam"),
            pl.col("image")
            .str.split("/")
            .list.get(6)
            .str.slice(0, 19)
            .str.to_datetime("%Y%m%d-%H%M%S%.3f", time_unit="ms")
            .alias("date"),
        )
        .sort("cam", "image", "date")
        .filter(pl.col("date") <= datetime(2024, 2, 20, 10, 0, 0))
        .explode("bbox")
        .drop_nulls()
        .with_columns(pl.col("bbox").list.concat([1]))
        .group_by("cam", "image")
        .agg("bbox")
        .sort("cam", "image")
        .collect()
    )
    print(df)

    output_dir = Path("/media/sda/dataset/fix-0203-track-100/")
    cam_list = df.get_column("cam").unique().to_list()
    # for cam in [2]:
    #     save_cropped(
    #         df.filter(pl.col("cam") == cam).sort("image").head(1000), cam, output_dir
    #     )
    # assert False

    executor = ThreadPoolExecutor(16)
    jobs = [
        executor.submit(
            save_cropped,
            df.filter(pl.col("cam") == cam).sort("image").head(1000),
            cam,
            output_dir,
        )
        for cam in cam_list
    ]
    wait(jobs)

    # for t in np.arange(1, 20, 1):
    #     track_df = tracking(df.filter(pl.col("cam") == cam).head(1000), cam, t, 1, 0.3)
    #     trackid_group = track_df.group_by("track id").len()
    #     filtered_ids = (
    #         trackid_group.filter(pl.col("len") > 300).get_column("track id").to_list()
    #     )
    #     print(
    #         f"max_age: {t}, track num: {len(trackid_group)}, filtered track num: {len(filtered_ids)}"
    #     )
