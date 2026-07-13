"""Convert FastVideo Parquet dataset to WebDataset tar format for FastGen."""
import pandas as pd, numpy as np, torch, webdataset as wds
import os, io, glob, sys, time

SRC = "/data/datasets/Wan-Syn_77x448x832_600k"
DST = "/data/datasets/wan_syn_wds"
SAMPLES_PER_SHARD = 200

os.makedirs(DST, exist_ok=True)

# Negative prompt embedding (zero placeholder)
neg_emb = np.zeros((1, 4096), dtype=np.float32)
np.save(os.path.join(DST, "neg_prompt_emb.npy"), neg_emb)

parquet_files = sorted(glob.glob(os.path.join(SRC, "**", "*.parquet"), recursive=True))
print(f"Found {len(parquet_files)} parquet files")

sample_count = 0
shard_pattern = os.path.join(DST, "shard-%06d.tar")
sink = wds.ShardWriter(shard_pattern, maxcount=SAMPLES_PER_SHARD)
t0 = time.time()

for fi, pf in enumerate(parquet_files):
    try:
        df = pd.read_parquet(pf)
    except Exception as e:
        print(f"SKIP {pf}: {e}")
        continue

    for _, row in df.iterrows():
        try:
            shape = tuple(row["vae_latent_shape"])
            dtype = np.dtype(str(row["vae_latent_dtype"]))
            latent = torch.from_numpy(
                np.frombuffer(row["vae_latent_bytes"], dtype=dtype).reshape(shape).copy()
            )

            shape_te = tuple(row["text_embedding_shape"])
            dtype_te = np.dtype(str(row["text_embedding_dtype"]))
            te = torch.from_numpy(
                np.frombuffer(row["text_embedding_bytes"], dtype=dtype_te).reshape(shape_te).copy()
            )

            buf_lat = io.BytesIO()
            torch.save(latent, buf_lat)
            buf_te = io.BytesIO()
            torch.save(te, buf_te)

            sink.write({
                "__key__": f"sample_{sample_count:06d}",
                "latent.pth": buf_lat.getvalue(),
                "txt_emb.pth": buf_te.getvalue(),
            })
            sample_count += 1
        except Exception as e:
            print(f"SKIP sample in {pf}: {e}")
            continue

    if (fi + 1) % 500 == 0:
        elapsed = time.time() - t0
        rate = sample_count / elapsed
        print(f"[{fi+1}/{len(parquet_files)}] {sample_count} samples, {elapsed:.0f}s, {rate:.1f} samples/s")
        sys.stdout.flush()

sink.close()
elapsed = time.time() - t0
print(f"\nDone! {sample_count} samples in {elapsed:.0f}s")
print(f"Output: {DST}")
shards = glob.glob(os.path.join(DST, "*.tar"))
print(f"Shards: {len(shards)}")
