import earthaccess
import h5py
import numpy as np
import matplotlib.pyplot as plt

SHORT_NAME = "NISAR_L2_GCOV_PROVISIONAL_V1"

DATASET = "science/LSAR/GCOV/grids/frequencyA/HHHH"
MASK = "science/LSAR/GCOV/grids/frequencyA/mask"
X_COORDS = "science/LSAR/GCOV/grids/frequencyA/xCoordinates"
Y_COORDS = "science/LSAR/GCOV/grids/frequencyA/yCoordinates"

# Small window: 2000 x 2000 pixels
ROW_START = 6000
ROW_END = 8000
COL_START = 5000
COL_END = 7000

print("Logging in to NASA Earthdata...")
earthaccess.login()

print("Searching for NISAR GCOV product...")

results = earthaccess.search_data(
    short_name=SHORT_NAME,
    count=3
)

if len(results) < 3:
    raise RuntimeError(f"Expected at least 3 results, got {len(results)}")

# Use the ~606 MB product we inspected earlier.
selected = results[2]

print("\nOpening authenticated remote file...")

files = earthaccess.open(
    selected.data_links(),
    provider="ASF",
    pqdm_kwargs={"n_jobs": 1}
)

remote_file = files[0]

print("Remote file ready.")
print("Reading only a 2000 x 2000 pixel window...")

with h5py.File(remote_file, "r") as h5:

    data = h5[DATASET][ROW_START:ROW_END, COL_START:COL_END]
    mask = h5[MASK][ROW_START:ROW_END, COL_START:COL_END]

    x = h5[X_COORDS][COL_START:COL_END]
    y = h5[Y_COORDS][ROW_START:ROW_END]

data = np.asarray(data, dtype=np.float32)
mask = np.asarray(mask)

# HHHH is power/backscatter.
# Convert positive values to dB.
valid = np.isfinite(data) & (data > 0) & (mask != 0)

db = np.full(data.shape, np.nan, dtype=np.float32)
db[valid] = 10.0 * np.log10(data[valid])

print("\nWindow shape:", data.shape)
print("Valid pixels:", int(np.count_nonzero(valid)))

if np.any(valid):
    print("Minimum dB:", float(np.nanmin(db)))
    print("Maximum dB:", float(np.nanmax(db)))
    print("Mean dB:", float(np.nanmean(db)))

# Save the small numerical product.
np.save("gcov_hhhh_2000x2000_db.npy", db)

# Visualization
plt.figure(figsize=(10, 8))

image = plt.imshow(
    db,
    cmap="gray",
    origin="upper",
    extent=[x[0], x[-1], y[-1], y[0]]
)

plt.colorbar(image, label="HHHH backscatter (dB)")
plt.xlabel("X coordinate")
plt.ylabel("Y coordinate")
plt.title("NISAR GCOV — HHHH Backscatter")

plt.tight_layout()
plt.savefig(
    "gcov_hhhh_2000x2000.png",
    dpi=150,
    bbox_inches="tight"
)

plt.show()

print("\nSaved:")
print("  gcov_hhhh_2000x2000.npy")
print("  gcov_hhhh_2000x2000.png")
