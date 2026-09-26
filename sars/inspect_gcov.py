import earthaccess
import h5py

print("Logging in to NASA Earthdata...")
earthaccess.login()

print("Searching for NISAR GCOV products...")

results = earthaccess.search_data(
    short_name="NISAR_L2_GCOV_PROVISIONAL_V1",
    count=3
)

if len(results) < 3:
    raise RuntimeError(f"Expected 3 results, got {len(results)}")

# Result 3 from our previous search was ~606 MB.
selected = results[2]

url = selected.data_links()[0]

print("\nSelected product:")
print(url)

print("\nOpening authenticated remote file...")

files = earthaccess.open(
    selected.data_links(),
    provider="ASF",
    pqdm_kwargs={"n_jobs": 1}
)

remote_file = files[0]

print("\nRemote file opened:")
print(remote_file)

print("\nInspecting HDF5 structure...")

with h5py.File(remote_file, "r") as h5:

    def show(name, obj):
        if isinstance(obj, h5py.Dataset):
            print(f"DATASET  {name}  shape={obj.shape} dtype={obj.dtype}")
        else:
            print(f"GROUP    {name}")

    h5.visititems(show)
