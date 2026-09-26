import earthaccess

print("Logging in to NASA Earthdata...")
earthaccess.login()

print("\nSearching for NISAR GCOV products...")

results = earthaccess.search_data(
    short_name="NISAR_L2_GCOV_PROVISIONAL_V1",
    count=3
)

print(f"\nFound {len(results)} result(s).\n")

for i, result in enumerate(results, 1):
    print("=" * 70)
    print(f"RESULT {i}")
    print("=" * 70)
    print(result)
