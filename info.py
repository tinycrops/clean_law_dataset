from datasets import load_dataset

dataset = load_dataset("the-ride-never-ends/american_law", split="train", streaming=True)

# Get the first batch to inspect column names
first_batch = next(iter(dataset))
print(first_batch.keys())

# Print the first 5 lines
dataset_iter = iter(dataset)
for _ in range(100):
    print(next(dataset_iter))
