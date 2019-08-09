import main3
import multiprocessing
import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument("--read-data", action="store_true")
parser.add_argument("--write-data", action="store_true")
parser.add_argument("--batches", type=int, default=100)
parser.add_argument("--n-players", type=int, default=1024)

args = parser.parse_args()

main3.batches = args.batches
main3.n_players = args.n_players
main3.should_read_training_data = args.read_data
main3.should_write_training_data = True

master_file = "savedata.json"
filenames = [f"savedata0{i}.json" for i in range(multiprocessing.cpu_count())]

with multiprocessing.Pool(multiprocessing.cpu_count()) as p:
    results = [p.apply_async(main3.main, (master_file, filename)) for filename in filenames]
    print([res.get(timeout=120) for res in results])


data = {
    "training_data":[]
}
for file in filenames:
    with open(file, "r") as fd:
        data_part = json.load(fd)
        print(f"extending training data with {len(data_part['training_data'])} parameter sets")
        data["training_data"].extend(data_part["training_data"])

print(f"total parameter sets = {len(data['training_data'])}")

with open(master_file, "w") as fd:
    json.dump(data, fd)
