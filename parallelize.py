import main as main
import multiprocessing
import argparse
import json

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--readfile", type=str, default=None, help="file to read from")
    parser.add_argument(
        "--writefile", type=str, default=None, help="file to write from"
    )
    parser.add_argument("--batches", type=int, default=100)
    parser.add_argument("--wait", type=int, default=180)
    parser.add_argument("--n-players", type=int, default=256)
    N_THREADS = 22

    args = parser.parse_args()

    main.batches = args.batches
    main.n_players = args.n_players
    main.HEADLESS = True
    main.should_read_training_data = args.readfile is not None
    main.should_write_training_data = True

    master_file = "savedata/savedata.json" if args.readfile is None else args.readfile
    filenames = [
        f"savedata/savedata{i}.json"
        if args.writefile is None
        else f"{args.writefile}{i}.json"
        for i in range(N_THREADS)
    ]

    with multiprocessing.Pool(N_THREADS) as p:

        results = []
        for filename in filenames:
            new_args = main.parser.parse_args()
            new_args.readfile = master_file
            new_args.writefile = filename
            new_args.headless = True
            results.append(p.apply_async(main.main, ([new_args])))
        print([res.get(timeout=args.wait) for res in results])

    data = {"training_data": []}
    for file in filenames:
        with open(file, "r") as fd:
            data_part = json.load(fd)
            print(
                f"extending training data with {len(data_part['training_data'])} parameter sets"
            )
            data["training_data"].extend(data_part["training_data"])

    print(f"total parameter sets = {len(data['training_data'])}")

    with open(master_file, "w") as fd:
        json.dump(data, fd)
