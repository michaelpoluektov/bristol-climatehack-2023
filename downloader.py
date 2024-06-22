import traceback

from huggingface_hub import snapshot_download, HfFileSystem


def main() -> None:
    datasets = [
        file
        for file in HfFileSystem().ls("datasets/climatehackai/climatehackai-2023")
        if file["type"] == "directory"
    ]

    print("Available datasets:")
    for index, dataset in enumerate(datasets):
        print(f'{index} - {dataset["name"]}')
    print("Select which datasets to download")
    print("e.g. > 1 2")
    user_input = input("> ")
    selected_datasets = [datasets[int(i)] for i in user_input.split()]

    print(f'You have selected: {f", ".join(d["name"] for d in selected_datasets)}')

    for _ in range(5):
        try:
            for dataset in selected_datasets:
                snapshot_download(
                    repo_id="climatehackai/climatehackai-2023",
                    repo_type="dataset",
                    local_dir="dataset/",
                    local_dir_use_symlinks=True,
                    allow_patterns=[f"{dataset['name'].split('/')[-1]}/*"],
                    resume_download=True,
                    max_workers=4,
                )
        except Exception:
            traceback.print_exc()
            print("Trying again...")
            continue


if __name__ == "__main__":
    main()
