from argparse import ArgumentParser
import json
from pathlib import Path

import tqdm

from utils.retriever_manager import get_retriever
from utils import load_csv_data, get_test_question, get_image, remove_list_duplicates


def eval_recall(candidates, ground_truth, top_ks=None):
    if top_ks is None:
        top_ks = [1, 5, 10, 20, 100]

    recall = {k: 0 for k in top_ks}
    for k in top_ks:
        if ground_truth in candidates[:k]:
            recall[k] = 1
    return recall


def run_test(
    sample_file_path: str,
    query_expansion_path: str,
    top_ks: list,
    retrieval_top_k: int,
    alpha: float = 0.5,
    **kwargs
):
    sample_list, sample_header = load_csv_data(sample_file_path)

    query_expansion_file = Path(query_expansion_path)
    use_original_questions = False
    print(f"Loading query expansion data from: {query_expansion_path}")
    if query_expansion_file.exists():
        with open(query_expansion_file, "r") as f:
            query_expansion_data = json.load(f)
        print(f"Loaded {len(query_expansion_data)} query expansion entries")
    else:
        print(
            f"Warning: query expansion file not found: {query_expansion_path}. "
            "Using original questions from sample file."
        )
        query_expansion_data = {}
        use_original_questions = True

    print("Initializing retriever client...")
    retriever = get_retriever(
        service_url=kwargs["retriever_service_url"],
        max_retries=kwargs["max_retries"],
    )
    print("Retriever client initialized successfully")

    recalls = {k: 0 for k in top_ks}
    retrieval_result = {}

    for it, _ in tqdm.tqdm(enumerate(sample_list), desc="Test Retrieval with Service"):
        example = get_test_question(it, sample_list, sample_header)

        ground_truth = example["wikipedia_url"]
        image_path = get_image(
            example["dataset_image_ids"].split("|")[0],
            example["dataset_name"],
        )

        if example["dataset_name"] == "infoseek":
            data_id = example["data_id"]
        else:
            data_id = "E-VQA_{}".format(it)

        if use_original_questions:
            query = example["question"]
        elif data_id in query_expansion_data:
            query = query_expansion_data[data_id].get("query", "")
        else:
            print(f"Warning: {data_id} not found in query expansion data")
            query = example["question"]

        top_k = retriever.search_by_path(
            img_path=image_path,
            query=query,
            top_k=retrieval_top_k,
            alpha=alpha,
        )

        top_k_wiki = remove_list_duplicates(top_k)
        if kwargs["save_result_path"] != "None":
            retrieval_result[data_id] = {
                "retrieved_entities": [{"url": url} for url in top_k_wiki[:20]]
            }

        recall = eval_recall(top_k_wiki, ground_truth, top_ks)
        for k in top_ks:
            recalls[k] += recall[k]

    for k in top_ks:
        print("Avg Recall@{}: ".format(k), recalls[k] / (it + 1))

    if kwargs["save_result_path"] != "None":
        save_result_path = Path(kwargs["save_result_path"])
        save_result_path.parent.mkdir(parents=True, exist_ok=True)
        print("Save retrieval result to: ", kwargs["save_result_path"])
        with open(save_result_path, "w") as f:
            json.dump(retrieval_result, f, indent=4)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--sample_file", type=str, required=True)
    parser.add_argument(
        "--query_expansion",
        type=str,
        required=True,
        help="Path to query expansion JSON file",
    )
    parser.add_argument(
        "--retriever_service_url",
        type=str,
        default="http://localhost:5678",
        help="URL of the retriever service",
    )
    parser.add_argument(
        "--top_ks",
        type=str,
        default="1,5,10,20,100",
        help="comma separated list of top k values, e.g. 1,5,10,20,100",
    )
    parser.add_argument("--retrieval_top_k", type=int, default=20)
    parser.add_argument(
        "--alpha", type=float, default=0.5,
        help="Weight for visual embedding (0-1), text weight is (1-alpha)",
    )
    parser.add_argument(
        "--save_result_path",
        type=str,
        default="None",
        help="Path to save retrieval result",
    )
    parser.add_argument(
        "--max_retries",
        type=int,
        default=10,
        help="Maximum retries while waiting for the retriever service",
    )

    args = parser.parse_args()

    test_config = {
        "sample_file_path": args.sample_file,
        "query_expansion_path": args.query_expansion,
        "retriever_service_url": args.retriever_service_url,
        "top_ks": [int(k) for k in args.top_ks.split(",")],
        "retrieval_top_k": args.retrieval_top_k,
        "alpha": args.alpha,
        "save_result_path": args.save_result_path,
        "max_retries": args.max_retries,
    }
    print("Test config: ", test_config)
    run_test(**test_config)
