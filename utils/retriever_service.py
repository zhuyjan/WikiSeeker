"""Standalone retriever service."""

from argparse import ArgumentParser
from pathlib import Path
import sys
import traceback

from flask import Flask, jsonify, request
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from retriever import WikiRetriever

app = Flask(__name__)
retriever = None


def parse_gpu_ids(gpu_ids):
    if gpu_ids is None or gpu_ids.strip().lower() in {"", "none"}:
        return None
    return [int(gpu_id.strip()) for gpu_id in gpu_ids.split(",")]


def initialize_retriever(args):
    """Initialize the WikiRetriever."""
    global retriever
    print("[INFO] Initializing WikiRetriever...")
    retriever = WikiRetriever(
        device=f"cuda:{args.vis_model_device}",
        model=args.retriever_vit,
        text_model_path=args.text_model_path,
        text_model_device=args.text_model_device,
    )
    retriever.load_knowledge_base(args.knowledge_base)

    faiss_gpu_ids = parse_gpu_ids(args.faiss_gpu_ids)
    if faiss_gpu_ids is not None and len(faiss_gpu_ids) > 1:
        retriever.load_faiss_index_multi_gpu(args.faiss_index, gpu_ids=faiss_gpu_ids)
    elif faiss_gpu_ids is not None and len(faiss_gpu_ids) == 1:
        retriever.load_faiss_index(args.faiss_index, gpu_id=faiss_gpu_ids[0])
    else:
        retriever.load_faiss_index(args.faiss_index, gpu_id=0)

    print("[INFO] WikiRetriever initialized successfully")


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({"status": "healthy", "retriever_loaded": retriever is not None})


@app.route('/search', methods=['POST'])
def search():
    """Search endpoint."""
    try:
        data = request.json
        query = data['query']
        img_path = data['img_path']
        top_k = data.get('top_k', 200)
        alpha = data.get('alpha', 0.6)

        image = Image.open(img_path)
        results = retriever.search(image, query, alpha=alpha, top_k=top_k)

        return jsonify({"results": results, "success": True})
    except Exception as e:
        print(f"[ERROR] Search failed: {e}")
        traceback.print_exc()
        return jsonify({"error": str(e), "success": False}), 500


def build_parser():
    parser = ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5678)
    parser.add_argument("--knowledge_base", type=str, required=True)
    parser.add_argument("--faiss_index", type=str, required=True)
    parser.add_argument("--retriever_vit", type=str, default="eva-clip")
    parser.add_argument(
        "--text_model_path",
        type=str,
        default="/data/share/model/Qwen/Qwen3-Embedding-0.6B",
    )
    parser.add_argument("--faiss_gpu_ids", type=str, default="0")
    parser.add_argument("--vis_model_device", type=int, default=0)
    parser.add_argument("--text_model_device", type=int, default=0)
    return parser


if __name__ == '__main__':
    service_args = build_parser().parse_args()
    initialize_retriever(service_args)
    app.run(host=service_args.host, port=service_args.port, threaded=True)
