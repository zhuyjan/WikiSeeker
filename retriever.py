import os
import torch
import tqdm
import pickle
import json
from transformers import AutoModel, AutoProcessor, CLIPImageProcessor, AutoTokenizer
import faiss
import numpy as np
from faiss import write_index, read_index
import torch
import torch.nn.functional as F
import faiss.contrib.torch_utils


class KnowledgeBase:
    """Knowledge base for OMGM system.

    Returns:
        KnowledgeBase
    """

    def __len__(self):
        """Return the length of the knowledge base.

        Args:

        Returns:
            int
        """
        return len(self.knowledge_base)

    def __getitem__(self, index):
        """Return the knowledge base entry at the given index.

        Args:
            index (int): The index of the knowledge base entry to return.

        Returns:
            KnowledgeBaseEntry
        """
        return self.knowledge_base[index]

    def __init__(self, knowledge_base_path):
        """Initialize the KnowledgeBase class.

        Args:
            knowledge_base_path (str): The path to the knowledge base.
        """
        self.knowledge_base_path = knowledge_base_path
        self.knowledge_base = None

    def load_knowledge_base(self):
        """Load the knowledge base."""
        raise NotImplementedError


class WikipediaKnowledgeBase(KnowledgeBase):
    """Knowledge base for OMGM."""

    def __init__(self, knowledge_base_path):
        """Initialize the KnowledgeBase class.

        Args:
            knowledge_base_path (str): The path to the knowledge base.
        """
        super().__init__(knowledge_base_path)
        self.knowledge_base = []

    def load_knowledge_base_full(
        self, image_dict=None, scores_path=None, visual_attr=None
    ):
        """Load the knowledge base from multiple score files.

        Args:
            image_dict: The image dictionary to load.
            scores_path: The parent folder path to the vision similarity scores to load.
            visual_attr: The visual attribute dictionary to load. Deprecated in the current version.
        """
        try:
            with open(self.knowledge_base_path, "rb") as f:
                knowledge_base_dict = json.load(f)
        except:
            raise FileNotFoundError(
                "Knowledge base not found, which should be a url or path to a json file."
            )
        # image_dict and load_scores_path can't be both None and can't be both not None

        if visual_attr is not None:
            try:
                with open(visual_attr, "r") as f:
                    visual_attr_dict = json.load(f)
            except:
                raise FileNotFoundError(
                    "Visual Attr not found, which should be a url or path to a json file."
                )

        if scores_path is not None:
            # get the image scores for each entry
            # get all the *.pkl files in the scores_path
            print("Loading knowledge base score from {}.".format(scores_path))
            import glob

            score_files = glob.glob(scores_path + "/*.pkl")
            image_scores = {}
            for score_file in tqdm.tqdm(score_files):
                try:
                    with open(score_file, "rb") as f:
                        image_scores.update(pickle.load(f))
                except:
                    raise FileNotFoundError(
                        "Image scores not found, which should be a url or path to a pickle file."
                    )
            print("Loaded {} image scores.".format(len(image_scores)))
            for wiki_url, entry in knowledge_base_dict.items():
                wiki_entry = WikipediaKnowledgeBaseEntry(entry)
                for url in wiki_entry.image_urls:
                    if url in image_scores:
                        wiki_entry.score[url] = image_scores[url]
                self.knowledge_base.append(wiki_entry)
        else:
            for wiki_url, entry in knowledge_base_dict.items():
                wiki_entry = WikipediaKnowledgeBaseEntry(entry)
                self.knowledge_base.append(wiki_entry)

        print("Loaded knowledge base with {} entries.".format(len(self.knowledge_base)))
        return self.knowledge_base

    def load_knowledge_base(self, image_dict=None, scores_path=None, visual_attr=None):
        """Load the knowledge base.

        Args:
            image_dict: The image dictionary to load.
            scores_path: The path to the vision similarity score file to load.
            visual_attr: The visual attribute dictionary to load. Deprecated in the current version.
        """
        try:
            with open(self.knowledge_base_path, "rb") as f:
                knowledge_base_dict = json.load(f)
        except:
            raise FileNotFoundError(
                "Knowledge base not found, which should be a url or path to a json file."
            )
        # image_dict and load_scores_path can't be both None and can't be both not None
        if visual_attr is not None:
            # raise NotImplementedError
            try:
                with open(visual_attr, "r") as f:
                    visual_attr_dict = json.load(f)
            except:
                raise FileNotFoundError(
                    "Visual Attr not found, which should be a url or path to a json file."
                )

        if (
            scores_path is not None
        ):  # TODO: fix the knowledge base and visual_attr is None:
            # get the image scores for each entry
            print("Loading knowledge base score from {}.".format(scores_path))
            try:
                with open(scores_path, "rb") as f:
                    image_scores = pickle.load(f)
            except:
                raise FileNotFoundError(
                    "Image scores not found, which should be a url or path to a pickle file."
                )
            for wiki_url, entry in knowledge_base_dict.items():
                wiki_entry = WikipediaKnowledgeBaseEntry(entry)
                for url in wiki_entry.image_urls:
                    if url in image_scores:
                        wiki_entry.score[url] = image_scores[url]
                self.knowledge_base.append(wiki_entry)
        else:
            # print("Loading knowledge base without image scores.")
            for wiki_url, entry in knowledge_base_dict.items():
                wiki_entry = WikipediaKnowledgeBaseEntry(entry)
                self.knowledge_base.append(wiki_entry)

        print("Loaded knowledge base with {} entries.".format(len(self.knowledge_base)))
        return self.knowledge_base


class WikipediaKnowledgeBaseEntry:
    """Knowledge base entry for OMGM.

    Returns:
    """

    def __init__(self, entry_dict, visual_attr=None):
        """Initialize the KnowledgeBaseEntry class.

        Args:
            entry_dict: The dictionary containing the knowledge base entry.
            visual_attr: The visual attribute. Deprecated in the current version.

        Returns:
            KnowledgeBaseEntry
        """
        self.title = entry_dict["title"]
        self.url = entry_dict["url"]
        self.image_urls = entry_dict["image_urls"]
        self.image_reference_descriptions = entry_dict["image_reference_descriptions"]
        self.image_section_indices = entry_dict["image_section_indices"]
        self.section_titles = entry_dict["section_titles"]
        self.section_texts = entry_dict["section_texts"]
        self.image = {}
        self.score = {}
        self.visual_attr = visual_attr


class Retriever:
    """Retriever parent class for OMGM."""

    def __init__(self, model=None):
        """Initialize the Retriever class.

        Args:
            model: The model to use for retrieval.
        """
        self.model = model

    def load_knowledge_base(self, knowledge_base_path):
        """Load the knowledge base.

        Args:
            knowledge_base_path: The knowledge base to load.
        """
        raise NotImplementedError

    def retrieve_image(self, image):
        """Retrieve the image.

        Args:
            image: The image to retrieve.
        """
        raise NotImplementedError


class WikiRetriever(Retriever):
    """Wiki retriever with CLIP-based VIT."""

    def __init__(self, model="eva-clip", device="cpu", text_model_path='/data/share/model/Qwen/Qwen3-Embedding-0.6B', text_model_device=1):
        """Initialize the WikiRetriever class.

        Args:
            model: The model to use for retrieval. Should be 'clip' or 'eva-clip'.
            device: The device to use for retrieval.
            text_model_path: Path to Qwen3 text embedding model.
            text_model_device: GPU device ID for text model (default: 1).
        """
        super().__init__(model)
        self.model_type = model
        self.text_model_device = text_model_device
        if model == "clip":
            self.model = AutoModel.from_pretrained(
                "/data/share/model/openai/clip-vit-large-patch14",
                torch_dtype=torch.float16,
            )
            del self.model.text_projection
            del self.model.text_model  # avoiding OOM
            self.model.to("cuda").eval()
            self.processor = AutoProcessor.from_pretrained(
                "/data/share/model/openai/clip-vit-large-patch14"
            )
        elif model == "eva-clip":
            print('Loading EVA-CLIP...')
            self.model = AutoModel.from_pretrained(
                "/data/share/model/BAAI/EVA-CLIP-8B",
                torch_dtype=torch.float16,
                trust_remote_code=True,
            )
            del self.model.text_projection
            del self.model.text_model  # avoiding OOM
            self.model.to(device).eval()
            self.processor = CLIPImageProcessor.from_pretrained(
                "/data/share/model/openai/clip-vit-large-patch14"
            )
            print(f"Loading Qwen3 Embedding model from {text_model_path}")
            self.text_tokenizer = AutoTokenizer.from_pretrained(
                text_model_path, padding_side='left'
            )
            self.text_model = AutoModel.from_pretrained(
                text_model_path, device_map=f'cuda:{text_model_device}'
            ).eval()

        self.device = device
        self.model.to(device)
        self.knowledge_base = None
        self.faiss_gpu_ids = None
        self.faiss_gpu_id = None
        

    def load_knowledge_base(
        self, knowledge_base_path, image_dict=None, scores_path=None, visual_attr=None
    ):
        """Load the knowledge base.

        Args:
            knowledge_base_path: The knowledge base to load.
        """
        self.knowledge_base = WikipediaKnowledgeBase(knowledge_base_path)
        knowledge_base_list = self.knowledge_base.load_knowledge_base(
            image_dict=image_dict, scores_path=scores_path, visual_attr=visual_attr
        )
        return knowledge_base_list


    def save_knowledge_base_faiss(
        self,
        knowledge_base_path,
        image_dict=None,
        scores_path=None,
        visual_attr=None,
        save_path=None,
    ):
        """Save the knowledge base with faiss index.

        Args:
            knowledge_base_path: The knowledge base to load.
            image_dict: The image dictionary to load.
            scores_path: The path to the vision similarity score file to load.
            visual_attr: The visual attribute dictionary to load. Deprecated in the current version.
            save_path: The path to save the faiss index.
        """
        self.knowledge_base = WikipediaKnowledgeBase(knowledge_base_path)
        if scores_path[-4:] == ".pkl":
            print("Loading knowledge base from {}.".format(scores_path))
            self.knowledge_base.load_knowledge_base(
                image_dict=image_dict, scores_path=scores_path, visual_attr=visual_attr
            )
        else:
            print("Loading full knowledge base from {}.".format(scores_path))
            self.knowledge_base.load_knowledge_base_full(
                image_dict=image_dict, scores_path=scores_path, visual_attr=visual_attr
            )
        self.prepare_faiss_index()
        self.save_faiss_index(save_path)


    def save_faiss_index(self, save_index_path):
        """Save the faiss index.
        
        Args:
            save_index_path: The path to save the faiss index.
        """
        if save_index_path is not None:
            write_index(self.faiss_index, save_index_path + "kb_index.faiss")
            with open(os.path.join(save_index_path, "kb_index_ids.pkl"), "wb") as f:
                pickle.dump(self.faiss_index_ids, f)


    def load_faiss_index(self, load_index_path, gpu_id=1):
        """Load the faiss index with ids on a single GPU.
        
        Args:
            load_index_path: The path to load the faiss index.
            gpu_id: GPU device ID to load the index (default: 1).
        """
        if load_index_path is not None:
            print('Loading index...')
            self.faiss_index = read_index(os.path.join(load_index_path, "kb_index.faiss"))
            res = faiss.StandardGpuResources()
            self.faiss_index = faiss.index_cpu_to_gpu(res, gpu_id, self.faiss_index)
            self.faiss_gpu_id = gpu_id
            with open(os.path.join(load_index_path, "kb_index_ids.pkl"), "rb") as f:
                self.faiss_index_ids = pickle.load(f)
            print("Faiss index loaded with {} entries on GPU {}.".format(self.faiss_index.ntotal, gpu_id))


    def load_faiss_index_multi_gpu(self, load_index_path, gpu_ids=None):
        print('Loading index...')
        cpu_index = read_index(os.path.join(load_index_path, "kb_index.faiss"))
        # Load index IDs
        with open(os.path.join(load_index_path, "kb_index_ids.pkl"), "rb") as f:
            self.faiss_index_ids = pickle.load(f)
        ngpus = len(gpu_ids)
        print(f"Loading index to {ngpus} GPUs: {gpu_ids}")
        gpu_resources = [faiss.StandardGpuResources() for _ in range(ngpus)]
        co = faiss.GpuMultipleClonerOptions()
        co.shard = True
        co.useFloat16 = False
        # Load index to multiple GPUs using index_cpu_to_gpu_multiple_py
        vres = faiss.GpuResourcesVector()
        vdev = faiss.Int32Vector()
        for i in range(ngpus):
            vres.push_back(gpu_resources[i])
            vdev.push_back(gpu_ids[i])
        
        self.faiss_index = faiss.index_cpu_to_gpu_multiple(
            vres, vdev, cpu_index, co
        )

        self.faiss_gpu_ids = gpu_ids

        print(f"Faiss index loaded on {ngpus} GPUs (sharding mode) with {self.faiss_index.ntotal} entries.")


    def prepare_faiss_index(self):
        """Prepare the faiss index from scores in the knowledge base."""
        # use the knowledge base's score element to build the index
        # get the image scores for each entry
        scores = [
            score for entry in self.knowledge_base for score in entry.score.values()
        ]
        score_ids = [
            i
            for i in range(len(self.knowledge_base))
            for j in range(len(self.knowledge_base[i].score))
        ]
        
        # import ipdb; ipdb.set_trace()
        index = faiss.IndexFlatIP(scores[0].shape[0])
        # res = faiss.StandardGpuResources()
        # index = faiss.index_cpu_to_gpu(res, 0, index)
        np_scores = np.array(scores)
        np_scores = np_scores.astype(np.float32)
        faiss.normalize_L2(np_scores)
        index.add(np_scores)
        self.faiss_index = index
        self.faiss_index_ids = score_ids
        print("Faiss index built with {} entries.".format(index.ntotal))

        return

    
    def last_token_pool(self, last_hidden_states, attention_mask):
        """Extract last token pooling for Qwen3 embeddings."""
        left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
        if left_padding:
            return last_hidden_states[:, -1]
        else:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_states.shape[0]
            return last_hidden_states[
                torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths
            ]
    

    @torch.no_grad()
    def extract_text_embedding(self, text, max_length=8192):
        """Extract text embedding using Qwen3 model."""
        if self.text_model is None:
            self.load_text_model()
        
        batch_dict = self.text_tokenizer(
            [text],
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        batch_dict = batch_dict.to(f'cuda:{self.text_model_device}')
        
        outputs = self.text_model(**batch_dict)
        embeddings = self.last_token_pool(
            outputs.last_hidden_state, batch_dict['attention_mask']
        )
        embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings
    
    @torch.no_grad()
    def search(
        self, image, text, alpha=0.6, top_k=100, return_entry_list=False
    ):
        """
        Retrieve using hybrid embeddings (visual + text).
        
        Args:
            image: Input image
            text: Input text (caption + query)
            alpha: Weight for visual embedding (0-1). Text weight is (1-alpha)
            top_k: Number of top results to return
            return_entry_list: Whether to return entry list
            
        Returns:
            List of top k retrieved entries
        """
        # Extract visual embedding
        if self.model_type == "clip":
            inputs = (
                self.processor(images=image, return_tensors="pt")
                .pixel_values.to(self.device)
                .half()
            )
            image_score = self.model.get_image_features(inputs)
        elif self.model_type == "eva-clip":
            inputs = (
                self.processor(images=image, return_tensors="pt")
                .pixel_values.to(self.device)
                .half()
            )
            image_score = self.model.encode_image(inputs)
        
        # Normalize visual embedding
        visual_embed = F.normalize(image_score.float(), p=2, dim=1)
        
        # Extract and normalize text embedding
        text_embed = self.extract_text_embedding(text)
        
        # Combine embeddings: concat(alpha * visual, (1-alpha) * text)
        visual_weighted = alpha * visual_embed.to(text_embed.device)
        text_weighted = (1 - alpha) * text_embed
        hybrid_embed = torch.cat([visual_weighted, text_weighted], dim=1)
        
        # Normalize the final hybrid embedding
        query = F.normalize(hybrid_embed, p=2, dim=1)
        
        assert self.faiss_index and self.faiss_index_ids is not None
        
        # Multi-GPU mode: use CPU numpy array for search
        if self.faiss_gpu_ids is not None:
            query_cpu = query.detach().cpu().numpy().astype(np.float32)
            D, I = self.faiss_index.search(query_cpu, top_k)
            torch.cuda.synchronize(device=self.device)
            for gpu_id in self.faiss_gpu_ids:
                torch.cuda.synchronize(device=gpu_id)
        # Single GPU mode: use GPU tensor
        else:
            with torch.cuda.device(self.faiss_gpu_id):
                query_gpu = query.cuda(self.faiss_gpu_id)
                D, I = self.faiss_index.search(query_gpu, top_k)
            torch.cuda.synchronize(device=self.device)
            torch.cuda.synchronize(device=self.faiss_gpu_id)
        
        # Collect results
        top_k_entries = []
        for i in range(top_k):
            top_k_entries.append(self.knowledge_base[self.faiss_index_ids[I[0][i]]].url)
            # if return_entry_list:
            #     top_k_entries.append(self.knowledge_base[self.faiss_index_ids[I[0][i]]])
            # else:
            #     index_id = self.faiss_index_ids[I[0][i]]
            #     start_id = self.faiss_index_ids.index(index_id)
            #     offset = I[0][i] - start_id
            #     top_k_entries.append(
            #         {
            #             "url": self.knowledge_base[self.faiss_index_ids[I[0][i]]].url,
            #             "knowledge_base_index": self.faiss_index_ids[I[0][i]],
            #             "image_url": self.knowledge_base[
            #                 self.faiss_index_ids[I[0][i]]
            #             ].image_urls[offset],
            #             "similarity": D[0][i],
            #             "kb_entry": self.knowledge_base[self.faiss_index_ids[I[0][i]]],
            #         }
            #     )
        return top_k_entries

    @torch.no_grad()
    def search_record(
        self, image, text, alpha=0.5, top_k=100, return_entry_list=False
    ):
        """
        Retrieve using hybrid embeddings (visual + text).
        
        Args:
            image: Input image
            text: Input text (caption + query)
            alpha: Weight for visual embedding (0-1). Text weight is (1-alpha)
            top_k: Number of top results to return
            return_entry_list: Whether to return entry list
            
        Returns:
            List of top k retrieved entries
        """
        # Extract visual embedding
        if self.model_type == "clip":
            inputs = (
                self.processor(images=image, return_tensors="pt")
                .pixel_values.to(self.device)
                .half()
            )
            image_score = self.model.get_image_features(inputs)
        elif self.model_type == "eva-clip":
            inputs = (
                self.processor(images=image, return_tensors="pt")
                .pixel_values.to(self.device)
                .half()
            )
            image_score = self.model.encode_image(inputs)
        
        # Normalize visual embedding
        visual_embed = F.normalize(image_score.float(), p=2, dim=1)
        
        # Extract and normalize text embedding
        text_embed = self.extract_text_embedding(text)
        
        # Combine embeddings: concat(alpha * visual, (1-alpha) * text)
        visual_weighted = alpha * visual_embed.to(text_embed.device)
        text_weighted = (1 - alpha) * text_embed
        hybrid_embed = torch.cat([visual_weighted, text_weighted], dim=1)
        
        # Normalize the final hybrid embedding
        query = F.normalize(hybrid_embed, p=2, dim=1)
        
        assert self.faiss_index and self.faiss_index_ids is not None
        
        # Multi-GPU mode: use CPU numpy array for search
        if self.faiss_gpu_ids is not None:
            query_cpu = query.detach().cpu().numpy().astype(np.float32)
            D, I = self.faiss_index.search(query_cpu, top_k)
            torch.cuda.synchronize(device=self.device)
            for gpu_id in self.faiss_gpu_ids:
                torch.cuda.synchronize(device=gpu_id)
        # Single GPU mode: use GPU tensor
        else:
            with torch.cuda.device(self.faiss_gpu_id):
                query_gpu = query.cuda(self.faiss_gpu_id)
                D, I = self.faiss_index.search(query_gpu, top_k)
            torch.cuda.synchronize(device=self.device)
            torch.cuda.synchronize(device=self.faiss_gpu_id)
        
        # Collect results
        top_k_entries = []
        for i in range(top_k):
            index_id = self.faiss_index_ids[I[0][i]]
            start_id = self.faiss_index_ids.index(index_id)
            offset = I[0][i] - start_id
            top_k_entries.append(
                {
                    "url": self.knowledge_base[self.faiss_index_ids[I[0][i]]].url,
                    "knowledge_base_index": self.faiss_index_ids[I[0][i]],
                    "image_url": self.knowledge_base[
                        self.faiss_index_ids[I[0][i]]
                    ].image_urls[offset],
                    "similarity": D[0][i],
                    "kb_entry": self.knowledge_base[self.faiss_index_ids[I[0][i]]],
                }
            )
        return top_k_entries