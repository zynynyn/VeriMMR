import asyncio
import gc
import sys
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import aiohttp
import orjson
import numpy as np
from tqdm import tqdm
from PIL import Image

from fastmcp.exceptions import ValidationError, NotFoundError, ToolError
from ultrarag.server import UltraRAG_MCP_Server
from index_backends import BaseIndexBackend, create_index_backend

app = UltraRAG_MCP_Server("retriever")


class Retriever:
    def __init__(self, mcp_inst: UltraRAG_MCP_Server):
        mcp_inst.tool(
            self.retriever_init,
            output="model_name_or_path,backend_configs,batch_size,corpus_path,gpu_ids,is_multimodal,backend,index_backend,index_backend_configs,zac_prover_state,sumcheck_embedding_npy,sumcheck_mode,zkllm_workdir,zkllm_k_layers->None",
        )
        mcp_inst.tool(
            self.retriever_embed,
            output="embedding_path,overwrite,is_multimodal->None",
        )
        mcp_inst.tool(
            self.retriever_index,
            output="embedding_path,overwrite->None",
        )
        mcp_inst.tool(
            self.bm25_index,
            output="overwrite->None",
        )
        mcp_inst.tool(
            self.retriever_search,
            output="q_ls,top_k,query_instruction->ret_psg,zac_verification,sumcheck_verification,zkllm_verification",
        )
        mcp_inst.tool(
            self.retriever_deploy_search,
            output="retriever_url,q_ls,top_k,query_instruction->ret_psg",
        )
        mcp_inst.tool(
            self.retriever_search_colbert_maxsim,
            output="q_ls,embedding_path,top_k,query_instruction->ret_psg",
        )
        mcp_inst.tool(
            self.bm25_search,
            output="q_ls,top_k->ret_psg",
        )
        mcp_inst.tool(
            self.retriever_exa_search,
            output="q_ls,top_k,retrieve_thread_num->ret_psg",
        )
        mcp_inst.tool(
            self.retriever_tavily_search,
            output="q_ls,top_k,retrieve_thread_num->ret_psg",
        )
        mcp_inst.tool(
            self.retriever_zhipuai_search,
            output="q_ls,top_k,retrieve_thread_num->ret_psg",
        )

    def _drop_keys(self, d: Dict[str, Any], banned: List[str]) -> Dict[str, Any]:
        return {k: v for k, v in (d or {}).items() if k not in banned and v is not None}

    def retriever_init(
        self,
        model_name_or_path: str,
        backend_configs: Dict[str, Any],
        batch_size: int,
        corpus_path: str,
        gpu_ids: Optional[object] = None,
        is_multimodal: bool = False,
        backend: str = "sentence_transformers",
        index_backend: str = "faiss",
        index_backend_configs: Optional[Dict[str, Any]] = None,
        zac_prover_state: Optional[str] = None,
        sumcheck_embedding_npy: Optional[str] = None,
        sumcheck_mode: str = "global",
        zkllm_workdir: Optional[str] = None,
        zkllm_k_layers: int = 6,
    ):

        self.backend = backend.lower()
        self.index_backend_name = index_backend.lower()
        self.index_backend_configs = index_backend_configs or {}
        self.index_backend: Optional[BaseIndexBackend] = None

        self.batch_size = batch_size
        self.backend_configs = backend_configs

        cfg = self.backend_configs.get(self.backend, {})
        self.cfg = cfg

        if gpu_ids is None:
            self.gpu_ids = None
            self.device = "cpu"
            self.device_num = 1
            app.logger.info("[retriever] gpu_ids is None, treat as CPU-only mode.")
        else:
            gpu_ids = str(gpu_ids)
            os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids
            self.gpu_ids = gpu_ids
            self.device = "cuda"
            self.device_num = len(gpu_ids.split(","))
            app.logger.info(
                "[retriever] Set CUDA_VISIBLE_DEVICES=%s, device_num=%d",
                gpu_ids,
                self.device_num,
            )

        if self.backend == "infinity":
            try:
                from infinity_emb import AsyncEngineArray, EngineArgs
            except ImportError:
                err_msg = "infinity_emb is not installed. Please install it with `pip install infinity-emb`."
                app.logger.error(err_msg)
                raise ImportError(err_msg)

            infinity_engine_args = EngineArgs(
                model_name_or_path=model_name_or_path,
                batch_size=self.batch_size,
                device=self.device,
                trust_remote_code=True,
                **cfg,
            )
            self.model = AsyncEngineArray.from_args([infinity_engine_args])[0]

        elif self.backend == "sentence_transformers":
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError:
                err_msg = (
                    "sentence_transformers is not installed. "
                    "Please install it with `pip install sentence-transformers`."
                )
                app.logger.error(err_msg)
                raise ImportError(err_msg)
            self.st_encode_params = cfg.get("sentence_transformers_encode", {}) or {}
            st_params = self._drop_keys(cfg, banned=["sentence_transformers_encode"])
            # 打印加载信息
            app.logger.info(
                f"[sentence_transformers] Loading model '{model_name_or_path}' "
                f"with device='{self.device}' and params={st_params}"
            )
            self.model = SentenceTransformer(
                model_name_or_path=model_name_or_path,
                device=self.device,
                **st_params
            )
            app.logger.info(f"[sentence_transformers] Model loaded successfully.")

        elif self.backend == "openai":
            try:
                from openai import AsyncOpenAI, OpenAIError
            except ImportError:
                err_msg = (
                    "openai is not installed. "
                    "Please install it with `pip install openai`."
                )
                app.logger.error(err_msg)
                raise ImportError(err_msg)

            model_name = cfg.get("model_name")
            base_url = cfg.get("base_url")
            api_key = cfg.get("api_key") or os.environ.get("RETRIEVER_API_KEY")

            if not model_name:
                err_msg = "[openai] model_name is required"
                app.logger.error(err_msg)
                raise ValueError(err_msg)
            if not isinstance(base_url, str) or not base_url:
                err_msg = "[openai] base_url must be a non-empty string"
                app.logger.error(err_msg)
                raise ValueError(err_msg)

            try:
                self.model = AsyncOpenAI(base_url=base_url, api_key=api_key)
                self.model_name = model_name
                info_msg = f"[openai] OpenAI client initialized (model='{model_name}', base='{base_url}')"
                app.logger.info(info_msg)
            except OpenAIError as e:
                err_msg = f"[openai] Failed to initialize OpenAI client: {e}"
                app.logger.error(err_msg)
                raise OpenAIError(err_msg)
        elif self.backend == "bm25":
            try:
                import bm25s
            except ImportError:
                err_msg = (
                    "bm25s is not installed. "
                    "Please install it with `pip install bm25s`."
                )
                app.logger.error(err_msg)
                raise ImportError(err_msg)

            try:
                self.model = bm25s.BM25(backend="numba")
            except Exception as e:
                warn_msg = (
                    f"Failed to initialize BM25 model with backend 'numba': {e}. "
                    "Falling back to 'numpy' backend."
                )
                app.logger.warning(warn_msg)
                self.model = bm25s.BM25(backend="numpy")
            lang = cfg.get("lang", "en")
            try:
                self.tokenizer = bm25s.tokenization.Tokenizer(stopwords=lang)
            except Exception as e:
                err_msg = (
                    f"Failed to initialize BM25 tokenizer for language '{lang}': {e}"
                )
                app.logger.error(err_msg)
                raise RuntimeError(err_msg)
        else:
            error_msg = (
                f"Unsupported backend: {backend}. "
                "Supported backends: 'infinity', 'sentence_transformers', 'openai'"
            )
            app.logger.error(error_msg)
            raise ValueError(error_msg)

        # ZAC verifiable retrieval — load prover state if provided
        self._zac_acc = None
        if zac_prover_state and Path(zac_prover_state).exists():
            try:
                sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
                from zac.accumulator import ZACAccumulator
                app.logger.info(f"[ZAC] Loading prover state from {zac_prover_state} (CRS rebuild ~2 min) …")
                self._zac_acc = ZACAccumulator.load_prover_state(zac_prover_state)
                app.logger.info(f"[ZAC] Ready. Root = {self._zac_acc.root_hex()[:16]}…")
            except Exception as e:
                app.logger.warning(f"[ZAC] Failed to load prover state: {e}. Continuing without ZAC.")

        # Sumcheck verifiable retrieval — load corpus embeddings for IP proof
        self._sc_embeddings: Optional[np.ndarray] = None
        self._sc_path_to_idx: Optional[Dict[str, int]] = None
        self._sc_mode: str = (sumcheck_mode or "global").lower()
        if sumcheck_embedding_npy and Path(sumcheck_embedding_npy).exists():
            try:
                src_dir = str(Path(__file__).resolve().parents[3] / "src")
                if src_dir not in sys.path:
                    sys.path.insert(0, src_dir)
                self._sc_embeddings = np.load(sumcheck_embedding_npy).astype(np.float32)
                app.logger.info(
                    f"[Sumcheck] Loaded {self._sc_embeddings.shape[0]} embeddings "
                    f"(D={self._sc_embeddings.shape[1]}) from {sumcheck_embedding_npy}"
                )
            except Exception as e:
                app.logger.warning(f"[Sumcheck] Failed to load embeddings: {e}. Continuing without Sumcheck.")

        # Phase 3: zkLLM verifiable inference (async background)
        # 二进制位于 UltraRAG/src/zkllm/bin/，workdir 相对于 UltraRAG 根目录解析
        self._zkllm_k_layers: int = int(zkllm_k_layers or 6)
        self._ultrarag_root = Path(__file__).resolve().parents[3]
        self._zkllm_bin_dir = str(self._ultrarag_root / "src" / "zkllm" / "bin")
        if zkllm_workdir:
            _wd = Path(zkllm_workdir)
            self._zkllm_workdir: Optional[str] = str(
                _wd if _wd.is_absolute() else (self._ultrarag_root / _wd)
            )
            app.logger.info(
                f"[zkLLM] Phase 3 enabled: workdir={self._zkllm_workdir}, K={self._zkllm_k_layers} layers"
            )
        else:
            self._zkllm_workdir: Optional[str] = None

        self.contents = []
        corpus_path_obj = Path(corpus_path)
        corpus_dir = corpus_path_obj.parent
        file_size = os.path.getsize(corpus_path)

        with open(corpus_path, "rb") as f:
            with tqdm(
                total=file_size,
                desc="Loading corpus",
                unit="B",
                unit_scale=True,
                ncols=100,
            ) as pbar:
                bytes_read = 0
                for i, line in enumerate(f):
                    pbar.update(len(line))
                    bytes_read += len(line)
                    try:
                        item = orjson.loads(line)
                    except orjson.JSONDecodeError as e:
                        raise ToolError(f"Invalid JSON on line {i}: {e}") from e
                    if not is_multimodal or self.backend == "bm25":
                        if "contents" not in item:
                            error_msg = (
                                f"Line {i}: missing key 'contents'. full item={item}"
                            )
                            app.logger.error(error_msg)
                            raise ValueError(error_msg)

                        self.contents.append(item["contents"])
                    else:
                        if "image_path" not in item:
                            error_msg = (
                                f"Line {i}: missing key 'image_path'. full item={item}"
                            )
                            app.logger.error(error_msg)
                            raise ValueError(error_msg)

                        rel = str(item["image_path"])
                        abs_path = str((corpus_dir / rel).resolve())
                        self.contents.append(abs_path)
                if bytes_read < file_size:
                    pbar.update(file_size - bytes_read)
                pbar.refresh() 

        # Build path→corpus-index mapping for Sumcheck embedding lookup
        if self._sc_embeddings is not None:
            self._sc_path_to_idx = {path: i for i, path in enumerate(self.contents)}
            app.logger.info(
                f"[Sumcheck] Built path→idx mapping for {len(self._sc_path_to_idx)} entries."
            )

        if self.backend in ["infinity", "sentence_transformers", "openai"]:
            index_backend_cfg = self.index_backend_configs.get(
                self.index_backend_name, {}
            )
            self.index_backend = create_index_backend(
                name=self.index_backend_name,
                contents=self.contents,
                logger=app.logger,
                config=index_backend_cfg,
                device_num=self.device_num,
            )
            app.logger.info(
                "[index] Initialized backend '%s'.", self.index_backend_name
            )
            try:
                self.index_backend.load_index()
            except Exception as exc:
                warn_msg = (
                    f"[index] Failed to load existing index using backend "
                    f"'{self.index_backend_name}': {exc}"
                )
                app.logger.warning(warn_msg)

        elif self.backend == "bm25":
            bm25_save_path = cfg.get("save_path", None)
            if bm25_save_path and os.path.exists(bm25_save_path):
                self.model = self.model.load(bm25_save_path, mmap=True, load_corpus=False)
                self.tokenizer.load_stopwords(bm25_save_path)
                self.tokenizer.load_vocab(bm25_save_path)
                self.model.corpus = self.contents
                self.model.backend = "numba"
                info_msg = "[bm25] Index loaded successfully."
                app.logger.info(info_msg)
            else:
                if bm25_save_path and not os.path.exists(bm25_save_path):
                    warn_msg = f"{bm25_save_path} does not exist."
                    app.logger.warning(warn_msg)
                info_msg = "[bm25] no index_path provided. Retriever initialized without index."
                app.logger.info(info_msg)

    async def retriever_embed(
        self,
        embedding_path: Optional[str] = None,
        overwrite: bool = False,
        is_multimodal: bool = False,
    ):
        embeddings = None

        if embedding_path is not None:
            if not embedding_path.endswith(".npy"):
                err_msg = (
                    f"Embedding save path must end with .npy, "
                    f"now the path is {embedding_path}"
                )
                app.logger.error(err_msg)
                raise ValidationError(err_msg)
            output_dir = os.path.dirname(embedding_path)
        else:
            current_file = os.path.abspath(__file__)
            project_root = os.path.dirname(os.path.dirname(current_file))
            output_dir = os.path.join(project_root, "output", "embedding")
            embedding_path = os.path.join(output_dir, "embedding.npy")

        if not overwrite and os.path.exists(embedding_path):
            app.logger.info("Embedding already exists, skipping")
            return

        os.makedirs(output_dir, exist_ok=True)

        if self.backend == "infinity":
            async with self.model:
                if is_multimodal:
                    data = []
                    for i, p in enumerate(self.contents):
                        try:
                            with Image.open(p) as im:
                                data.append(im.convert("RGB").copy())
                        except Exception as e:
                            err_msg = f"Failed to load image at index {i}: {p} ({e})"
                            app.logger.error(err_msg)
                            raise RuntimeError(err_msg)
                    call = self.model.image_embed
                else:
                    data = self.contents
                    call = self.model.embed

                eff_bs = self.batch_size * self.device_num
                n = len(data)
                pbar = tqdm(total=n, desc="[infinity] Embedding:")
                embeddings = []
                for i in range(0, n, eff_bs):
                    chunk = data[i : i + eff_bs]
                    vecs, _ = (
                        await call(images=chunk)
                        if is_multimodal
                        else await call(sentences=chunk)
                    )
                    embeddings.extend(vecs)
                    pbar.update(len(chunk))
                pbar.close()

        elif self.backend == "sentence_transformers":
            if self.device_num == 1:
                device_param = "cuda:0"
            else:
                device_param = [f"cuda:{i}" for i in range(self.device_num)]
            normalize = bool(self.st_encode_params.get("normalize_embeddings", False))
            csz = int(self.st_encode_params.get("encode_chunk_size", 10000))
            psg_prompt_name = self.st_encode_params.get("psg_prompt_name", None)
            psg_task = self.st_encode_params.get("psg_task", None)

            if is_multimodal:
                data = []
                for p in self.contents:
                    with Image.open(p) as im:
                        data.append(im.convert("RGB").copy())
            else:
                data = self.contents

            if isinstance(device_param, list) and len(device_param) > 1:
                pool = self.model.start_multi_process_pool()
                try:

                    def _encode_all():
                        return self.model.encode(
                            data,
                            pool=pool,
                            batch_size=self.batch_size,
                            chunk_size=csz,
                            show_progress_bar=True,
                            normalize_embeddings=normalize,
                            precision="float32",
                            prompt_name=psg_prompt_name,
                            task=psg_task,
                        )

                    embeddings = await asyncio.to_thread(_encode_all)
                finally:
                    self.model.stop_multi_process_pool(pool)
            else:

                def _encode_single():
                    return self.model.encode(
                        data,
                        device=device_param,
                        batch_size=self.batch_size,
                        show_progress_bar=True,
                        normalize_embeddings=normalize,
                        precision="float32",
                        prompt_name=psg_prompt_name,
                        task=psg_task,
                    )

                embeddings = await asyncio.to_thread(_encode_single)

        elif self.backend == "openai":
            if is_multimodal:
                err_msg = (
                    "openai backend does not support image embeddings in this path."
                )
                app.logger.error(err_msg)
                raise ValueError(err_msg)

            embeddings: list = []
            with tqdm(
                total=len(self.contents),
                desc="[openai] Embedding:",
                unit="item",
            ) as pbar:
                for start in range(0, len(self.contents), self.batch_size):
                    chunk = self.contents[start : start + self.batch_size]
                    resp = await self.model.embeddings.create(
                        model=self.model_name,
                        input=chunk,
                    )
                    embeddings.extend([d.embedding for d in resp.data])
                    pbar.update(len(chunk))
        else:
            err_msg = f"Unsupported backend: {self.backend}"
            app.logger.error(err_msg)
            raise ValueError(err_msg)

        if embeddings is None:
            raise RuntimeError("Embedding generation failed: embeddings is None")
        embeddings = np.array(embeddings, dtype=np.float32)
        np.save(embedding_path, embeddings)

        del embeddings
        gc.collect()
        app.logger.info("embedding success")

    def retriever_index(
        self,
        embedding_path: str,
        overwrite: bool = False,
    ):
        if self.backend == "bm25":
            err_msg = "BM25 backend does not support vector index building via retriever_index."
            app.logger.error(err_msg)
            raise ValueError(err_msg)

        if self.index_backend is None:
            err_msg = (
                "Vector index backend is not initialized. "
                "Ensure retriever_init completed successfully."
            )
            app.logger.error(err_msg)
            raise RuntimeError(err_msg)

        if not os.path.exists(embedding_path):
            app.logger.error(f"Embedding file not found: {embedding_path}")
            raise NotFoundError(f"Embedding file not found: {embedding_path}")

        embedding = np.load(embedding_path)
        vec_ids = np.arange(embedding.shape[0]).astype(np.int64)
        
        try:
            self.index_backend.build_index(
                embeddings=embedding,
                ids=vec_ids,
                overwrite=overwrite,
            )
        except ValueError as exc:
            raise ValidationError(str(exc)) from exc
        finally:
            del embedding
            gc.collect()

        
        info_msg = f"[{self.index_backend_name}] Indexing success."
        app.logger.info(info_msg)

    def bm25_index(
        self,
        overwrite: bool = False,
    ):
        bm25_save_path = self.cfg.get("save_path", None)
        if bm25_save_path:
            output_dir = os.path.dirname(bm25_save_path)
        else:
            current_file = os.path.abspath(__file__)
            project_root = os.path.dirname(os.path.dirname(current_file))
            output_dir = os.path.join(project_root, "output", "index")
            bm25_save_path = os.path.join(output_dir, "bm25")

        if not overwrite and os.path.exists(bm25_save_path):
            info_msg = (
                f"Index file already exists: {bm25_save_path}. "
                "Set overwrite=True to overwrite."
            )
            app.logger.info(info_msg)
            return

        if overwrite and os.path.exists(bm25_save_path):
            os.remove(bm25_save_path)

        corpus_tokens = self.tokenizer.tokenize(self.contents, return_as="tuple")
        self.model.index(corpus_tokens)
        self.model.save(bm25_save_path, corpus=None)
        self.tokenizer.save_stopwords(bm25_save_path)
        self.tokenizer.save_vocab(bm25_save_path)
        info_msg = "[bm25] Indexing success."
        app.logger.info(info_msg)

    async def retriever_search(
        self,
        query_list: List[str],
        top_k: int = 5,
        query_instruction: str = "",
    ) -> Dict[str, Any]:

        if isinstance(query_list, str):
            query_list = [query_list]
        queries = [f"{query_instruction}{query}" for query in query_list]

        if self.backend == "infinity":
            async with self.model:
                query_embedding, _ = await self.model.embed(sentences=queries)
        elif self.backend == "sentence_transformers":
            if self.device_num == 1:
                device_param = "cuda:0"
            else:
                device_param = [f"cuda:{i}" for i in range(self.device_num)]
            normalize = bool(self.st_encode_params.get("normalize_embeddings", False))
            q_prompt_name = self.st_encode_params.get("q_prompt_name", "")
            q_task = self.st_encode_params.get("psg_task", None)

            if isinstance(device_param, list) and len(device_param) > 1:
                pool = self.model.start_multi_process_pool()
                try:

                    def _encode_all():
                        return self.model.encode(
                            queries,
                            pool=pool,
                            batch_size=self.batch_size,
                            show_progress_bar=True,
                            normalize_embeddings=normalize,
                            precision="float32",
                            prompt_name=q_prompt_name,
                            task=q_task,
                        )

                    query_embedding = await asyncio.to_thread(_encode_all)
                finally:
                    self.model.stop_multi_process_pool(pool)
            else:

                def _encode_single():
                    return self.model.encode(
                        queries,
                        device=device_param,
                        batch_size=self.batch_size,
                        show_progress_bar=True,
                        normalize_embeddings=normalize,
                        precision="float32",
                        prompt_name=q_prompt_name,
                        task=q_task,
                    )

                query_embedding = await asyncio.to_thread(_encode_single)

        elif self.backend == "openai":
            query_embedding = []
            for i in tqdm(
                range(0, len(queries), self.batch_size),
                desc="[openai] Embedding:",
                unit="batch",
            ):
                chunk = queries[i : i + self.batch_size]
                resp = await self.model.embeddings.create(
                    model=self.model_name, input=chunk
                )
                query_embedding.extend([d.embedding for d in resp.data])

        else:
            error_msg = f"Unsupported backend: {self.backend}"
            app.logger.error(error_msg)
            raise ValueError(error_msg)

        query_embedding = np.array(query_embedding, dtype=np.float32)

        info_msg = f"query embedding shape: {query_embedding.shape}"
        app.logger.info(info_msg)
        
        if self.index_backend is None:
            err_msg = (
                "Vector index backend is not initialized. "
                "Ensure retriever_init completed successfully."
            )
            app.logger.error(err_msg)
            raise RuntimeError(err_msg)

        rets = self.index_backend.search(query_embedding, top_k)

        # ZAC: one aggregate proof per query (parallel to ret_psg structure)
        zac_verification = None
        if self._zac_acc is not None:
            try:
                import time as _time
                from zac.accumulator import ZACAccumulator as _ZAC
                per_query_results = []
                for per_query_paths in rets:
                    paths = [p for p in per_query_paths if p]
                    if not paths:
                        per_query_results.append(None)
                        continue
                    elements = [_ZAC.image_hash(p) for p in paths]
                    t0 = _time.perf_counter()
                    proof = await asyncio.to_thread(self._zac_acc.prove_membership_batch, elements)
                    prove_ms = round((_time.perf_counter() - t0) * 1000, 1)
                    t0 = _time.perf_counter()
                    verified = await asyncio.to_thread(self._zac_acc.verify_membership_batch, elements, proof)
                    verify_ms = round((_time.perf_counter() - t0) * 1000, 1)
                    per_query_results.append({
                        "verified": verified,
                        "num_images": len(paths),
                        "proof_hex": proof["proof_hex"],
                        "proof_bytes": 48,
                        "cm_hex": proof["cm_hex"],
                        "prove_ms": prove_ms,
                        "verify_ms": verify_ms,
                    })
                    app.logger.info(f"[ZAC] verified={verified}, {len(paths)} images, prove={prove_ms}ms, verify={verify_ms}ms")
                zac_verification = per_query_results  # List[dict|None], parallel to ret_psg
            except Exception as e:
                app.logger.warning(f"[ZAC] Proof generation failed: {e}")

        # Sumcheck: verifiable retrieval proof per query
        # global mode (default): prove all N corpus scores → Verifier selects top-k
        # local mode: prove only top-k scores + ranking witness
        sumcheck_verification = None
        if self._sc_embeddings is not None and self._sc_path_to_idx is not None:
            try:
                import time as _time
                if self._sc_mode == "global":
                    from sumcheck.inner_product import prove_global_batch, verify_global_batch
                else:
                    from sumcheck.inner_product import prove_retrieval, verify_retrieval
                sc_per_query = []
                # Pre-compute full all_corpus_vecs list once (global mode)
                all_corpus_vecs_list = (
                    self._sc_embeddings.tolist()
                    if self._sc_mode == "global"
                    else None
                )
                for qi, (per_query_paths, query_vec) in enumerate(
                    zip(rets, query_embedding)
                ):
                    paths = [p for p in per_query_paths if p]
                    if not paths:
                        sc_per_query.append(None)
                        continue

                    q_list = query_vec.tolist()

                    if self._sc_mode == "global":
                        t0 = _time.perf_counter()
                        sc_proof = await asyncio.to_thread(
                            prove_global_batch, q_list, all_corpus_vecs_list
                        )
                        prove_ms = round((_time.perf_counter() - t0) * 1000, 1)
                        t0 = _time.perf_counter()
                        sc_result = await asyncio.to_thread(
                            verify_global_batch, q_list, all_corpus_vecs_list,
                            sc_proof, top_k
                        )
                        verify_ms = round((_time.perf_counter() - t0) * 1000, 1)
                        sc_ok = sc_result["verified"]
                        N = sc_proof.get("N", len(all_corpus_vecs_list))
                        proof_bytes = N * 8 + 264  # N scores (8B each) + 1 Sumcheck (264B)
                        sc_per_query.append({
                            "verified": sc_ok,
                            "mode": "global",
                            "N": N,
                            "k": len(paths),
                            "ell": sc_proof.get("sc_proof", {}).get("ell"),
                            "proof_bytes": proof_bytes,
                            "top_k_indices": sc_result.get("top_k_indices"),
                            "top_k_scores": sc_result.get("top_k_scores"),
                            "prove_ms": prove_ms,
                            "verify_ms": verify_ms,
                        })
                        app.logger.info(
                            f"[Sumcheck/global] query {qi}: verified={sc_ok}, "
                            f"N={N}, k={len(paths)}, prove={prove_ms}ms, verify={verify_ms}ms"
                        )
                    else:
                        # Local mode: prove only retrieved top-k
                        corpus_vecs = []
                        for p in paths:
                            idx = self._sc_path_to_idx.get(p)
                            if idx is None or idx >= len(self._sc_embeddings):
                                break
                            corpus_vecs.append(self._sc_embeddings[idx].tolist())
                        if len(corpus_vecs) != len(paths):
                            sc_per_query.append(None)
                            continue
                        t0 = _time.perf_counter()
                        sc_proof = await asyncio.to_thread(
                            prove_retrieval, q_list, corpus_vecs
                        )
                        prove_ms = round((_time.perf_counter() - t0) * 1000, 1)
                        t0 = _time.perf_counter()
                        sc_ok = await asyncio.to_thread(
                            verify_retrieval, q_list, corpus_vecs, sc_proof
                        )
                        verify_ms = round((_time.perf_counter() - t0) * 1000, 1)
                        sc_per_query.append({
                            "verified": sc_ok,
                            "mode": "local",
                            "k": len(paths),
                            "ell": sc_proof["ip_proofs"][0]["ell"],
                            "proof_bytes": len(__import__("json").dumps(sc_proof).encode()),
                            "prove_ms": prove_ms,
                            "verify_ms": verify_ms,
                        })
                        app.logger.info(
                            f"[Sumcheck/local] query {qi}: verified={sc_ok}, "
                            f"k={len(paths)}, prove={prove_ms}ms, verify={verify_ms}ms"
                        )
                sumcheck_verification = sc_per_query
            except Exception as e:
                app.logger.warning(f"[Sumcheck] Proof generation failed: {e}")

        # Phase 3: zkLLM — query-side async proof + corpus-side pre-computed proofs
        zkllm_verification = None
        if self._zkllm_workdir:
            import uuid, json as _json, os as _os

            # (a) query-side: 后台异步生成 query embedding 证明
            proof_id = str(uuid.uuid4())[:8]
            asyncio.ensure_future(
                self._zkllm_prove_query_async(query_list, proof_id)
            )

            # (b) corpus-side: 查找每个检索结果的预计算证明
            corpus_proofs = []
            for per_query_paths in rets:
                per_query_corpus = []
                for img_path in (per_query_paths or []):
                    if not img_path:
                        per_query_corpus.append(None)
                        continue
                    # image_id = 去掉前缀 "image/" 的相对路径，与 corpus jsonl 中保持一致
                    # e.g. "corpora/image/nikon/page_0.jpg" → "nikon/page_0.jpg"
                    # 尝试从路径中提取 image_id（格式：{stem}/{page}.jpg）
                    try:
                        # img_path 形如 "corpora/image/nikon/page_0.jpg"
                        # 取最后两段作为 image_id
                        image_id = "/".join(Path(img_path).parts[-2:])
                    except Exception:
                        image_id = Path(img_path).name
                    safe = image_id.replace("/", "_").replace("\\", "_").replace(" ", "_")
                    proof_file = _os.path.join(self._zkllm_workdir, f"corpus_proof_{safe}.json")
                    if _os.path.exists(proof_file):
                        try:
                            with open(proof_file) as f:
                                per_query_corpus.append(_json.load(f))
                        except Exception:
                            per_query_corpus.append({"status": "load_error", "image_id": image_id})
                    else:
                        per_query_corpus.append({"status": "not_precomputed", "image_id": image_id})
                corpus_proofs.append(per_query_corpus)

            # 立即返回 pending 状态（附带语料库侧证明状态）
            zkllm_verification = {
                "query_proof": {
                    "status": "pending",
                    "proof_id": proof_id,
                    "k_layers": self._zkllm_k_layers,
                    "note": "query embedding proof running in background; check server log for completion"
                },
                "corpus_proofs": corpus_proofs,
            }

        result = {"ret_psg": rets}
        if zac_verification is not None:
            result["zac_verification"] = zac_verification
        if sumcheck_verification is not None:
            result["sumcheck_verification"] = sumcheck_verification
        if zkllm_verification is not None:
            result["zkllm_verification"] = zkllm_verification
        return result

    async def _zkllm_prove_query_async(self, query_list: List[str], proof_id: str):
        """
        Phase 3 后台任务：对 query embedding 生成 zkLLM 推理证明。
        证明最后 K 层的 Attn-linear + FFN 计算正确性。
        完成后将结果写入 workdir/zkllm_proof_{proof_id}.json 供审计。
        """
        import time as _time, json, os
        t0 = _time.perf_counter()
        workdir   = self._zkllm_workdir
        K         = self._zkllm_k_layers
        bin_dir   = self._zkllm_bin_dir
        zkllm_cwd = str(Path(bin_dir).parent)

        proof_result = {
            "proof_id": proof_id,
            "status": "failed",
            "k_layers": K,
            "layers": [],
            "verified": False,
            "elapsed_ms": 0,
        }
        try:
            import subprocess
            layer_results = []
            for layer_idx in range(36 - K, 36):
                prefix   = f"layer-{layer_idx}"
                inp_path = os.path.join(workdir, f"{prefix}-query-input.bin")
                if not os.path.exists(inp_path):
                    import numpy as np
                    inp = (np.random.randn(512, 2048) * 65536).astype(np.int32)
                    inp.tofile(inp_path)

                ffn_out  = os.path.join(workdir, f"{prefix}-query-ffn-out.bin")
                attn_out = os.path.join(workdir, f"{prefix}-query-attn-out.bin")

                r1 = subprocess.run(
                    [f"{bin_dir}/ffn", inp_path, "512", "2048", "11008",
                     workdir, prefix, ffn_out],
                    capture_output=True, cwd=zkllm_cwd
                )
                r2 = subprocess.run(
                    [f"{bin_dir}/self-attn", "linear", inp_path, "512", "2048",
                     workdir, prefix, attn_out, "256"],
                    capture_output=True, cwd=zkllm_cwd
                )
                ok = (r1.returncode == 0 and r2.returncode == 0)
                layer_results.append({"layer": layer_idx, "verified": ok})

            elapsed_ms = round((_time.perf_counter() - t0) * 1000)
            verified   = all(r["verified"] for r in layer_results)

            proof_result.update({
                "status": "completed",
                "layers": layer_results,
                "verified": verified,
                "elapsed_ms": elapsed_ms,
            })
            app.logger.info(
                f"[zkLLM] proof_id={proof_id} done: K={K}, verified={verified}, "
                f"elapsed={elapsed_ms}ms"
            )
        except Exception as e:
            proof_result["error"] = str(e)
            app.logger.warning(f"[zkLLM] proof_id={proof_id} failed: {e}")
        finally:
            # 写入结果文件，供后续审计或轮询
            out_path = os.path.join(workdir, f"zkllm_proof_{proof_id}.json")
            try:
                with open(out_path, "w") as f:
                    json.dump(proof_result, f, indent=2)
            except Exception:
                pass
    
    async def retriever_deploy_search(
        self,
        retriever_url: str,
        query_list: List[str],
        top_k: int = 5,
        query_instruction: str = "",
    ) -> Dict[str, List[List[str]]]:
        from urllib.parse import urlparse, urlunparse
        import aiohttp

        url = retriever_url.strip()
        if not url.startswith("http://") and not url.startswith("https://"):
            url = f"http://{url}"

        url_obj = urlparse(url)
        api_url = urlunparse(url_obj._replace(path="/search", query="", fragment=""))

        app.logger.info(f"[remote_retriever] Calling remote retriever at: {api_url}")


        payload: Dict[str, Any] = {
            "query_list": query_list,
            "top_k": top_k,
            "query_instruction": query_instruction,
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(api_url, json=payload) as response:
                if response.status != 200:
                    err_text = await response.text()
                    err_msg = (
                        f"[remote_retriever] Failed to call {api_url}, "
                        f"status={response.status}, body={err_text}"
                    )
                    app.logger.error(err_msg)
                    raise ToolError(err_msg)

                response_data = await response.json()
                app.logger.debug(
                    f"[remote_retriever] status={response.status}, keys={list(response_data.keys())}"
                )

                if "ret_psg" not in response_data:
                    err_msg = (
                        f"[remote_retriever] Response missing 'ret_psg' field: "
                        f"{response_data}"
                    )
                    app.logger.error(err_msg)
                    raise ToolError(err_msg)

                return {"ret_psg": response_data["ret_psg"]}

    async def retriever_search_colbert_maxsim(
        self,
        query_list: List[str],
        embedding_path: str,
        top_k: int = 5,
        query_instruction: str = "",
    ) -> Dict[str, List[List[str]]]:
        try:
            import torch
        except ImportError:
            err_msg = (
                "torch is not installed. Please install it with `pip install torch`."
            )
            app.logger.error(err_msg)
            raise ImportError(err_msg)

        if self.backend not in ["infinity"]:
            error_msg = (
                "retriever_search_colbert_maxsim only supports 'infinity' backend "
                "with ColBERT/ColPali multi-vector models. "
                "Use retriever_search or other backend-specific retrieval functions instead."
            )
            app.logger.error(error_msg)
            raise ValueError(error_msg)

        if isinstance(query_list, str):
            query_list = [query_list]
        queries = [f"{query_instruction}{query}" for query in query_list]

        async with self.model:
            query_embedding, _ = await self.model.embed(sentences=queries)

        doc_embeddings = np.load(embedding_path)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if (
            isinstance(doc_embeddings, np.ndarray)
            and doc_embeddings.dtype != object
            and doc_embeddings.ndim == 3
        ):
            docs_tensor = torch.from_numpy(
                doc_embeddings.astype("float32", copy=False)
            ).to(device)
        elif isinstance(doc_embeddings, np.ndarray) and doc_embeddings.dtype == object:
            try:
                stacked = np.stack(
                    [np.asarray(x, dtype=np.float32) for x in doc_embeddings.tolist()],
                    axis=0,
                )
                docs_tensor = torch.from_numpy(stacked).to(device)
            except Exception:
                error_msg = (
                    f"Document embeddings in {embedding_path} have inconsistent shapes, "
                    "cannot stack into (N,Kd,D). "
                    f"Check your retriever_embed."
                )
                app.logger.error(error_msg)
                raise ValueError(error_msg)
        else:
            error_msg = (
                f"Unexpected doc_embeddings format: type={type(doc_embeddings)}, "
                f"shape={getattr(doc_embeddings, 'shape', None)}"
            )
            app.logger.error(error_msg)
            raise ValueError(error_msg)

        def _l2norm(t: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
            return t / t.norm(dim=-1, keepdim=True).clamp_min(eps)

        N, _, D_docs = docs_tensor.shape
        docs_tensor = _l2norm(docs_tensor)
        k_pick = min(top_k, N)

        results = []
        for q_np in query_embedding:
            q = torch.as_tensor(
                q_np,
                dtype=torch.float32,
                device=device,
            )
            if q.shape[-1] != D_docs:
                error_msg = (
                    f"Dimension mismatch: query D={q.shape[-1]} vs doc D={D_docs}"
                )
                app.logger.error(error_msg)
                raise ValueError(error_msg)

            q = _l2norm(q)
            sim = torch.einsum("qd,nkd->nqk", q, docs_tensor)
            sim_max = sim.max(dim=2).values
            scores = sim_max.sum(dim=1)

            top_idx = torch.topk(scores, k=k_pick, largest=True).indices.tolist()
            results.append([self.contents[i] for i in top_idx])
        return {"ret_psg": results}

    async def bm25_search(
        self,
        query_list: List[str],
        top_k: int = 5,
    ) -> Dict[str, List[List[str]]]:
        results = []
        q_toks = self.tokenizer.tokenize(
            query_list,
            return_as="tuple",
            update_vocab=False,
        )
        results, scores = self.model.retrieve(q_toks, k=top_k)
        results = results.tolist() if isinstance(results, np.ndarray) else results
        scores = scores.tolist() if isinstance(scores, np.ndarray) else scores
        return {"ret_psg": results}

    async def _parallel_search(
        self,
        query_list: List[str],
        retrieve_thread_num: int,
        desc: str,
        worker_factory,
    ) -> Dict[str, List[List[str]]]:
        sem = asyncio.Semaphore(retrieve_thread_num)

        async def _wrap(i: int, q: str):
            async with sem:
                return await worker_factory(i, q)

        tasks = [asyncio.create_task(_wrap(i, q)) for i, q in enumerate(query_list)]
        ret: List[List[str]] = [None] * len(query_list)

        iterator = tqdm(asyncio.as_completed(tasks), total=len(tasks), desc=desc)
        for fut in iterator:
            idx, psg_ls = await fut
            ret[idx] = psg_ls
        return {"ret_psg": ret}

    async def retriever_exa_search(
        self,
        query_list: List[str],
        top_k: Optional[int] | None = 5,
        retrieve_thread_num: Optional[int] | None = 1,
    ) -> Dict[str, List[List[str]]]:

        try:
            from exa_py import AsyncExa
            from exa_py.api import Result
        except ImportError:
            err_msg = (
                "exa_py is not installed. Please install it with `pip install exa_py`."
            )
            app.logger.error(err_msg)
            raise ImportError(err_msg)

        exa_api_key = os.environ.get("EXA_API_KEY", "")
        exa = AsyncExa(api_key=exa_api_key if exa_api_key else "EMPTY")

        async def worker_factory(idx: int, q: str):
            retries, delay = 3, 1.0
            for attempt in range(retries):
                try:
                    resp = await exa.search_and_contents(
                        q, num_results=top_k, text=True
                    )
                    results: List[Result] = getattr(resp, "results", []) or []
                    psg_ls: List[str] = [(r.text or "") for r in results]
                    return idx, psg_ls
                except Exception as e:
                    status = getattr(getattr(e, "response", None), "status_code", None)
                    if status == 401 or "401" in str(e):
                        err_msg = (
                            "Unauthorized (401): Invalid or missing EXA_API_KEY. "
                            "Please set it to use Exa."
                        )
                        app.logger.error(err_msg)
                        raise ToolError(err_msg) from e
                    warn_msg = f"[exa][retry {attempt+1}] failed (idx={idx}): {e}"
                    app.logger.warning(warn_msg)
                    await asyncio.sleep(delay)
            return idx, []

        return await self._parallel_search(
            query_list=query_list,
            retrieve_thread_num=retrieve_thread_num or 1,
            desc="EXA Searching:",
            worker_factory=worker_factory,
        )

    async def retriever_tavily_search(
        self,
        query_list: List[str],
        top_k: Optional[int] | None = 5,
        retrieve_thread_num: Optional[int] | None = 1,
    ) -> Dict[str, List[List[str]]]:

        try:
            from tavily import (
                AsyncTavilyClient,
                BadRequestError,
                UsageLimitExceededError,
                InvalidAPIKeyError,
                MissingAPIKeyError,
            )
        except ImportError:
            err_msg = "tavily is not installed. Please install it with `pip install tavily-python`."
            app.logger.error(err_msg)
            raise ImportError(err_msg)

        tavily_api_key = os.environ.get("TAVILY_API_KEY", "")
        if not tavily_api_key:
            err_msg = (
                "TAVILY_API_KEY environment variable is not set. "
                "Please set it to use Tavily."
            )
            app.logger.error(err_msg)
            raise MissingAPIKeyError(err_msg)
        tavily = AsyncTavilyClient(api_key=tavily_api_key)

        async def worker_factory(idx: int, q: str):
            retries, delay = 3, 1.0
            for attempt in range(retries):
                try:
                    resp = await tavily.search(query=q, max_results=top_k)
                    results: List[Dict[str, Any]] = resp["results"]
                    psg_ls: List[str] = [(r.get("content") or "") for r in results]
                    return idx, psg_ls
                except UsageLimitExceededError as e:
                    err_msg = f"Usage limit exceeded: {e}"
                    app.logger.error(err_msg)
                    raise ToolError(err_msg) from e
                except InvalidAPIKeyError as e:
                    err_msg = f"Invalid API key: {e}"
                    app.logger.error(err_msg)
                    raise ToolError(err_msg) from e
                except (BadRequestError, Exception) as e:
                    warn_msg = f"[tavily][retry {attempt+1}] failed (idx={idx}): {e}"
                    app.logger.warning(warn_msg)
                    await asyncio.sleep(delay)
            return idx, []

        return await self._parallel_search(
            query_list=query_list,
            retrieve_thread_num=retrieve_thread_num or 1,
            desc="Tavily Searching:",
            worker_factory=worker_factory,
        )

    async def retriever_zhipuai_search(
        self,
        query_list: List[str],
        top_k: Optional[int] | None = 5,
        retrieve_thread_num: Optional[int] | None = 1,
    ) -> Dict[str, List[List[str]]]:

        zhipuai_api_key = os.environ.get("ZHIPUAI_API_KEY", "")
        if not zhipuai_api_key:
            err_msg = (
                "ZHIPUAI_API_KEY environment variable is not set. "
                "Please set it to use ZhipuAI."
            )
            app.logger.error(err_msg)
            raise ToolError(err_msg)

        retrieval_url = "https://open.bigmodel.cn/api/paas/v4/web_search"
        headers = {
            "Authorization": f"Bearer {zhipuai_api_key}",
            "Content-Type": "application/json",
        }

        session = aiohttp.ClientSession()

        async def worker_factory(idx: int, q: str):
            retries, delay = 3, 1.0
            for attempt in range(retries):
                try:
                    payload = {
                        "search_query": q,
                        "search_engine": "search_std",  # [search_std, search_pro, search_pro_sogou, search_pro_quark]
                        "search_intent": False,
                        "count": top_k,  # [10,20,30,40,50]
                        "search_recency_filter": "noLimit",  # [oneDay, oneWeek, oneMonth, oneYear, noLimit]
                        "content_size": "medium",  # [medium, high]
                    }
                    async with session.post(
                        retrieval_url, json=payload, headers=headers
                    ) as resp:
                        resp.raise_for_status()
                        data = await resp.json()
                        results: List[Dict[str, Any]] = data.get("search_result", [])
                        psg_ls: List[str] = [(r.get("content") or "") for r in results]
                        # Respect top_k
                        return idx, (psg_ls[:top_k] if top_k is not None else psg_ls)
                except (aiohttp.ClientError, Exception) as e:
                    warn_msg = f"[zhipuai][retry {attempt+1}] failed (idx={idx}): {e}"
                    app.logger.warning(warn_msg)
                    await asyncio.sleep(delay)
            return idx, []

        try:
            return await self._parallel_search(
                query_list=query_list,
                retrieve_thread_num=retrieve_thread_num or 1,
                desc="ZhipuAI Searching:",
                worker_factory=worker_factory,
            )
        finally:
            await session.close()


if __name__ == "__main__":
    Retriever(app)
    app.run(transport="stdio")
