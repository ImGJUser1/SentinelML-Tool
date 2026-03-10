import re
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
"""
Hugging Face Transformers adapter for SentinelML.

Provides unified interface for HF models including
text generation, classification, and embeddings.
"""

from typing import Any, Dict, List, Optional, Union

import numpy as np
import numpy.typing as npt

from sentinelml.core.base import BaseSentinelComponent


class HuggingfaceAdapter(BaseSentinelComponent):
    """
    Adapter for Hugging Face Transformers models.

    Unified interface for HF models including
    text generation, classification, and embeddings.

    Parameters
    ----------
    model_name : str
        HuggingFace model identifier (e.g., 'gpt2', 'bert-base-uncased').
    task : str, default='text-generation'
        Task type ('text-generation', 'classification', 'embeddings', 'token-classification').
    device : str, default='auto'
        Device for inference ('cpu', 'cuda', 'cuda:0', 'auto').
    batch_size : int, default=8
        Batch size for inference.
    max_length : int, default=512
        Maximum sequence length.
    torch_dtype : str, optional
        Torch dtype for model loading ('float16', 'bfloat16', 'float32').

    Examples
    --------
    >>> # Text generation
    >>> adapter = HuggingfaceAdapter('gpt2', task='text-generation')
    >>> output = adapter.generate("Hello, how are you?", max_length=50)

    >>> # Classification
    >>> classifier = HuggingfaceAdapter(
    ...     'distilbert-base-uncased-finetuned-sst-2-english',
    ...     task='classification'
    ... )
    >>> preds = classifier.predict(["Great movie!", "Terrible film."])

    >>> # Embeddings
    >>> encoder = HuggingfaceAdapter('sentence-transformers/all-MiniLM-L6-v2', task='embeddings')
    >>> embeddings = encoder.encode(["Text to encode"])
    """

    VALID_TASKS = [
        "text-generation",
        "classification",
        "embeddings",
        "token-classification",
        "question-answering",
        "summarization",
        "translation",
    ]

    def __init__(
        self,
        model_name: str,
        name: str = "HuggingfaceAdapter",
        task: str = "text-generation",
        device: str = "auto",
        batch_size: int = 8,
        max_length: int = 512,
        torch_dtype: Optional[str] = None,
        trust_remote_code: bool = False,
        verbose: bool = False,
        **model_kwargs,
    ):
        super().__init__(name=name, verbose=verbose)
        self.model_name = model_name
        self.task = task
        self.device = self._resolve_device(device)
        self.batch_size = batch_size
        self.max_length = max_length
        self.torch_dtype = torch_dtype
        self.trust_remote_code = trust_remote_code
        self.model_kwargs = model_kwargs

        self._tokenizer = None
        self._model = None
        self._pipeline = None

        # Validate task
        if self.task not in self.VALID_TASKS:
            raise ValueError(f"Invalid task: {task}. Must be one of {self.VALID_TASKS}")

    def _resolve_device(self, device: str) -> str:
        """Resolve device string to actual device."""
        if device == "auto":
            try:
                import torch

                return "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                return "cpu"
        return device

    def fit(self, X=None, y=None):
        """
        Initialize model and tokenizer.

        This loads the model into memory. Call this explicitly
        to control when loading happens, or let it happen
        lazily on first use.
        """
        try:
            import torch
            from transformers import (
                AutoModel,
                AutoModelForCausalLM,
                AutoModelForQuestionAnswering,
                AutoModelForSequenceClassification,
                AutoModelForTokenClassification,
                AutoTokenizer,
                pipeline,
            )
        except ImportError:
            raise ImportError(
                "transformers and torch required. " "Install: pip install transformers torch"
            )

        # Determine torch dtype
        dtype = None
        if self.torch_dtype:
            dtype_map = {
                "float16": torch.float16,
                "bfloat16": torch.bfloat16,
                "float32": torch.float32,
            }
            dtype = dtype_map.get(self.torch_dtype)

        # Load tokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=self.trust_remote_code,
            **{k: v for k, v in self.model_kwargs.items() if k.startswith("tokenizer_")},
        )

        # Set padding token if not present
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        # Load model based on task
        model_kwargs = {
            "trust_remote_code": self.trust_remote_code,
            **{k: v for k, v in self.model_kwargs.items() if not k.startswith("tokenizer_")},
        }

        if dtype:
            model_kwargs["torch_dtype"] = dtype

        if self.task == "text-generation":
            self._model = AutoModelForCausalLM.from_pretrained(self.model_name, **model_kwargs)
            self._pipeline = pipeline(
                "text-generation",
                model=self._model,
                tokenizer=self._tokenizer,
                device=0 if self.device.startswith("cuda") else -1,
            )

        elif self.task == "classification":
            self._model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name, **model_kwargs
            )

        elif self.task == "embeddings":
            self._model = AutoModel.from_pretrained(self.model_name, **model_kwargs)

        elif self.task == "token-classification":
            self._model = AutoModelForTokenClassification.from_pretrained(
                self.model_name, **model_kwargs
            )

        elif self.task == "question-answering":
            self._model = AutoModelForQuestionAnswering.from_pretrained(
                self.model_name, **model_kwargs
            )

        # Move model to device if not using pipeline
        if self._pipeline is None and hasattr(self._model, "to"):
            self._model = self._model.to(self.device)

        if self._model is not None:
            self._model.eval()

        self.is_fitted_ = True

        if self.verbose:
            print(f"Loaded {self.model_name} for {self.task} on {self.device}")

        return self

    def generate(
        self,
        prompts: Union[str, List[str]],
        max_length: Optional[int] = None,
        max_new_tokens: Optional[int] = None,
        temperature: float = 1.0,
        top_p: float = 0.9,
        top_k: int = 50,
        num_return_sequences: int = 1,
        do_sample: bool = True,
        repetition_penalty: float = 1.0,
        pad_token_id: Optional[int] = None,
        **gen_kwargs,
    ) -> List[str]:
        """
        Generate text from prompts.

        Parameters
        ----------
        prompts : str or list of str
            Input prompt(s).
        max_length : int, optional
            Maximum total length (prompt + generation).
        max_new_tokens : int, optional
            Maximum new tokens to generate.
        temperature : float
            Sampling temperature.
        top_p : float
            Nucleus sampling parameter.
        top_k : int
            Top-k sampling.
        num_return_sequences : int
            Number of sequences to return per prompt.
        do_sample : bool
            Whether to use sampling vs greedy decoding.
        repetition_penalty : float
            Penalty for repeating tokens.
        pad_token_id : int, optional
            Padding token ID.

        Returns
        -------
        list of generated texts.
        """
        if self.task != "text-generation":
            raise ValueError(f"generate() only for text-generation task, not {self.task}")

        self._check_is_fitted()

        # Ensure list
        if isinstance(prompts, str):
            prompts = [prompts]

        max_length = max_length or self.max_length

        # Use pipeline if available
        if self._pipeline is not None:
            results = []
            for i in range(0, len(prompts), self.batch_size):
                batch = prompts[i : i + self.batch_size]

                outputs = self._pipeline(
                    batch,
                    max_length=max_length,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    num_return_sequences=num_return_sequences,
                    do_sample=do_sample,
                    repetition_penalty=repetition_penalty,
                    pad_token_id=pad_token_id or self._tokenizer.pad_token_id,
                    **gen_kwargs,
                )

                # Extract generated text
                for item in outputs:
                    if isinstance(item, list):
                        results.extend([o["generated_text"] for o in item])
                    else:
                        results.append(item["generated_text"])

            return results

        # Manual generation
        import torch

        results = []

        for i in range(0, len(prompts), self.batch_size):
            batch = prompts[i : i + self.batch_size]

            inputs = self._tokenizer(
                batch, return_tensors="pt", padding=True, truncation=True, max_length=max_length
            ).to(self.device)

            with torch.no_grad():
                output_ids = self._model.generate(
                    **inputs,
                    max_length=max_length,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    num_return_sequences=num_return_sequences,
                    do_sample=do_sample,
                    repetition_penalty=repetition_penalty,
                    pad_token_id=pad_token_id or self._tokenizer.pad_token_id,
                    **gen_kwargs,
                )

            # Decode
            generated = self._tokenizer.batch_decode(output_ids, skip_special_tokens=True)
            results.extend(generated)

        return results

    def predict(self, texts: Union[str, List[str]]) -> npt.NDArray:
        """
        Classify texts.

        Parameters
        ----------
        texts : str or list of str
            Texts to classify.

        Returns
        -------
        predictions : ndarray
            Class predictions.
        """
        if self.task != "classification":
            raise ValueError(f"predict() only for classification task, not {self.task}")

        self._check_is_fitted()

        if isinstance(texts, str):
            texts = [texts]

        import torch

        predictions = []

        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]

            inputs = self._tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length,
            ).to(self.device)

            with torch.no_grad():
                outputs = self._model(**inputs)
                logits = outputs.logits

                # Get predictions
                batch_preds = torch.argmax(logits, dim=-1).cpu().numpy()
                predictions.extend(batch_preds)

        return np.array(predictions)

    def predict_proba(self, texts: Union[str, List[str]]) -> npt.NDArray:
        """
        Get classification probabilities.

        Parameters
        ----------
        texts : str or list of str
            Texts to classify.

        Returns
        -------
        probabilities : ndarray
            Class probabilities.
        """
        if self.task != "classification":
            raise ValueError(f"predict_proba() only for classification task, not {self.task}")

        self._check_is_fitted()

        if isinstance(texts, str):
            texts = [texts]

        import torch

        all_probs = []

        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]

            inputs = self._tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length,
            ).to(self.device)

            with torch.no_grad():
                outputs = self._model(**inputs)
                logits = outputs.logits

                # Softmax to get probabilities
                probs = torch.softmax(logits, dim=-1).cpu().numpy()
                all_probs.append(probs)

        return np.concatenate(all_probs, axis=0)

    def encode(
        self,
        texts: Union[str, List[str]],
        batch_size: Optional[int] = None,
        normalize: bool = True,
        pooling: str = "mean",
    ) -> npt.NDArray:
        """
        Get embeddings for texts.

        Parameters
        ----------
        texts : str or list of str
            Texts to encode.
        batch_size : int, optional
            Batch size (default: self.batch_size).
        normalize : bool
            Whether to L2 normalize embeddings.
        pooling : str
            Pooling strategy ('mean', 'cls', 'max').

        Returns
        -------
        embeddings : ndarray
            Text embeddings.
        """
        if self.task != "embeddings":
            # Allow encoding for other tasks too
            pass

        self._check_is_fitted()

        if isinstance(texts, str):
            texts = [texts]

        batch_size = batch_size or self.batch_size
        import torch

        embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]

            inputs = self._tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length,
            ).to(self.device)

            with torch.no_grad():
                outputs = self._model(**inputs)

                # Pool embeddings
                if pooling == "cls":
                    # Use [CLS] token
                    batch_embeddings = outputs.last_hidden_state[:, 0]
                elif pooling == "mean":
                    # Mean pooling with attention mask
                    attention_mask = inputs["attention_mask"]
                    token_embeddings = outputs.last_hidden_state
                    input_mask_expanded = attention_mask.unsqueeze(-1).float()
                    sum_embeddings = (token_embeddings * input_mask_expanded).sum(dim=1)
                    batch_embeddings = sum_embeddings / input_mask_expanded.sum(dim=1).clamp(
                        min=1e-9
                    )
                elif pooling == "max":
                    # Max pooling
                    attention_mask = inputs["attention_mask"]
                    token_embeddings = outputs.last_hidden_state
                    mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
                    token_embeddings[mask == 0] = -1e9
                    batch_embeddings = torch.max(token_embeddings, dim=1)[0]
                else:
                    raise ValueError(f"Unknown pooling: {pooling}")

                # Normalize
                if normalize:
                    batch_embeddings = torch.nn.functional.normalize(batch_embeddings, p=2, dim=1)

                embeddings.append(batch_embeddings.cpu().numpy())

        return np.concatenate(embeddings, axis=0)

    def extract_entities(
        self, texts: Union[str, List[str]], aggregation_strategy: str = "simple"
    ) -> List[List[Dict[str, Any]]]:
        """
        Extract named entities from texts.

        Parameters
        ----------
        texts : str or list of str
            Texts to process.
        aggregation_strategy : str
            How to aggregate tokens ('none', 'simple', 'first', 'average', 'max').

        Returns
        -------
        list of entity lists per text.
        """
        if self.task != "token-classification":
            raise ValueError(
                f"extract_entities() requires token-classification task, not {self.task}"
            )

        self._check_is_fitted()

        if isinstance(texts, str):
            texts = [texts]

        try:
            from transformers import pipeline

            ner_pipeline = pipeline(
                "ner",
                model=self._model,
                tokenizer=self._tokenizer,
                aggregation_strategy=aggregation_strategy,
                device=0 if self.device.startswith("cuda") else -1,
            )

            results = ner_pipeline(texts)
            return results if isinstance(results, list) else [results]

        except Exception as e:
            if self.verbose:
                print(f"NER extraction failed: {e}")
            return [[] for _ in texts]

    def answer_question(
        self, questions: Union[str, List[str]], contexts: Union[str, List[str]]
    ) -> List[Dict[str, Any]]:
        """
        Answer questions based on context.

        Parameters
        ----------
        questions : str or list of str
            Questions.
        contexts : str or list of str
            Context passages.

        Returns
        -------
        list of answers with scores.
        """
        if self.task != "question-answering":
            raise ValueError(f"answer_question() requires question-answering task, not {self.task}")

        self._check_is_fitted()

        # Ensure lists
        if isinstance(questions, str):
            questions = [questions]
        if isinstance(contexts, str):
            contexts = [contexts] * len(questions)

        if len(questions) != len(contexts):
            raise ValueError("questions and contexts must have same length")

        try:
            from transformers import pipeline

            qa_pipeline = pipeline(
                "question-answering",
                model=self._model,
                tokenizer=self._tokenizer,
                device=0 if self.device.startswith("cuda") else -1,
            )

            inputs = [{"question": q, "context": c} for q, c in zip(questions, contexts)]
            results = qa_pipeline(inputs)

            return results if isinstance(results, list) else [results]

        except Exception as e:
            if self.verbose:
                print(f"QA failed: {e}")
            return [{"answer": "", "score": 0.0} for _ in questions]

    def get_tokenizer(self):
        """Return the tokenizer."""
        self._check_is_fitted()
        return self._tokenizer

    def get_model(self):
        """Return the underlying model."""
        self._check_is_fitted()
        return self._model

    def save_pretrained(self, save_directory: str):
        """
        Save model and tokenizer.

        Parameters
        ----------
        save_directory : str
            Directory to save to.
        """
        self._check_is_fitted()

        import os

        os.makedirs(save_directory, exist_ok=True)

        if self._model is not None:
            self._model.save_pretrained(save_directory)
        if self._tokenizer is not None:
            self._tokenizer.save_pretrained(save_directory)

        if self.verbose:
            print(f"Saved to {save_directory}")

    def __repr__(self):
        return (
            f"HuggingfaceAdapter("
            f"model_name='{self.model_name}', "
            f"task='{self.task}', "
            f"device='{self.device}')"
        )
