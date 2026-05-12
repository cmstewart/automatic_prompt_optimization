from __future__ import annotations
import concurrent.futures
from abc import ABC, abstractmethod
from typing import List, Dict

import utils                      # original OpenAI wrapper
import vectorize                  # builds the Chroma retriever
import threading


class GPT4Predictor(ABC):
    """Minimal interface every predictor must implement."""

    def __init__(self, opt: Dict | None = None):
        self.opt = opt or {}

    
    @abstractmethod
    def inference(self, ex: Dict, prompt: str) -> str:
        pass

    def batch_inference(self, examples: List[Dict], prompt: str) -> List[str]:
        with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
            futures = [
                pool.submit(self.inference, ex, prompt) for ex in examples
            ]
            results = []
            for f in futures:
                try:
                    results.append(f.result())
                except utils.DailyRateLimitError:
                    raise
                except Exception as e:
                    print(f"[WARN] batch_inference thread failed: {e}", flush=True)
                    results.append("")
            return results


# Binary Predictor kept just in case
class BinaryPredictor(GPT4Predictor):

    categories = {0: "No", 1: "Yes"}

    def __init__(self, opt: Dict | None = None):
        super().__init__(opt)
        self.temperature = self.opt.get("temperature", 0.0)

    def _render(self, ex: Dict, prompt: str) -> str:
        return prompt.format(text=ex["text"])

    def inference(self, ex: Dict, prompt: str) -> str:
        rendered = self._render(ex, prompt)
        answer   = utils.chatgpt(
            rendered,
            temperature=self.temperature,
            max_tokens=self.opt.get("max_tokens", 1),
            n=1,
            timeout=10,
        )[0]
        answer = answer.lower()
        if "yes" in answer:
            return "Yes"
        if "no" in answer:
            return "No"
        # default fallback
        return answer.strip()



class QA_Generator(GPT4Predictor):
    def __init__(self, opt=None):
        super().__init__(opt)
        self.top_k = self.opt.get("top_k", 3)
        self.retrievers = {}           # doc_name → retriever
        self.lock = threading.Lock() 

    def get_retriever(self, doc_name):
        if doc_name in self.retrievers:          # fast path
            return self.retrievers[doc_name]

        with self.lock:                          # first builder wins
            if doc_name not in self.retrievers:  # 2nd check inside lock
                r = vectorize.build_vectorstore_retriever(doc_name)
                r.search_kwargs["k"] = self.top_k
                self.retrievers[doc_name] = r
        return self.retrievers[doc_name]

    def inference(self, ex, prompt):
        if "{question}" not in prompt or "{context}" not in prompt:
            raise KeyError("Prompt must contain {question} and {context} placeholders")

        try:
            docs = self.get_retriever(ex["doc_name"]).invoke(ex["question"])
            ctx = "\n".join(d.page_content for d in docs)
        except Exception as e:
            print(f"[WARN] retrieval failed for doc={ex.get('doc_name','?')}: {e}", flush=True)
            return ""
        safe_prompt = prompt.replace("{question}", "\x00Q\x00").replace("{context}", "\x00C\x00")
        safe_prompt = safe_prompt.replace("{", "{{").replace("}", "}}")
        safe_prompt = safe_prompt.replace("\x00Q\x00", "{question}").replace("\x00C\x00", "{context}")
        filled_prompt = safe_prompt.format(question=ex["question"], context=ctx)
        try:
            answer = utils.chatgpt(filled_prompt, temperature=0.0, n=1, timeout=60)[0]
            return answer.strip()
        except utils.DailyRateLimitError:
            raise
        except (RuntimeError, Exception) as e:
            print(f"[WARN] inference failed for doc={ex.get('doc_name','?')}: {e}", flush=True)
            return ""

