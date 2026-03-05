"""Local LLM inference with HuggingFace Transformers.

Loads a quantized open-weight model that runs on a single RTX 2080 Ti (11GB).
Provides full access to hidden states for memory injection hooks.
"""
from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


class LocalLLM:
    """Manages a local LLM with memory injection hooks."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-7B-Instruct",
        injection_layer: int | None = None,
        device: str = "cuda",
    ) -> None:
        """
        Args:
            model_name: HuggingFace model ID. Good options for 11GB VRAM:
                - "Qwen/Qwen2.5-7B-Instruct" (3584 hidden dim, 28 layers)
                - "mistralai/Mistral-7B-Instruct-v0.3" (4096 hidden dim, 32 layers)
                - "meta-llama/Llama-3.1-8B-Instruct" (4096 hidden dim, 32 layers)
                - For tighter VRAM: "Qwen/Qwen2.5-3B-Instruct" (2048 hidden dim, 36 layers)
            injection_layer: Which transformer layer to inject memory into.
                If None, defaults to middle layer (num_layers // 2).
                Middle layers are best — early layers handle syntax,
                late layers handle output. Middle layers handle semantics.
            device: "cuda" for GPU, "cpu" for CPU (slow but works)
        """
        self.model_name = model_name
        self.device = device
        self.model = None
        self.tokenizer = None
        self.hidden_dim: int = 0
        self.num_layers: int = 0
        self._hook_handle = None
        self._memory_vector: torch.Tensor | None = None
        self._injection_layer = injection_layer
        self._projection = None  # Learned projection from brain dim to LLM dim

    def load(self) -> None:
        """Load the model with 4-bit quantization to fit in 11GB VRAM."""
        print(f"Loading {self.model_name} (4-bit quantized)...")

        # 4-bit quantization config — reduces VRAM from ~14GB to ~4-5GB
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=quantization_config,
            device_map="auto",
            torch_dtype=torch.float16,
        )

        # Determine model architecture details
        config = self.model.config
        self.hidden_dim = config.hidden_size
        self.num_layers = config.num_hidden_layers

        # Default injection at middle layer
        if self._injection_layer is None:
            self._injection_layer = self.num_layers // 2

        print(f"  Model loaded: {self.num_layers} layers, hidden_dim={self.hidden_dim}")
        print(f"  Memory injection at layer {self._injection_layer}")
        print(f"  VRAM: ~{torch.cuda.memory_allocated() / 1e9:.1f}GB")

        # Register the injection hook
        self._register_hook()

    def _get_layer(self, layer_idx: int):
        """Get the transformer layer module by index.

        Different model architectures name their layers differently.
        """
        # Try common attribute paths for different model architectures
        if hasattr(self.model, "model"):
            inner = self.model.model
            # Qwen, Llama, Mistral all use model.model.layers[i]
            if hasattr(inner, "layers"):
                return inner.layers[layer_idx]
        # Fallback: try transformer.h[i] (GPT-style)
        if hasattr(self.model, "transformer"):
            if hasattr(self.model.transformer, "h"):
                return self.model.transformer.h[layer_idx]
        raise ValueError(f"Cannot find transformer layers in {type(self.model)}")

    def _register_hook(self) -> None:
        """Register a forward hook that injects memory into hidden states."""
        layer = self._get_layer(self._injection_layer)

        def injection_hook(module, input, output):
            if self._memory_vector is None:
                return output  # No memory to inject, pass through

            # output is typically a tuple: (hidden_states, ...)
            # or just hidden_states depending on architecture
            if isinstance(output, tuple):
                hidden_states = output[0]
            else:
                hidden_states = output

            # hidden_states shape: [batch_size, seq_len, hidden_dim]
            # Inject memory into the LAST token position
            # (the token being generated — this is where the LLM "thinks"
            # about what to say next)
            memory_projected = self._memory_vector.to(
                hidden_states.device, dtype=hidden_states.dtype
            )

            # Scale injection relative to the LLM's hidden state norm
            # so the memory signal is always proportional to internal scale
            hidden_norm = hidden_states[:, -1, :].norm(dim=-1, keepdim=True)
            memory_projected = memory_projected * hidden_norm * 0.1

            # Add memory signal to last token's hidden state
            hidden_states = hidden_states.clone()
            hidden_states[:, -1, :] += memory_projected

            if isinstance(output, tuple):
                return (hidden_states,) + output[1:]
            return hidden_states

        self._hook_handle = layer.register_forward_hook(injection_hook)

    def set_memory_vector(self, vector: torch.Tensor | None) -> None:
        """Set the memory vector to inject on the next forward pass.

        Args:
            vector: Tensor of shape [hidden_dim] already projected to LLM space.
                    Set to None to disable injection.
        """
        self._memory_vector = vector

    def generate(
        self,
        messages: list[dict],
        max_new_tokens: int = 512,
        temperature: float = 0.7,
    ) -> str:
        """Generate a response from the local LLM.

        Args:
            messages: Chat messages in OpenAI format
                      [{"role": "user", "content": "..."}]
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature

        Returns:
            Generated response text
        """
        # Format messages using the model's chat template
        if hasattr(self.tokenizer, "apply_chat_template"):
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            # Fallback: simple formatting
            text = ""
            for msg in messages:
                role = msg["role"]
                content = msg["content"]
                text += f"<|{role}|>\n{content}\n"
            text += "<|assistant|>\n"

        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        # Decode only the new tokens (not the prompt)
        new_tokens = outputs[0][inputs["input_ids"].shape[1] :]
        response = self.tokenizer.decode(new_tokens, skip_special_tokens=True)

        return response.strip()

    def unload(self) -> None:
        """Free VRAM."""
        if self._hook_handle:
            self._hook_handle.remove()
            self._hook_handle = None
        if self.model:
            del self.model
            self.model = None
        if self.tokenizer:
            del self.tokenizer
            self.tokenizer = None
        torch.cuda.empty_cache()
