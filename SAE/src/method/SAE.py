"""Minimal SAE support for LLaVA vision tower activations and alignment.

Typical usage:

```python
from SAE.src.method.SAE import train_saes_from_llava_images

result = train_saes_from_llava_images(
    images_bchw,
    model_name="liuhaotian/llava-v1.5-7b",
    layers=[-2],
    latent_dim=16384,
    epochs=5,
)

sae = result.saes[-2]
acts = result.activations[-2]
```
"""

from __future__ import annotations

import dataclasses
import json
import os
import pathlib
import sys
from collections.abc import Sequence
from typing import Any
from typing import Literal

import torch
import torch.nn.functional as F
from PIL import Image
from torch import Tensor
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset

try:
    from SAE.src.model_loader import load_llava_model
except ModuleNotFoundError:
    this_dir = os.path.dirname(__file__)
    src_root = os.path.abspath(os.path.join(this_dir, ".."))
    if src_root not in sys.path:
        sys.path.insert(0, src_root)
    from model_loader import load_llava_model


@dataclasses.dataclass(frozen=True)
class SAEConfig:
    """Configuration for a ReLU SAE with L1 sparsity."""

    input_dim: int
    latent_dim: int
    l1_coeff: float = 1e-3
    dtype: torch.dtype = torch.float32
    device: str | torch.device | None = None


@dataclasses.dataclass(frozen=True)
class TrainerConfig:
    """Lightweight trainer configuration."""

    batch_size: int = 4096
    lr: float = 1e-3
    epochs: int = 5
    shuffle: bool = True
    device: str | torch.device | None = None


@dataclasses.dataclass(frozen=True)
class SAEForwardOutput:
    """Outputs from an SAE forward pass."""

    latent_nl: Tensor
    recon_nd: Tensor
    reconstruction_loss: Tensor
    sparsity_loss: Tensor
    total_loss: Tensor


@dataclasses.dataclass(frozen=True)
class VisionLayerActivations:
    """Patch and optional CLS activations for one vision layer."""

    patch_bsd: Tensor
    flat_nd: Tensor
    cls_bd: Tensor | None


@dataclasses.dataclass(frozen=True)
class SAETrainResult:
    """Bundle returned by the end-to-end helper."""

    tokenizer: Any
    model: Any
    image_processor: Any
    context_len: int
    activations: dict[int, Tensor]
    saes: dict[int, "SparseAutoencoder"]


@dataclasses.dataclass(frozen=True)
class CocoObjectRecord:
    """One supervision record from the COCO JSONL file."""

    image_id: int
    image: str
    pos_object_list: list[str]
    neg_object_list: list[str]


@dataclasses.dataclass(frozen=True)
class ImageAlignmentSample:
    """One image-level pooled SAE sample."""

    image_id: int
    image: str
    image_path: str
    pooled_sae: Tensor
    pos_object_list: list[str]
    neg_object_list: list[str]


@dataclasses.dataclass(frozen=True)
class MentionAlignmentSample:
    """One mention-level sample reusing the pooled image SAE vector."""

    image_id: int
    image: str
    image_path: str
    object_name: str
    label: Literal["supported", "hallucinated"]
    pooled_sae: Tensor


@dataclasses.dataclass(frozen=True)
class PresenceStats:
    """Mean pooled SAE stats for one object class."""

    object_name: str
    present_mean: Tensor
    absent_mean: Tensor
    difference: Tensor
    n_present: int
    n_absent: int


@dataclasses.dataclass(frozen=True)
class MentionStats:
    """Mean pooled SAE stats for supported vs hallucinated mentions."""

    supported_mean: Tensor
    hallucinated_mean: Tensor
    difference: Tensor
    n_supported: int
    n_hallucinated: int


class SparseAutoencoder(torch.nn.Module):
    """Simple ReLU SAE with L1 sparsity on latent activations."""

    def __init__(self, cfg: SAEConfig):
        super().__init__()
        self.cfg = cfg
        self.encoder = torch.nn.Linear(cfg.input_dim, cfg.latent_dim, bias=True)
        self.decoder = torch.nn.Linear(cfg.latent_dim, cfg.input_dim, bias=True)

        if cfg.device is not None or cfg.dtype is not None:
            self.to(device=cfg.device, dtype=cfg.dtype)

    def encode(self, x_nd: Tensor) -> Tensor:
        return F.relu(self.encoder(x_nd))

    def decode(self, latent_nl: Tensor) -> Tensor:
        return self.decoder(latent_nl)

    def forward(self, x_nd: Tensor) -> SAEForwardOutput:
        latent_nl = self.encode(x_nd)
        recon_nd = self.decode(latent_nl)

        reconstruction_loss = F.mse_loss(recon_nd, x_nd)
        # ReLU latents are non-negative, so L1 reduces to the latent mean.
        sparsity_loss = latent_nl.mean() * self.cfg.l1_coeff
        total_loss = reconstruction_loss + sparsity_loss

        return SAEForwardOutput(
            latent_nl=latent_nl,
            recon_nd=recon_nd,
            reconstruction_loss=reconstruction_loss,
            sparsity_loss=sparsity_loss,
            total_loss=total_loss,
        )


class LlavaVisionActivationExtractor:
    """Extract patch-token activations from the local LLaVA vision tower.

    Input tensors must already be preprocessed vision `pixel_values` with shape
    `[batch, 3, height, width]`, for example from the LLaVA image processor.
    """

    def __init__(self, model: Any):
        self.model = model
        # Local LLaVA returns a model wrapper that exposes a CLIP vision tower
        # wrapper via `get_vision_tower()`. That wrapper in turn owns the actual
        # `CLIPVisionModel` at `.vision_tower`.
        self.vision_tower_wrapper = self._get_vision_tower_wrapper(model)
        self.vision_model = self._get_vision_model(self.vision_tower_wrapper)

    @staticmethod
    def _get_vision_tower_wrapper(model: Any) -> Any:
        if hasattr(model, "get_vision_tower"):
            wrapper = model.get_vision_tower()
        elif hasattr(model, "model") and hasattr(model.model, "vision_tower"):
            wrapper = model.model.vision_tower
        else:
            wrapper = None

        if isinstance(wrapper, (list, tuple)):
            wrapper = wrapper[0] if wrapper else None

        msg = "LLaVA model does not expose a vision tower wrapper."
        assert wrapper is not None, msg
        return wrapper

    @staticmethod
    def _get_vision_model(vision_tower_wrapper: Any) -> torch.nn.Module:
        if hasattr(vision_tower_wrapper, "vision_tower"):
            return vision_tower_wrapper.vision_tower
        return vision_tower_wrapper

    @property
    def device(self) -> torch.device:
        return next(self.vision_model.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.vision_model.parameters()).dtype

    def _resolve_layers(self, hidden_states: Sequence[Tensor], layers: Sequence[int]) -> list[int]:
        n_layers = len(hidden_states)
        resolved = []
        for layer in layers:
            layer_i = layer if layer >= 0 else n_layers + layer
            msg = f"Layer index {layer} resolved to {layer_i}, outside [0, {n_layers})."
            assert 0 <= layer_i < n_layers, msg
            resolved.append(layer_i)
        return resolved

    def _split_tokens(
        self,
        layer_bsd: Tensor,
        *,
        include_cls: bool,
    ) -> tuple[Tensor, Tensor | None]:
        seq_len = layer_bsd.shape[1]
        n_patches = getattr(self.vision_tower_wrapper, "num_patches", None)

        if n_patches is None:
            has_cls = seq_len > 1
        elif seq_len == n_patches + 1:
            has_cls = True
        elif seq_len == n_patches:
            has_cls = False
        else:
            raise RuntimeError(
                f"Unexpected vision sequence length {seq_len} for num_patches={n_patches}."
            )

        if has_cls:
            cls_bd = layer_bsd[:, 0, :].detach() if include_cls else None
            patch_bsd = layer_bsd[:, 1:, :].detach()
        else:
            cls_bd = None
            patch_bsd = layer_bsd.detach()

        return patch_bsd, cls_bd

    @torch.no_grad()
    def extract(
        self,
        pixel_values_bchw: Tensor,
        *,
        layers: Sequence[int],
        include_cls: bool = False,
    ) -> dict[int, VisionLayerActivations]:
        """Extract selected hidden states from preprocessed vision inputs.

        Args:
            pixel_values_bchw: Preprocessed vision inputs with shape
                `[batch, 3, height, width]`.
            layers: Hidden-state indices from the CLIP vision model output.
            include_cls: Whether to return CLS token activations when present.
        """
        msg = "pixel_values_bchw must have shape [batch, channels, height, width]."
        assert pixel_values_bchw.ndim == 4, msg

        inputs_bchw = pixel_values_bchw.to(device=self.device, dtype=self.dtype)
        outputs = self.vision_model(inputs_bchw, output_hidden_states=True)
        hidden_states = outputs.hidden_states

        msg = "Vision model did not return hidden states."
        assert hidden_states is not None, msg

        resolved_layers = self._resolve_layers(hidden_states, layers)
        acts_by_layer: dict[int, VisionLayerActivations] = {}

        for requested_layer, resolved_layer in zip(layers, resolved_layers, strict=True):
            layer_bsd = hidden_states[resolved_layer]
            msg = f"Expected [batch, seq, dim] hidden states, got {tuple(layer_bsd.shape)}."
            assert layer_bsd.ndim == 3, msg
            patch_bsd, cls_bd = self._split_tokens(
                layer_bsd,
                include_cls=include_cls,
            )
            flat_nd = patch_bsd.flatten(0, 1)
            msg = f"Extracted non-finite activations for layer {requested_layer}."
            assert torch.isfinite(flat_nd).all(), msg
            acts_by_layer[requested_layer] = VisionLayerActivations(
                patch_bsd=patch_bsd,
                flat_nd=flat_nd,
                cls_bd=cls_bd,
            )

        return acts_by_layer


def train_sae(
    activations_nd: Tensor,
    *,
    sae: SparseAutoencoder | None = None,
    sae_cfg: SAEConfig | None = None,
    trainer_cfg: TrainerConfig | None = None,
) -> SparseAutoencoder:
    """Train an SAE on a tensor of activations with shape [N, d_model]."""
    msg = "activations_nd must have shape [N, d_model]."
    assert activations_nd.ndim == 2, msg
    msg = "SAE training activations contain non-finite values."
    assert torch.isfinite(activations_nd).all(), msg

    trainer_cfg = trainer_cfg or TrainerConfig()
    device = torch.device(trainer_cfg.device or activations_nd.device)

    if sae is None:
        msg = "Need sae_cfg when sae is not provided."
        assert sae_cfg is not None, msg
        sae = SparseAutoencoder(sae_cfg)

    sae = sae.to(device=device, dtype=torch.float32)
    sae.train()

    acts_nd = activations_nd.detach().to(dtype=torch.float32).cpu()
    dataset = TensorDataset(acts_nd)
    dataloader = DataLoader(
        dataset,
        batch_size=trainer_cfg.batch_size,
        shuffle=trainer_cfg.shuffle,
    )
    optimizer = torch.optim.Adam(sae.parameters(), lr=trainer_cfg.lr)

    for epoch in range(trainer_cfg.epochs):
        total_loss = 0.0
        total_recon = 0.0
        total_sparse = 0.0
        n_samples = 0
        n_positive = 0.0
        n_entries = 0
        latent_sum = 0.0
        active_per_sample_sum = 0.0
        feature_fired_l = torch.zeros(
            sae.cfg.latent_dim,
            dtype=torch.bool,
        )

        for (batch_nd,) in dataloader:
            batch_nd = batch_nd.to(device=device, dtype=torch.float32)
            msg = f"Encountered non-finite SAE input batch at epoch {epoch + 1}."
            assert torch.isfinite(batch_nd).all(), msg
            out = sae(batch_nd)
            msg = f"Encountered non-finite SAE losses at epoch {epoch + 1}."
            assert torch.isfinite(out.total_loss), msg
            assert torch.isfinite(out.reconstruction_loss), msg
            assert torch.isfinite(out.sparsity_loss), msg

            optimizer.zero_grad(set_to_none=True)
            out.total_loss.backward()

            for name, param in sae.named_parameters():
                if param.grad is None:
                    continue
                msg = f"Non-finite gradient in parameter '{name}' at epoch {epoch + 1}."
                assert torch.isfinite(param.grad).all(), msg

            optimizer.step()

            for name, param in sae.named_parameters():
                msg = f"Non-finite parameter in '{name}' after optimizer step at epoch {epoch + 1}."
                assert torch.isfinite(param).all(), msg

            batch_size = batch_nd.shape[0]
            n_samples += batch_size
            total_loss += out.total_loss.item() * batch_size
            total_recon += out.reconstruction_loss.item() * batch_size
            total_sparse += out.sparsity_loss.item() * batch_size

            latent_nl = out.latent_nl.detach()
            positive_nl = latent_nl > 0
            n_positive += positive_nl.sum().item()
            n_entries += latent_nl.numel()
            latent_sum += latent_nl.sum().item()
            active_per_sample_sum += positive_nl.sum(dim=1).sum().item()
            feature_fired_l |= positive_nl.any(dim=0).cpu()

        msg = "SAE dataloader yielded zero samples."
        assert n_samples > 0, msg
        msg = "SAE epoch saw zero latent entries."
        assert n_entries > 0, msg
        denom = n_samples
        density = n_positive / n_entries
        active_per_sample = active_per_sample_sum / n_samples
        mean_latent = latent_sum / n_entries
        dead_frac = (~feature_fired_l).float().mean().item()
        print(
            f"[SAE] epoch={epoch + 1}/{trainer_cfg.epochs} "
            f"loss={total_loss / denom:.6f} "
            f"recon={total_recon / denom:.6f} "
            f"sparse={total_sparse / denom:.6f} "
            f"density={density:.6f} "
            f"active_per_sample={active_per_sample:.2f} "
            f"mean_latent={mean_latent:.6f} "
            f"dead_frac={dead_frac:.6f}"
        )

    sae.eval()
    return sae


def train_saes_on_layers(
    activations_by_layer: dict[int, Tensor],
    *,
    latent_dim: int,
    l1_coeff: float = 1e-3,
    trainer_cfg: TrainerConfig | None = None,
    dtype: torch.dtype = torch.float32,
    device: str | torch.device | None = None,
) -> dict[int, SparseAutoencoder]:
    """Train one SAE per selected layer."""
    msg = f"SAE training dtype must be torch.float32, got {dtype}."
    assert dtype == torch.float32, msg
    saes: dict[int, SparseAutoencoder] = {}
    for layer, activations_nd in activations_by_layer.items():
        msg = f"Layer {layer} activations must have shape [N, d_model]."
        assert activations_nd.ndim == 2, msg

        sae_cfg = SAEConfig(
            input_dim=activations_nd.shape[-1],
            latent_dim=latent_dim,
            l1_coeff=l1_coeff,
            dtype=torch.float32,
            device=device,
        )
        saes[layer] = train_sae(
            activations_nd,
            sae_cfg=sae_cfg,
            trainer_cfg=trainer_cfg,
        )
    return saes


def extract_llava_vision_activations(
    model: Any,
    pixel_values_bchw: Tensor,
    *,
    layers: Sequence[int],
    include_cls: bool = False,
) -> dict[int, VisionLayerActivations]:
    """Convenience wrapper for extracting LLaVA vision activations."""
    extractor = LlavaVisionActivationExtractor(model)
    return extractor.extract(
        pixel_values_bchw,
        layers=layers,
        include_cls=include_cls,
    )


def train_saes_from_llava_images(
    pixel_values_bchw: Tensor,
    *,
    model_name: str = "liuhaotian/llava-v1.5-7b",
    model_base: str | None = None,
    model: Any | None = None,
    tokenizer: Any | None = None,
    image_processor: Any | None = None,
    context_len: int | None = None,
    layers: Sequence[int] = (-2,),
    latent_dim: int = 16384,
    l1_coeff: float = 1e-3,
    epochs: int = 5,
    batch_size: int = 4096,
    lr: float = 1e-3,
    device: str = "auto",
    include_cls: bool = False,
) -> SAETrainResult:
    """Load LLaVA, extract vision activations, and train one SAE per layer.

    Note:
        This helper is intentionally in-memory and suitable for small runs.
        `pixel_values_bchw` must already be preprocessed by the LLaVA image
        processor.
    """
    if model is None:
        tokenizer, model, image_processor, context_len = load_llava_model(
            model_name=model_name,
            model_base=model_base,
            device=device,
        )

    msg = "Need tokenizer, image_processor, and context_len after loading the model."
    assert tokenizer is not None and image_processor is not None and context_len is not None, msg

    layer_acts = extract_llava_vision_activations(
        model,
        pixel_values_bchw,
        layers=layers,
        include_cls=include_cls,
    )
    flat_acts = {layer: acts.flat_nd for layer, acts in layer_acts.items()}

    model_device = next(model.parameters()).device
    trainer_cfg = TrainerConfig(
        batch_size=batch_size,
        lr=lr,
        epochs=epochs,
        device=model_device,
    )
    saes = train_saes_on_layers(
        flat_acts,
        latent_dim=latent_dim,
        l1_coeff=l1_coeff,
        trainer_cfg=trainer_cfg,
        dtype=torch.float32,
        device=model_device,
    )

    return SAETrainResult(
        tokenizer=tokenizer,
        model=model,
        image_processor=image_processor,
        context_len=context_len,
        activations=flat_acts,
        saes=saes,
    )


def load_coco_object_records(
    jsonl_fpath: str | pathlib.Path,
    *,
    max_records: int | None = None,
) -> list[CocoObjectRecord]:
    """Load COCO positive/negative object supervision from JSONL."""
    records: list[CocoObjectRecord] = []
    with open(jsonl_fpath, encoding="utf-8") as fd:
        for i, line in enumerate(fd):
            if max_records is not None and i >= max_records:
                break
            dct = json.loads(line)
            records.append(
                CocoObjectRecord(
                    image_id=int(dct["image_id"]),
                    image=str(dct["image"]),
                    pos_object_list=list(dct["pos_object_list"]),
                    neg_object_list=list(dct["neg_object_list"]),
                )
            )
    msg = f"No COCO object records loaded from {jsonl_fpath}."
    assert records, msg
    return records


def topk_token_pool(token_latents_td: Tensor, *, k: int = 8) -> Tensor:
    """Pool token-wise SAE codes into one image-level vector via top-k token mean."""
    msg = f"Expected token latents with shape [num_tokens, latent_dim], got {tuple(token_latents_td.shape)}."
    assert token_latents_td.ndim == 2, msg
    msg = "Need at least one token for pooling."
    assert token_latents_td.shape[0] > 0, msg
    k_use = min(k, token_latents_td.shape[0])
    topk_v = torch.topk(token_latents_td, k_use, dim=0).values
    pooled_d = topk_v.mean(dim=0)
    msg = "Top-k pooled SAE vector contains non-finite values."
    assert torch.isfinite(pooled_d).all(), msg
    return pooled_d


def _preprocess_pil_image(image_processor: Any, image_fpath: pathlib.Path) -> Tensor:
    """Preprocess one image into LLaVA vision pixel values `[1, 3, H, W]`."""
    with Image.open(image_fpath) as img:
        image = img.convert("RGB")

    if hasattr(image_processor, "preprocess"):
        processed = image_processor.preprocess(image, return_tensors="pt")
    else:
        processed = image_processor(images=image, return_tensors="pt")

    if isinstance(processed, dict):
        pixel_values = processed["pixel_values"]
    else:
        pixel_values = processed.pixel_values

    msg = f"Expected preprocessed image shape [1, 3, H, W], got {tuple(pixel_values.shape)}."
    assert pixel_values.ndim == 4 and pixel_values.shape[0] == 1, msg
    return pixel_values


@torch.no_grad()
def make_image_alignment_samples(
    records: Sequence[CocoObjectRecord],
    *,
    sae: SparseAutoencoder,
    model: Any,
    image_processor: Any,
    layer: int,
    image_root: str | pathlib.Path = "SAE/data/MSCOCO/train2014",
    top_k: int = 8,
) -> list[ImageAlignmentSample]:
    """Build pooled image-level SAE samples for object alignment.

    This is intentionally in-memory and suited to small or medium runs.
    """
    image_root = pathlib.Path(image_root)
    extractor = LlavaVisionActivationExtractor(model)
    sae = sae.to(device=extractor.device, dtype=torch.float32)
    sae.eval()

    samples: list[ImageAlignmentSample] = []
    for record in records:
        image_fpath = image_root / record.image
        msg = f"Image missing for alignment: {image_fpath}"
        assert image_fpath.exists(), msg

        pixel_values_bchw = _preprocess_pil_image(image_processor, image_fpath)
        acts = extractor.extract(pixel_values_bchw, layers=[layer], include_cls=False)[
            layer
        ]
        token_latents_td = sae.encode(
            acts.flat_nd.to(device=extractor.device, dtype=torch.float32)
        )
        pooled_d = topk_token_pool(token_latents_td, k=top_k).cpu()
        samples.append(
            ImageAlignmentSample(
                image_id=record.image_id,
                image=record.image,
                image_path=str(image_fpath.resolve()),
                pooled_sae=pooled_d,
                pos_object_list=record.pos_object_list,
                neg_object_list=record.neg_object_list,
            )
        )

    return samples


def make_mention_alignment_samples(
    image_samples: Sequence[ImageAlignmentSample],
) -> list[MentionAlignmentSample]:
    """Expand image-level samples into mention-level samples.

    The same pooled image vector is reused for each mention from that image.
    This is the intended coarse approximation for the current alignment stage.
    """
    mention_samples: list[MentionAlignmentSample] = []
    for sample in image_samples:
        for obj in sample.pos_object_list:
            mention_samples.append(
                MentionAlignmentSample(
                    image_id=sample.image_id,
                    image=sample.image,
                    image_path=sample.image_path,
                    object_name=obj,
                    label="supported",
                    pooled_sae=sample.pooled_sae,
                )
            )
        for obj in sample.neg_object_list:
            mention_samples.append(
                MentionAlignmentSample(
                    image_id=sample.image_id,
                    image=sample.image,
                    image_path=sample.image_path,
                    object_name=obj,
                    label="hallucinated",
                    pooled_sae=sample.pooled_sae,
                )
            )
    return mention_samples


def _stack_vectors(vectors: list[Tensor], *, name: str) -> Tensor:
    msg = f"No vectors available for {name}."
    assert vectors, msg
    stacked = torch.stack(vectors)
    msg = f"Non-finite values found while stacking {name}."
    assert torch.isfinite(stacked).all(), msg
    return stacked


def compute_object_presence_stats(
    image_samples: Sequence[ImageAlignmentSample],
    object_name: str,
) -> PresenceStats:
    """Compute mean pooled SAE stats for one object being present vs absent."""
    present = []
    absent = []
    for sample in image_samples:
        if object_name in sample.pos_object_list:
            present.append(sample.pooled_sae)
        elif object_name in sample.neg_object_list:
            absent.append(sample.pooled_sae)

    present_nd = _stack_vectors(present, name=f"{object_name} present")
    absent_nd = _stack_vectors(absent, name=f"{object_name} absent")
    present_mean = present_nd.mean(dim=0)
    absent_mean = absent_nd.mean(dim=0)
    return PresenceStats(
        object_name=object_name,
        present_mean=present_mean,
        absent_mean=absent_mean,
        difference=present_mean - absent_mean,
        n_present=present_nd.shape[0],
        n_absent=absent_nd.shape[0],
    )


def rank_top_latents(values_d: Tensor, *, top_n: int = 10) -> list[tuple[int, float]]:
    """Return the top latent indices and values from a 1D score tensor."""
    msg = f"Expected 1D latent scores, got {tuple(values_d.shape)}."
    assert values_d.ndim == 1, msg
    n_use = min(top_n, values_d.shape[0])
    vals_n, idx_n = torch.topk(values_d, n_use)
    return [(int(idx.item()), float(val.item())) for val, idx in zip(vals_n, idx_n)]


def top_presence_latents(
    image_samples: Sequence[ImageAlignmentSample],
    object_name: str,
    *,
    top_n: int = 10,
) -> list[tuple[int, float]]:
    """Rank latents by `mean(feature | present) - mean(feature | absent)`."""
    stats = compute_object_presence_stats(image_samples, object_name)
    return rank_top_latents(stats.difference, top_n=top_n)


def compute_supported_vs_hallucinated_stats(
    mention_samples: Sequence[MentionAlignmentSample],
) -> MentionStats:
    """Compute mean pooled SAE difference for hallucinated vs supported mentions."""
    supported = [
        sample.pooled_sae for sample in mention_samples if sample.label == "supported"
    ]
    hallucinated = [
        sample.pooled_sae
        for sample in mention_samples
        if sample.label == "hallucinated"
    ]
    supported_nd = _stack_vectors(supported, name="supported mentions")
    hallucinated_nd = _stack_vectors(hallucinated, name="hallucinated mentions")
    supported_mean = supported_nd.mean(dim=0)
    hallucinated_mean = hallucinated_nd.mean(dim=0)
    return MentionStats(
        supported_mean=supported_mean,
        hallucinated_mean=hallucinated_mean,
        difference=hallucinated_mean - supported_mean,
        n_supported=supported_nd.shape[0],
        n_hallucinated=hallucinated_nd.shape[0],
    )


def top_hallucination_latents(
    mention_samples: Sequence[MentionAlignmentSample],
    *,
    top_n: int = 10,
) -> list[tuple[int, float]]:
    """Rank latents by `mean(feature | hallucinated) - mean(feature | supported)`."""
    stats = compute_supported_vs_hallucinated_stats(mention_samples)
    return rank_top_latents(stats.difference, top_n=top_n)


def _run_dummy_smoke_test() -> None:
    """Cheap validation for the SAE trainer when LLaVA weights are unavailable."""
    acts_nd = torch.randn(256, 64)
    sae = train_sae(
        acts_nd,
        sae_cfg=SAEConfig(input_dim=64, latent_dim=128, l1_coeff=1e-3),
        trainer_cfg=TrainerConfig(batch_size=64, lr=1e-3, epochs=1, device="cpu"),
    )
    out = sae(acts_nd[:8])
    assert out.recon_nd.shape == (8, 64)
    assert out.latent_nl.shape == (8, 128)


def _run_extractor_smoke_test() -> None:
    """Validate extractor assumptions with a fake local LLaVA-style wrapper."""

    class FakeVisionOutputs:
        def __init__(self, hidden_states: tuple[Tensor, ...]):
            self.hidden_states = hidden_states

    class FakeVisionModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.dummy = torch.nn.Parameter(torch.zeros(1))

        def forward(self, pixel_values_bchw: Tensor, output_hidden_states: bool = False):
            assert output_hidden_states
            bsz = pixel_values_bchw.shape[0]
            hidden_states = (
                torch.randn(bsz, 5, 8),
                torch.randn(bsz, 5, 8),
                torch.randn(bsz, 5, 8),
            )
            return FakeVisionOutputs(hidden_states)

    class FakeVisionTowerWrapper:
        def __init__(self):
            self.vision_tower = FakeVisionModel()
            self.num_patches = 4

    class FakeLlavaModel:
        def __init__(self):
            self.wrapper = FakeVisionTowerWrapper()

        def get_vision_tower(self):
            return self.wrapper

    extractor = LlavaVisionActivationExtractor(FakeLlavaModel())
    outs = extractor.extract(torch.randn(2, 3, 224, 224), layers=[-1], include_cls=True)
    out = outs[-1]
    assert out.patch_bsd.shape == (2, 4, 8)
    assert out.flat_nd.shape == (8, 8)
    assert out.cls_bd is not None and out.cls_bd.shape == (2, 8)


def _run_alignment_smoke_test() -> None:
    """Validate the alignment pooling and mean-difference helpers."""
    token_latents_td = torch.tensor(
        [
            [1.0, 0.0, 3.0],
            [2.0, 4.0, 1.0],
            [0.5, 2.0, 5.0],
        ]
    )
    pooled_d = topk_token_pool(token_latents_td, k=2)
    assert pooled_d.shape == (3,)

    image_samples = [
        ImageAlignmentSample(
            image_id=1,
            image="a.jpg",
            image_path="/tmp/a.jpg",
            pooled_sae=torch.tensor([2.0, 0.5, 1.0]),
            pos_object_list=["dog"],
            neg_object_list=["car"],
        ),
        ImageAlignmentSample(
            image_id=2,
            image="b.jpg",
            image_path="/tmp/b.jpg",
            pooled_sae=torch.tensor([0.5, 1.5, 0.5]),
            pos_object_list=["car"],
            neg_object_list=["dog"],
        ),
    ]
    presence = compute_object_presence_stats(image_samples, "dog")
    assert presence.difference.shape == (3,)
    mentions = make_mention_alignment_samples(image_samples)
    mention_stats = compute_supported_vs_hallucinated_stats(mentions)
    assert mention_stats.difference.shape == (3,)


if __name__ == "__main__":
    _run_extractor_smoke_test()
    _run_dummy_smoke_test()
    _run_alignment_smoke_test()
