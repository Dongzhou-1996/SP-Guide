from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal

from gsamllavanav.models.cma_with_map import CMAwithMap
from gsamllavanav.models.goal_predictor import GoalPredictor
from gsamllavanav.models.instr_decoder_with_map import (
    InstructionQueryDecoderWithMap,
    InstructionQueryDecoderWithUSCMap,
)
from gsamllavanav.models.seq2seq_with_map import Seq2SeqwithMap


ModelPipeline = Literal["goal_predictor", "baseline_with_map"]


@dataclass(frozen=True)
class ModelSpec:
    name: str
    factory: Callable[[int], object]
    pipeline: ModelPipeline
    description: str


MODEL_SPECS: dict[str, ModelSpec] = {
    "mgp": ModelSpec(
        name="mgp",
        factory=GoalPredictor,
        pipeline="goal_predictor",
        description="CityNav official map-guided predictor (MGP).",
    ),
    "seq2seq_with_map": ModelSpec(
        name="seq2seq_with_map",
        factory=Seq2SeqwithMap,
        pipeline="baseline_with_map",
        description="Seq2Seq baseline with GSM map input.",
    ),
    "cma_with_map": ModelSpec(
        name="cma_with_map",
        factory=CMAwithMap,
        pipeline="baseline_with_map",
        description="CMA baseline with GSM map input.",
    ),
    "instr_decoder_with_map": ModelSpec(
        name="instr_decoder_with_map",
        factory=InstructionQueryDecoderWithMap,
        pipeline="baseline_with_map",
        description="Stable instruction-as-query decoder baseline.",
    ),
    "instr_decoder_usc_with_map": ModelSpec(
        name="instr_decoder_usc_with_map",
        factory=InstructionQueryDecoderWithUSCMap,
        pipeline="baseline_with_map",
        description="Instruction decoder with unified spatial constraint (USC).",
    ),
}


ALL_MODEL_NAMES = tuple(MODEL_SPECS.keys())
BASELINE_WITH_MAP_MODELS = tuple(
    name for name, spec in MODEL_SPECS.items() if spec.pipeline == "baseline_with_map"
)
GOAL_PREDICTOR_MODELS = tuple(
    name for name, spec in MODEL_SPECS.items() if spec.pipeline == "goal_predictor"
)


def get_model_spec(name: str) -> ModelSpec:
    return MODEL_SPECS[name]
