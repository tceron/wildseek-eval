from dataclasses import dataclass, field


@dataclass(frozen=True)
class ClassificationTaskConfig:
    name: str
    train_csv: str
    eval_csv: str
    run_csv: str
    default_model_path: str
    default_output_root: str
    default_prediction_output_dir: str
    default_data_path: str
    default_file_path: str
    text_column: str
    source_label_column: str
    label_column: str
    label2id: dict[str, int]
    label_map: dict[str, str] = field(default_factory=dict)
    default_model_sizes: tuple[str, ...] = ("large",)
    default_max_lens: tuple[int, ...] = (256,)
    default_test_size: float = 0.2
    default_seed: int = 42

    @property
    def id2label(self) -> dict[int, str]:
        return {label_id: label for label, label_id in self.label2id.items()}

    @property
    def label_names(self) -> list[str]:
        return list(self.label2id.keys())


TASK_CONFIGS: dict[str, ClassificationTaskConfig] = {
    "highrisk": ClassificationTaskConfig(
        name="highrisk",
        train_csv="../data_prompts/annotated-high-risk.csv",
        eval_csv="../data_prompts/annotated-high-risk.csv",
        run_csv="",
        default_model_path="answerdotai/ModernBERT-large",
        default_output_root="/data1/ceron/modernbert_high-risk-classifier",
        default_prediction_output_dir="./modernbert-output/predictions",
        default_data_path="/data/milanlp/ceron/user_conversations",
        default_file_path="en_SES_history",
        text_column="content",
        source_label_column="high_risk_label",
        label_column="high_risk_label",
        label2id={
            "Other": 0,
            "Economic and Financial": 1,
            "Health": 2,
            "Politics": 3,
            "Judicial and Legal": 4,
            "Moral Values and Religion": 5,
            "Security": 6,
        },
        default_model_sizes=("large",),
        default_max_lens=(256, 384, 512),
        default_test_size=0.2,
        default_seed=42,
    ),
    "openendedness": ClassificationTaskConfig(
        name="openendedness",
        train_csv="../data_prompts/wildseek.csv",
        eval_csv="../data_prompts/wildseek.csv",
        run_csv="",
        default_model_path="answerdotai/ModernBERT-large",
        default_output_root="/data1/ceron/modernbert-information-needs",
        default_prediction_output_dir="./modernbert-output/predictions",
        default_data_path="/data/milanlp/ceron/user_conversations",
        default_file_path="test_set",
        text_column="content",
        source_label_column="factual_or_analytical",
        label_column="ground_truth",
        label2id={
            "Analytical": 0,
            "Factual": 1,
        },
        label_map={
            "Analytical": "Analytical",
            "Subjective": "Analytical",
            "Factual": "Factual",
            "Predictive": "Analytical",
            "Procedural": "Analytical",
        },
        default_model_sizes=("small", "large"),
        default_max_lens=(256, 384, 512),
        default_test_size=0.2,
        default_seed=42,
    ),
    "infoseek": ClassificationTaskConfig(
        name="infoseek",
        train_csv="../data_prompts/annotated-infoseek.csv",
        eval_csv="../data_prompts/annotated-infoseek.csv",
        run_csv="",
        default_model_path="answerdotai/ModernBERT-large",
        default_output_root="/data1/ceron/modernbert-infoseek-classifier",
        default_prediction_output_dir="./modernbert-output/predictions",
        default_data_path="/data/milanlp/ceron/user_conversations",
        default_file_path="en_elisa_history",
        text_column="content",
        source_label_column="ground_truth",
        label_column="ground_truth",
        label2id={
            "information seeking": 0,
            "content creation": 1,
            "coding": 2,
            "not english": 3,
            "no request": 4,
        },
        default_model_sizes=("base", "large"),
        default_max_lens=(256, 384, 512),
        default_test_size=0.2,
        default_seed=42,
    ),
}
