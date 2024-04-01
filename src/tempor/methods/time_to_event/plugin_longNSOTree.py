import os
import pickle
import torch
import dataclasses
from functools import partial
from typing import Any, List, Type

from typing_extensions import Self

# Define the model class
import mlflow.pyfunc

from tempor.core import plugins
from tempor.data import data_typing, dataset, samples
from tempor.methods.core.params import Params
from tempor.methods.time_to_event import BaseTimeToEventAnalysis
from tempor.models.longNSOTree import LongitudianlNSOTree_Wrapper, OutputMode, RnnMode

from .helper_embedding import DDHEmbedding


@dataclasses.dataclass
class LongitudinalNSOTreeTimeToEventAnalysisParams:
    """Training params"""
    optimizer: str = "Adam"
    n_iter: int = 1000
    batch_size: int = 100
    lr: float = 1e-3
    """network params"""
    # rnn params
    rnn_mode: RnnMode = "GRU"
    n_layers_hidden: int = 1
    n_units_hidden: int = 40
    # tree params
    tree_hidden_dim: int = 1
    tree_depth: int = 10
    tree_num_back_layer: int = 0
    tree_dense: bool = True
    tree_drop_type: str = 'node_dropconnect'
    tree_net_type: str = 'locally_constant'
    tree_approx: str = 'approx'
    dropout: float = 0.06
    """Number of discrete buckets."""
    split: int = 100
    """loss params"""
    alpha: float = 0.34
    """Weighting (0, 1) likelihood and rank loss (L2 in paper). 1 gives only likelihood, and 0 gives only rank loss."""
    beta: float = 0.27
    """Beta, see paper."""
    sigma: float = 0.21
    """From eta in rank loss (L2 in paper)."""
    device: str = "cuda:0"
    """PyTorch Device."""
    val_size: float = 0.1
    """Early stopping: size of validation set."""
    patience: int = 20
    """Early stopping: training patience without any improvement."""
    output_mode: OutputMode = "MLP"
    """Output network, on of `OutputMode`."""
    random_state: int = 0
    """Random seed."""
    clipping_value:int = 1
    """Gradient clipping."""


@plugins.register_plugin(name="long_nsotree", category="time_to_event")
class DynamicDeepHitTimeToEventAnalysis(BaseTimeToEventAnalysis, DDHEmbedding):
    ParamsDefinition = LongitudinalNSOTreeTimeToEventAnalysisParams
    params: LongitudinalNSOTreeTimeToEventAnalysisParams  # type: ignore
    def __init__(self, **params: Any) -> None:

        super().__init__(**params)

        self.model = LongitudianlNSOTree_Wrapper(
            split=self.params.split,
            # network params
            n_layers_hidden=self.params.n_layers_hidden,
            n_units_hidden=self.params.n_units_hidden,
            rnn_mode=self.params.rnn_mode,
            dropout=self.params.dropout,
            output_mode=self.params.output_mode,
            tree_hidden_dim=self.params.tree_hidden_dim,
            tree_depth=self.params.tree_depth,
            tree_num_back_layer=self.params.tree_num_back_layer,
            tree_dense=self.params.tree_dense,
            tree_drop_type=self.params.tree_drop_type,
            tree_approx=self.params.tree_approx,
            # loss params
            alpha=self.params.alpha,
            beta=self.params.beta,
            sigma=self.params.sigma,
            # training params
            optimizer=self.params.optimizer,
            val_size=self.params.val_size,
            patience=self.params.patience,
            lr=self.params.lr,
            batch_size=self.params.batch_size,
            n_iter=self.params.n_iter,
            clipping_value=self.params.clipping_value,
            device=self.params.device,
        )
        DDHEmbedding.__init__(self, emb_model=self.model)

    def _fit(
        self,
        data: dataset.BaseDataset,
        *args: Any,
        **kwargs: Any,
    ) -> Self:
        processed_data, event_times, event_values = self.prepare_fit(data)
        with mlflow.start_run() as run:
            self.model.fit(processed_data, event_times, event_values)
            model_info = mlflow.pytorch.log_model(self.model.model, "model")
            # print(model_info)
            # print(run.info.run_id)
            # print(run.info)
            with open(os.path.join(run.info.artifact_uri.replace("file://", ""), "model/data", "model.pkl"), "wb") as f:
                pickle.dump(self, f)

        return self

    def _predict(
        self,
        data: dataset.PredictiveDataset,
        horizons: data_typing.TimeIndex,
        *args: Any,
        **kwargs: Any,
    ) -> samples.TimeSeriesSamplesBase:
        # NOTE: kwargs will be passed to DynamicDeepHitModel.predict_risk().
        # E.g. `batch_size` batch size parameter can be provided this way.
        processed_data = self.prepare_predict(data, horizons, *args, **kwargs)
        risk = self.model.predict_risk(processed_data, horizons, **kwargs)
        return samples.TimeSeriesSamples(
            risk.reshape((risk.shape[0], risk.shape[1], 1)),
            sample_index=data.time_series.sample_index(),
            time_indexes=[horizons] * data.time_series.num_samples,  # pyright: ignore
            feature_index=["risk_score"],
        )

    def load_model(self, artifact_uri):
        # self.is_fitted = True
        self._fitted = True
        with open(os.path.join(artifact_uri, "artifacts/model/data", "model.pkl"), "rb") as f:
            loaded_model = pickle.load(f)
            print(loaded_model)
        # check_point = torch.load(model_uri)
        # self.model.model.load_state_dict(check_point)

    @staticmethod
    def hyperparameter_space(*args: Any, **kwargs: Any) -> List[Params]:  # noqa: D102
        return DDHEmbedding.hyperparameter_space(*args, **kwargs)
