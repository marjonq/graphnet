"""Class(es) for deploying GraphNeT models in icetray as I3Modules."""
from abc import abstractmethod
from typing import TYPE_CHECKING, Any, List, Union, Dict, Tuple

import dill
import numpy as np
import torch
from torch_geometric.data import Data

from graphnet.data.extractors import (
    I3FeatureExtractor,
    I3FeatureExtractorIceCubeUpgrade,
)
from graphnet.models import Model, StandardModel
from graphnet.utilities.imports import has_icecube_package

if has_icecube_package() or TYPE_CHECKING:
    from icecube.icetray import (
        I3Module,
        I3Frame,
    )  # pyright: reportMissingImports=false
    from icecube.dataclasses import (
        I3Double,
        I3MapKeyVectorDouble,
    )  # pyright: reportMissingImports=false
    from icecube import dataclasses, dataio, icetray


class GraphNeTI3Module:
    """Base I3 Module for GraphNeT.

    Contains methods for extracting pulsemaps, producing graphs and writing to
    frames.
    """

    def __init__(
        self,
        pulsemap: str,
        features: List[str],
        pulsemap_extractor: Union[
            List[I3FeatureExtractor], I3FeatureExtractor
        ],
    ):
        """Add member variables and extractors."""
        self._pulsemap = pulsemap
        self._features = features
        if isinstance(pulsemap_extractor, list):
            self._i3_extractors = pulsemap_extractor
        else:
            self._i3_extractors = [pulsemap_extractor]

    @abstractmethod
    def __call__(self, frame: I3Frame) -> bool:
        """Define here how the module acts on the frame.

        Must return True if successful.
        """
        return True

    def _make_graph(
        self, frame: I3Frame
    ) -> Data:  # py-l-i-n-t-:- -d-i-s-able=invalid-name
        """Process Physics I3Frame into graph."""
        # Extract features
        features = self._extract_feature_array_from_frame(frame)

        # Prepare graph data
        n_pulses = torch.tensor([features.shape[0]], dtype=torch.int32)
        data = Data(
            x=torch.tensor(features, dtype=torch.float32),
            edge_index=None,
            batch=torch.zeros(
                features.shape[0], dtype=torch.int64
            ),  # @TODO: Necessary?
            features=self._features,
        )
        # @TODO: This sort of hard-coding is not ideal; all features should be
        #        captured by `FEATURES` and included in the output of
        #        `I3FeatureExtractor`.
        data.n_pulses = n_pulses
        return data

    def _extract_feature_array_from_frame(self, frame: I3Frame) -> np.array:
        features = None
        for i3extractor in self._i3_extractors:
            feature_dict = i3extractor(frame)
            features_pulsemap = np.array(
                [feature_dict[key] for key in self._features]
            ).T
            if features is None:
                features = features_pulsemap
            else:
                features = np.concatenate(
                    (features, features_pulsemap), axis=0
                )
        return features

    def _submit_to_frame(
        self, frame: I3Frame, data: Dict[str, Any]
    ) -> I3Frame:
        """Write every field of data to frame."""
        assert isinstance(
            data, dict
        ), f"data must be of type dict. Got {type(data)}"
        for key in data.keys():
            frame.Put(key, data[key])
        return frame


class I3InferenceModule(GraphNeTI3Module):
    """General class for inference on i3 frames."""

    def __init__(
        self,
        pulsemap: str,
        features: List[str],
        pulsemap_extractor: Union[
            List[I3FeatureExtractor], I3FeatureExtractor
        ],
        model: Union[Model, StandardModel, str],
        model_name: str,
        prediction_columns: Union[List[str], str],
    ):
        """Add member variables and extractors."""
        super().__init__(
            pulsemap=pulsemap,
            features=features,
            pulsemap_extractor=pulsemap_extractor,
        )

        if isinstance(model, str):
            self.model = torch.load(
                model, pickle_module=dill, map_location="cpu"
            )
        else:
            self.model = model

        if isinstance(prediction_columns, str):
            self.prediction_columns = [prediction_columns]
        else:
            self.prediction_columns = prediction_columns

        self.model_name = model_name

    def __call__(self, frame: I3Frame) -> bool:
        """Write predictions from model to frame."""
        # inference
        graph = self._make_graph(frame)
        predictions = self._inference(graph)

        # Check dimensions of predictions and prediction columns
        if len(predictions.shape) > 1:
            dim = predictions.shape[1]
        else:
            dim = len(predictions)
        assert dim == len(
            self.prediction_columns
        ), f"predictions have shape {dim} but prediction columns have [{self.prediction_columns}]"

        # Build Dictionary of predictions
        data = {}
        for i in range(len(dim)):
            try:
                data[
                    self.model_name + "_" + self.prediction_columns[i]
                ] = icetray.I3Double(predictions[:, i])
            except IndexError:
                data[
                    self.model_name + "_" + self.prediction_columns[i]
                ] = icetray.I3Double(predictions)

        # Submission methods
        frame = self._submit_to_frame(frame=frame, data=data)
        return True

    def _inference(self, data: Data) -> np.ndarray:
        # Perform inference
        try:
            predictions = [p.detach().numpy()[0, :] for p in self.model(data)]
            predictions = np.concatenate(
                predictions
            )  # @TODO: Special case for single task
        except:  # noqa: E722
            print("data:", data)
            raise
        return predictions


class I3PulseCleanerModule(I3InferenceModule):
    """A specialized module for pulse cleaning.

    It is assumed that the model provided has been trained for this.
    """

    def __init__(
        self,
        pulsemap: str,
        features: List[str],
        pulsemap_extractor: Union[
            List[I3FeatureExtractor], I3FeatureExtractor
        ],
        model: Union[Model, StandardModel, str],
        model_name: str,
        prediction_columns: Union[List[str], str] = "",
        *,
        gcd_file: str,
        threshold: float = 0.7,
    ):
        """Add member variables and extractors."""
        super().__init__(
            pulsemap=pulsemap,
            features=features,
            pulsemap_extractor=pulsemap_extractor,
            model=model,
            model_name=model_name,
            prediction_columns=prediction_columns,
        )

        assert isinstance(gcd_file, str), "gcd_file must be string"
        self._gcd_file = gcd_file
        self._threshold = threshold
        self._predictions_key = f"{pulsemap}_{model_name}_Predictions"
        self._total_pulsemap_name = f"{pulsemap}_{model_name}_Pulses"

    def __call__(self, frame: I3Frame) -> bool:
        """Add a cleaned pulsemap to frame."""
        # inference
        gcd_file = self._gcd_file
        graph = self._make_graph(frame)
        predictions = self._inference(graph)

        assert predictions.shape[1] == 1

        # Build Dictionary of predictions
        data = {}

        predictions_map = self._construct_prediction_map(
            frame=frame, predictions=predictions
        )

        # Adds the raw predictions to dictionary
        if self._predictions_key not in frame.keys():
            data[self._predictions_key] = predictions_map

        # Create a pulse map mask, indicating the pulses that are over threshold (e.g. identified as signal) and therefore should be kept
        # Using a lambda function to evaluate which pulses to keep by checking the prediction for each pulse
        # (Adds the actual pulsemap to dictionary)
        if self._total_pulsemap_name not in frame.keys():
            data[
                self._total_pulsemap_name
            ] = dataclasses.I3RecoPulseSeriesMapMask(
                frame,
                self._pulsemap,
                lambda om_key, index, pulse: predictions_map[om_key][index]
                >= self._threshold,
            )

        # Adds an additional pulsemap for each DOM type
        if isinstance(
            self._i3_extractors[0], I3FeatureExtractorIceCubeUpgrade
        ):
            mDOMMap, DEggMap, IceCubeMap = self._split_pulsemap_in_dom_types(
                frame=frame, gcd_file=gcd_file
            )

            if f"{self._total_pulsemap_name}_mDOMs_Only" not in frame.keys():
                data[
                    f"{self._total_pulsemap_name}_mDOMs_Only"
                ] = dataclasses.I3RecoPulseSeriesMap(mDOMMap)

            if f"{self._total_pulsemap_name}_dEggs_Only" not in frame.keys():
                data[
                    f"{self._total_pulsemap_name}_dEggs_Only"
                ] = dataclasses.I3RecoPulseSeriesMap(DEggMap)

            if f"{self._total_pulsemap_name}_pDOMs_Only" not in frame.keys():
                data[
                    f"{self._total_pulsemap_name}_pDOMs_Only"
                ] = dataclasses.I3RecoPulseSeriesMap(IceCubeMap)

        # Submits the dictionary to the frame
        frame = self._submit_to_frame(frame=frame, data=data)

        return True

    def _split_pulsemap_in_dom_types(
        self, frame: I3Frame, gcd_file: Any
    ) -> Tuple[Dict[Any, Any], Dict[Any, Any], Dict[Any, Any]]:
        g = dataio.I3File(gcd_file)
        gFrame = g.pop_frame()
        while "I3Geometry" not in gFrame.keys():
            gFrame = g.pop_frame()
        omGeoMap = gFrame["I3Geometry"].omgeo

        mDOMMap, DEggMap, IceCubeMap = {}, {}, {}
        pulses = dataclasses.I3RecoPulseSeriesMap.from_frame(
            frame, self._total_pulsemap_name
        )
        for P in pulses:
            om = omGeoMap[P[0]]
            if om.omtype == 130:  # "mDOM"
                mDOMMap[P[0]] = P[1]
            elif om.omtype == 120:  # "DEgg"
                DEggMap[P[0]] = P[1]
            elif om.omtype == 20:  # "IceCube / pDOM"
                IceCubeMap[P[0]] = P[1]
        return mDOMMap, DEggMap, IceCubeMap

    def _construct_prediction_map(
        self, frame: I3Frame, predictions: np.ndarray
    ) -> I3MapKeyVectorDouble:
        pulsemap = dataclasses.I3RecoPulseSeriesMap.from_frame(
            frame, self._pulsemap
        )

        idx = 0
        predictions_map = dataclasses.I3MapKeyVectorDouble()
        for om_key, pulses in pulsemap.items():
            num_pulses = len(pulses)
            predictions_map[om_key] = predictions[
                idx : idx + num_pulses
            ].tolist()
            idx += num_pulses

        # Checks
        assert idx == len(
            predictions
        ), "Not all predictions were mapped to pulses, validation of predictions have failed."

        assert (
            pulsemap.keys() == predictions_map.keys()
        ), "Input pulse map and predictions map do not contain exactly the same OMs"
        return predictions_map
