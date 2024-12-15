import pandas as pd
import os
import warnings

from sklearn.preprocessing import LabelEncoder
from optuna.samplers import RandomSampler
from optuna.trial import FixedTrial
from optuna.distributions import BaseDistribution
from optuna import distributions
from typing import Optional, Any

def read_data(base_path, datasets_list=["Base"], seed=None):
    datasets_paths = {
        key : f"{base_path}{key}.parquet" if key == "Base" else f"{base_path}Variant {key.split()[-1]}.parquet" for key in datasets_list
    }
    for key in datasets_paths.keys():
        if os.path.exists(datasets_paths[key]):
            print(f"Dataset {key} already exists")
            continue
        df = pd.read_csv(datasets_paths[key].replace(".parquet", ".csv"))
        df.to_parquet(datasets_paths[key])
        print(f"Dataset {key} saved as parquet")
    datasets = {key: pd.read_parquet(path) for key, path in datasets_paths.items()}
    categorical_features = [
        "payment_type",
        "employment_status",
        "housing_status",
        "source",
        "device_os",
    ]
    train_dfs = {key: df[df["month"]<6].sample(frac=1, replace=False, random_state=seed) for key, df in datasets.items()}
    test_dfs = {key: df[df["month"]>=6].sample(frac=1, replace=False, random_state=seed)  for key, df in datasets.items()}
    for name in datasets.keys(): 
        train = train_dfs[name]
        test = test_dfs[name]
        for feat in categorical_features:
            encoder = LabelEncoder()
            encoder.fit(train[feat]) 
            train[feat] = encoder.transform(train[feat])
            test[feat] = encoder.transform(test[feat])  
    return datasets_paths, datasets, train_dfs, test_dfs





class RandomValueTrial(FixedTrial):
    """A Trial following optuna's API.
    Does not depend on an optuna.Study and can be used as a standalone object.
    """
    def __init__(
            self,
            number: int = 0,
            seed: Optional[int] = None,
            sampler: Optional[RandomSampler] = None
        ):
        assert not (seed and sampler), \
            f"Must provide at most one of (seed={seed}, sampler={sampler})"
        super().__init__(
            params=None, 
            number=number,
        )
        self.seed = seed
        self.sampler = sampler or RandomSampler(self.seed)

    def _suggest(self, name: str, distribution: BaseDistribution) -> Any:
        """Given a parameter's name and its distribution should return the
        suggested (sampled) value.
        (Template method from the Trial super-class).

        Parameters
        ----------
        name : str
            The parameter's name (so we don't suggest different values when the
            same parameter is sampled more than once).
        distribution : BaseDistribution
            The distribution to draw from.

        Returns
        -------
        The sampled value.
        """

        if name in self._distributions:
            # No need to sample if already suggested.
            distributions.check_distribution_compatibility(
                self._distributions[name], distribution,
            )
            param_value = self._suggested_params[name]

        else:
            if self._is_fixed_param(name, distribution):
                param_value = self.system_attrs["fixed_params"][name]
            elif distribution.single():
                param_value = distributions._get_single_value(distribution)
            else:
                param_value = self.sampler.sample_independent(
                    study=None, trial=self,     # type: ignore
                    param_name=name, param_distribution=distribution,
                )

        self._suggested_params[name] = param_value
        self._distributions[name] = distribution

        return self._suggested_params[name]

    def _is_fixed_param(self, name: str, distribution: BaseDistribution) -> bool:
        """Checks if the given parameter name corresponds to a fixed parameter.
        This implementation does not depend on an optuna.study.

        Parameters
        ----------
        name : str
        distribution : BaseDistribution

        Returns
        -------
        Whether the parameter is a fixed parameter.
        """
        system_attrs = self._system_attrs
        if "fixed_params" not in system_attrs:
            return False

        if name not in system_attrs["fixed_params"]:
            return False

        param_value = system_attrs["fixed_params"][name]
        param_value_in_internal_repr = distribution.to_internal_repr(param_value)

        contained = distribution._contains(param_value_in_internal_repr)
        if not contained:
            warnings.warn(
                f"Fixed parameter '{name}' with value {param_value} is out of range "
                f"for distribution {distribution}."
            )
        return contained