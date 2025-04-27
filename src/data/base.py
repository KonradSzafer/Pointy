from typing import Callable

import torch


class RandomChoice:
    def __init__(self, transforms: list[tuple[Callable, dict[str, any], float]] = None):
        """
        Apply one of the transformations with a given probability.
        Probabilities of the transformations should sum to 1.
        Each transform is a tuple of:
            - A callable function
            - A dictionary of keyword arguments for the function (optional)
            - A probability of applying the transform
        """
        if transforms is not None:
            probabilities = [transform[-1] for transform in transforms]
            if sum(probabilities) > 1:
                raise ValueError(
                    "Probabilities of the transformations should be less than or equal to 1"
                )

        self.transforms = transforms

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if self.transforms is None:
            return x

        # Choose a transform based on the probabilities
        probabilities = [transform[-1] for transform in self.transforms]
        index = torch.multinomial(torch.tensor(probabilities), 1).item()
        func, args, _ = self.transforms[index]
        return func(x, **args)


class TransformsCompose:
    def __init__(self, transforms: list[tuple[Callable, dict[str, any], float]] = None):
        """
        Compose a list of transformations to be applied to the input data.
        Transforms are applied in the order they are passed in the list.
        Each transform is a tuple of:
            - A callable function
            - A dictionary of keyword arguments for the function (optional)
            - A probability of applying the transform
        """
        self.transforms = transforms

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if self.transforms is None:
            return x

        for transform in self.transforms:
            func, args, probability = transform
            if torch.rand(1) < probability:
                x = func(x, **args)
        return x
