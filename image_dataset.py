"""Simple image-label style dataset `ImageDataset`.
Jim's code, used without alteration.
"""
from typing import List, Union, Optional, Tuple, Any
from pathlib import Path
from collections import Counter

from torch.utils.data import Dataset
from torch import Tensor

from torchvision.transforms import Compose


class _ImgBase(Dataset):
    """Base class with helpful attributes and methods for image datasets.

    Parameters
    ----------
    inputs : List[Union[str, Path]]
        List of paths to the images.
    targets : Union[Tensor, List[Union[str, Path]]], optional
        Tensor, or list of path-like, encoding the targets.
        For example, classification tasks may simply require a simple n-hot
        encoded target, whereas segmentation tasks may require paths to
        masks or other forms of ground truths.

    """

    def __init__(
        self,
        inputs: List[Union[str, Path]],
        targets: Union[List[Union[str, Path]], Tensor, None],
    ):
        """Construct ImageDatasetBase."""
        self.inputs = inputs
        self.targets = targets
        self._check_input_and_target_lengths_match()
        self._type_check_inputs()
        self._type_check_targets()

    def _check_input_and_target_lengths_match(self) -> None:
        """Check the length of the inputs and targets match.

        Raises
        ------
        RuntimeError
            If the inputs and targets don't have the same length.

        """
        if self.targets is None:
            return
        if not (num_x := len(self.inputs)) == (num_y := len(self.targets)):
            msg = f"Got {num_x} inputs and {num_y} targets. "
            msg += "The number of inputs should match the number of targets."
            raise RuntimeError(msg)

    def _type_check_inputs(self) -> None:
        """Check the types of the items in `self.inputs`."""
        _ensure_list_of_path_like(self.inputs)

    def _type_check_targets(self) -> None:
        """Check `self.targets` is a Tensor, None or list of path-like.

        Raises
        ------
        TypeError
            If `self.targets` is not a Tensor, None or list.

        """
        if not isinstance(self.targets, (Tensor, list, type(None))):
            msg = "targets should be a Tensor, list or None. "
            msg += f"Got {type(self.targets)}."
            raise TypeError(msg)

        if isinstance(self.targets, list):
            _ensure_list_of_path_like(self.targets)

    def __getitem__(self, index: int) -> Union[Tuple[Any, Any], Any]:
        """Return an input-target pair."""
        raise NotImplementedError

    def __len__(self) -> int:
        """Return the number of items in the dataset.

        Returns
        -------
        int
            Number of entries in the data set.

        """
        return len(self.inputs)


class ImageDataset(_ImgBase):
    """Pytorch dataset for image classification and segmentation experiments.

    Parameters
    ----------
    inputs : List[Union[str, Path]]
        List of paths the images.
    image_transforms : Compose
        Transforms which map an image path to a `Tensor`.
    targets : List[Union[str, Path]] or Tensor, optional
        The ground truths associated with `inputs`. If you are doing inference,
        and don't have any ground truths, simply let `targets = None`
        and `self.__get_item__` with only return images.
    target_transforms : Compose, optional
        Transforms which map a given item in `targets` to a `Tensor`.

    """

    def __init__(
        self,
        inputs: List[Union[str, Path]],
        image_transforms: Compose,
        targets: Optional[Union[List[Union[str, Path]], Tensor]] = None,
        target_transforms: Optional[Compose] = None,
    ):
        """Construct SimpleImageDataset."""
        super().__init__(inputs, targets)
        self._img_tf = image_transforms
        self._trgt_tf = target_transforms
        self._check_transform_types()

    def _check_transform_types(self):
        """Check `self._img_tf` and `self._trgt_df` are of correct types.

        Raises
        ------
        TypeError
            If `self._img_tf` does not have type `Compose`.
        TypeError
            If `self._trgt_tfm` does not have type `Compose` or `type(None)`.

        """
        if not isinstance(self._img_tf, Compose):
            msg = "image_transfroms should have type Compose. "
            msg += f"Got type {type(self._img_tf)}."
            raise TypeError(msg)
        if not isinstance(self._trgt_tf, (Compose, type(None))):
            msg = "target_transforms should have type Compose or None. "
            msg += f"Got {type(self._trgt_tf)}."
            raise TypeError(msg)

    def _prepare_img(self, to_tfm: Union[str, Path]) -> Tensor:
        """Apply the image transfrom to yield a Tensor.

        Parameters
        ----------
        to_tfm : str or Path
            Path to an image we want to load.

        Returns
        -------
        Tensor
            A transformed image ready for training.

        """
        return self._img_tf(to_tfm)

    def _prepare_target(self, to_tfm: Union[str, Path, Tensor]) -> Any:
        """Apply transfrom to target to yield a Tensor."""
        return self._trgt_tf(to_tfm) if self._trgt_tf is not None else to_tfm

    def __getitem__(self, idx: int) -> Union[Tuple[Any, Any], Any]:
        """Return a single input.

        Parameters
        ----------
        idx : int
            Index of the row in the data frame we wish to take the item from.

        Returns
        -------
        x_item : Tensor
            Model input with self.x_tfms applied.
        y_item : Tensor
            Ground truth associated with `x_item`.

        """
        image = self._prepare_img(self.inputs[idx])

        if self.targets is not None:
            target = self._prepare_target(self.targets[idx])
            return image, target

        return image


def _get_unique_list_types(input_list: List[Any]) -> List[Any]:
    """Return the unique types in `input_list`.

    Returns
    -------
    list of Any
        List of the types in `input_list`.

    """
    return list(Counter(map(type, input_list)).keys())


def _ensure_list_of_path_like(in_list: List[Union[str, Path]]) -> None:
    """Raise `TypeError` if `in_list` is not a list of path-like.

    TypeError
    ---------
    If `in_list` is not of type `list`.


    TypeError
    ---------
    If `in_list` does not contain only str or Path.

    """
    if not isinstance(in_list, list):
        msg = f"in_list should have type list. Got {type(in_list)}"
        raise TypeError(msg)

    correct = map(lambda x: isinstance(x, (str, Path)), in_list)
    if not all(correct):
        unique = _get_unique_list_types(in_list)
        msg = f"Input item types should be str or Path, got {unique}."
        raise TypeError(msg)
