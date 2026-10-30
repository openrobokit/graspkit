from typing import (
    Iterator,
    List,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    Union,
    overload,
)

import numpy as np
import torch
from einops import repeat

T = TypeVar("T")


@overload
def batchify_tensor(x: torch.Tensor, batch_size: int) -> Iterator[torch.Tensor]: ...
@overload
def batchify_tensor(x: np.ndarray, batch_size: int) -> Iterator[np.ndarray]: ...
@overload
def batchify_tensor(x: Sequence[T], batch_size: int) -> Iterator[Sequence[T]]: ...
def batchify_tensor(
    x: Union[torch.Tensor, np.ndarray, Sequence[T]], batch_size: int
) -> Iterator[Union[torch.Tensor, np.ndarray, Sequence[T]]]:
    """Split data into sequential batches of specified size.

    Args:
        x: Input data to split into batches. Can be:
            - torch.Tensor: Batched along first dimension
            - np.ndarray: Batched along first dimension
            - Sequence: Any homogeneous sequence
        batch_size: Size of each batch. Must be positive.

    Yields:
        Sequential batches of the input with the same type as input; each batch has size <= batch_size.

    Raises:
        ValueError: If batch_size <= 0
    """
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")

    num_items = len(x)
    for start_idx in range(0, num_items, batch_size):
        end_idx = min(start_idx + batch_size, num_items)
        yield x[start_idx:end_idx]


@overload
def padding_tensor(x: torch.Tensor, target_size: int) -> Tuple[torch.Tensor, torch.Tensor]: ...
@overload
def padding_tensor(x: np.ndarray, target_size: int) -> Tuple[np.ndarray, np.ndarray]: ...
@overload
def padding_tensor(x: List[T], target_size: int) -> Tuple[List[T], List[bool]]: ...
def padding_tensor(
    x: Union[torch.Tensor, np.ndarray, List[T]], target_size: int
) -> Tuple[Union[torch.Tensor, np.ndarray, List[T]], Union[torch.Tensor, np.ndarray, List[bool]]]:
    """Pad data to target size by repeating last element.

    Args:
        x: Input data to pad (torch.Tensor, np.ndarray, or List)
        target_size: Desired size of first dimension after padding

    Returns:
        Tuple of (padded_data, mask) where:
            padded_data: Input padded to target_size by repeating last element
            mask: Boolean mask indicating original (True) vs padded (False) elements
    """
    current_size = len(x) if isinstance(x, list) else x.shape[0]
    if current_size >= target_size:
        if isinstance(x, torch.Tensor):
            return x, torch.ones(current_size, device=x.device, dtype=torch.bool)
        elif isinstance(x, np.ndarray):
            return x, np.ones(current_size, dtype=bool)
        return x, [True] * current_size

    padding_size = target_size - current_size
    last_element = [x[-1]] if isinstance(x, list) else x[-1:]

    if isinstance(x, torch.Tensor):
        padded = torch.cat([x, repeat(last_element, "1 ... -> n ...", n=padding_size)], dim=0)
        mask = torch.ones(target_size, device=x.device, dtype=torch.bool)
        mask[current_size:] = False
    elif isinstance(x, np.ndarray):
        padded = np.concatenate([x, repeat(last_element, "1 ... -> n ...", n=padding_size)], axis=0)
        mask = np.ones(target_size, dtype=bool)
        mask[current_size:] = False
    else:
        padded = x + [x[-1]] * padding_size
        mask = [True] * current_size + [False] * padding_size

    return padded, mask


@overload
def atleast_nd(tensor: None, expected_ndim: int, add_dim_to_front: bool = ...) -> None: ...
@overload
def atleast_nd(tensor: np.ndarray, expected_ndim: int, add_dim_to_front: bool = ...) -> np.ndarray: ...
@overload
def atleast_nd(tensor: torch.Tensor, expected_ndim: int, add_dim_to_front: bool = ...) -> torch.Tensor: ...
def atleast_nd(
    tensor: Optional[Union[np.ndarray, torch.Tensor]], expected_ndim: int, add_dim_to_front: bool = True
) -> Optional[Union[np.ndarray, torch.Tensor]]:
    """Convert input to at least nD tensor; unlike `np.atleast_nd`/`torch.atleast_nd`, can add dims front or back."""
    if tensor is None:
        return None
    actual_ndim = tensor.ndim
    if actual_ndim >= expected_ndim:
        return tensor
    num_dims_to_add = expected_ndim - actual_ndim
    if isinstance(tensor, np.ndarray):
        if add_dim_to_front:
            return np.expand_dims(tensor, axis=tuple(range(num_dims_to_add)))
        else:
            return np.expand_dims(tensor, axis=tuple(range(-num_dims_to_add, 0)))
    elif isinstance(tensor, torch.Tensor):
        if add_dim_to_front:
            return tensor.view((1,) * num_dims_to_add + tensor.shape)
        else:
            return tensor.view(tensor.shape + (1,) * num_dims_to_add)
