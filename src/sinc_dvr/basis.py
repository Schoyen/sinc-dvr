import collections.abc
import logging
import math
import typing

import numpy as np
import numpy.typing as npt

logger = logging.getLogger(__name__)


type WFArrayType = np.float32 | np.float64 | np.complex64 | np.complex128
IndexType = typing.TypeVar("IndexType", np.int32, np.int64, bound=np.generic)


class SincDVR[IndexType]:
    """Class setting up index arrays for the grid points. The indices cover
    the points from

        `[-num_positive_points[i] * steps[i], num_positive_points[i] * steps[i]]`

    with the origin included. In total this gives

        `2 * num_positive_points[i] + 1`

    points for the grid.


    Parameters
    ----------
    num_positive_points
        The number of grid points on the positive axis. The total number of
        grid points will then be `2 * num_positive_points + 1`, as it should
        be equally many points on the positive and the negative axis, and we
        include the origin.
    steps
        The (positive) step length between each grid point. The extent of the
        grid will be determined from the number of points and the step length.
    """

    def __init__(self, num_positive_points: list[int], steps: list[float]) -> None:
        assert all(npp > 0 for npp in num_positive_points)
        assert all(s > 0 for s in steps)
        num_dim = len(num_positive_points)
        assert len(steps) == num_dim

        num_points_shape = [2 * npp + 1 for npp in num_positive_points]
        num_elements = math.prod(num_points_shape)
        logger.debug(
            f"Number of elements: {num_elements} distributed as: {num_points_shape}"
            "Approximate size requirement for largest (complex128) tensor (in GB): "
            f"{2**num_dim * num_elements * 32 / 1e9}"
        )
        positive_extent = [npp * s for npp, s in zip(num_points_shape, steps)]
        logger.debug(f"Positive spatial extent of grid: {positive_extent}")

        slices = [slice(-npp, npp + 1) for npp in num_positive_points]
        self.index_set = np.ogrid[*slices]
        self.steps = steps

    @staticmethod
    def evaluate_sinc_dvr_spfs(
        index_set: collections.abc.Sequence[npt.NDArray[IndexType]],
        steps: collections.abc.Sequence[float],
        positions: collections.abc.Sequence[npt.NDArray[np.float32 | np.float64]],
    ) -> npt.NDArray[np.float32 | np.float64]:
        """Evaluate positions ((x, y, z)-coordinates) on the grid. This can be
        used for a single point for a single grid point, or a single point on
        all grid point, a collection of points on a single grid point, or an
        equal number of points as the DVR-functions on all the DVR-functions.
        By "grid point" we mean a combination of the index and the step-size.


        Parameters
        ----------
        index_set
            Indices of the DVR grid points/functions.
        steps
            The step size between each DVR grid point for each dimension.
        positions
            The cartesian points that we are evaluating.


        Returns
        -------
        npt.NDArray[np.float32 | np.float64]
            The result of evaluation the points on the DVR functions.
        """
        return math.prod(
            [
                1.0 / np.sqrt(dw) * np.sinc((pos - ind * dw) / dw)
                for ind, dw, pos in zip(index_set, steps, positions)
            ]
        )

    def get_inner_product_function(
        self,
    ) -> collections.abc.Callable[
        [
            npt.NDArray[WFArrayType],
            npt.NDArray[WFArrayType] | None,
        ],
        float | complex,
    ]:
        total_weight = math.prod(self.steps)

        def sinc_dvr_inner_product(
            c: npt.NDArray[WFArrayType],
            total_weight: float = total_weight,
            c_conj: npt.NDArray[WFArrayType] | None = None,
        ) -> float | complex:
            c_conj = c_conj or c.conj().T
            return c_conj @ c * total_weight

        return sinc_dvr_inner_product

    def get_matvec_position_operator(
        self,
        pos_func: npt.NDArray[WFArrayType]
        | collections.abc.Callable[..., npt.NDArray[WFArrayType]],
    ) -> collections.abc.Callable[[npt.NDArray[WFArrayType]], npt.NDArray[WFArrayType]]:
        shape = [len(ind.ravel()) for ind in self.index_set]

        if callable(pos_func):

            def matvec_position_func(
                c: npt.NDArray[WFArrayType],
                *func_args,
                shape=shape,
                pos_func=pos_func,
                **func_kwargs,
            ) -> npt.NDArray[WFArrayType]:
                return (pos_func(*func_args, **func_kwargs) * c.reshape(shape)).ravel()

            return matvec_position_func

        def matvec_position_vec(
            c: npt.NDArray[WFArrayType], shape=shape, pos_func=pos_func
        ):
            return (pos_func * c.reshape(shape)).ravel()

        return matvec_position_vec

    def matvec_kinetic_operator(
        self, c: npt.NDArray[WFArrayType]
    ) -> npt.NDArray[WFArrayType]:
        pass

    def matvec_momentum_operator(
        self, c: npt.NDArray[WFArrayType], axis: int
    ) -> npt.NDArray[WFArrayType]:
        pass

    def get_t_fft_circ(self) -> npt.NDArray[WFArrayType]:
        pass

    @staticmethod
    def setup_t_1d(
        i: int | npt.NDArray[IndexType],
        j: int | npt.NDArray[IndexType],
        step: float,
    ) -> npt.NDArray[WFArrayType]:
        i_min_j = i - j
        with np.errstate(divide="ignore"):
            return np.where(
                i == j,
                np.pi**2 / (6 * step**2),
                (-1.0) ** i_min_j / (step**2 * i_min_j**2),
            )

    @staticmethod
    def setup_p_1d(
        i: int | npt.NDArray[IndexType],
        j: int | npt.NDArray[IndexType],
        step: float,
    ) -> npt.NDArray[WFArrayType]:
        i_min_j = i - j
        with np.errstate(divide="ignore"):
            return (
                -1j
                / step
                * np.where(
                    i == j,
                    0,
                    (-1.0) ** i_min_j / i_min_j,
                )
            )
