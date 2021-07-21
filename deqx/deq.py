import typing as tp

import haiku as hk

from deqx import newton as newton_lib


class SolverType:
    FPI = "fpi"
    ROOT = "root"

    @classmethod
    def all(cls):
        return (SolverType.FPI, SolverType.ROOT)

    @classmethod
    def validate(cls, solver_type: str):
        valid = cls.all()
        if solver_type not in valid:
            raise ValueError(f"solver_type must be one of {valid}, got {solver_type}")


class DEQ(hk.Module):
    def __init__(
        self,
        fpi_fun: tp.Callable,
        solver_factory: tp.Callable = newton_lib.newton_with_vjp,
        solver_type: str = SolverType.ROOT,
        name: tp.Optional[str] = None,
    ):
        super().__init__(name=name)
        SolverType.validate(solver_type)
        self.solver_type = solver_type
        self.fpi_fun = fpi_fun
        self.solver_factory = solver_factory

    def __call__(self, z0, *args):
        transform = hk.transform(self.fpi_fun)

        def transform_fun(z, params, rng, *transform_args):
            assert not isinstance(z, tuple), (z, *transform_args)
            out = transform.apply(params, rng, z, *transform_args)
            if self.solver_type == SolverType.FPI:
                return out
            if self.solver_type == SolverType.ROOT:
                return out - z
            raise ValueError(
                f"Invalid solver_type '{self.solver_type}' - "
                f"must be in {SolverType.all()}"
            )

        inner_params = hk.experimental.lift(transform.init)(
            hk.next_rng_key(), z0, *args
        )

        equilibrium, info = self.solver_factory(transform_fun)(
            z0, inner_params, hk.next_rng_key(), *args
        )
        del info
        return equilibrium
