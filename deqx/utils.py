import typing as tp


def to_rootfind_fun(fpi_fun: tp.Callable):
    """Convert a fixed point function to a rootfind function."""

    def rootfind_fun(z, *args):
        return fpi_fun(z, *args) - z

    return rootfind_fun


def to_fpi_fun(rootfind_fun: tp.Callable):
    """Convert a rootfind function to a fixed point function."""

    def fpi_fun(z, *args):
        return rootfind_fun(z, *args) + z

    return fpi_fun
