"""Nox sessions."""

import tempfile
from typing import Any

import nox
from nox.sessions import Session

package = "volumentations"
nox.options.sessions = "lint", "mypy", "tests"
locations = "src", "tests", "noxfile.py", "docs/conf.py"


def install_with_constraints(session: Session, *args: str, **kwargs: Any) -> None:
    """Install packages constrained by UV's lock file."""
    with tempfile.NamedTemporaryFile() as requirements:
        session.install("uv")
        session.run(
            "uv",
            "pip",
            "compile",
            "pyproject.toml",
            "--extra=dev",
            "--output-file",
            requirements.name,
            external=False,
        )
        if kwargs.get("install_self", False):
            session.install(f"--constraint={requirements.name}", *args, "-e", ".")
        else:
            session.install(f"--constraint={requirements.name}", *args, **kwargs)


@nox.session(python="3.11")
def black(session: Session) -> None:
    """Run black code formatter."""
    args = session.posargs or locations
    install_with_constraints(session, "black")
    session.run("black", *args)


@nox.session(python=["3.8", "3.9", "3.10", "3.11", "3.12", "3.13"])
def lint(session: Session) -> None:
    """Lint using flake8."""
    args = session.posargs or locations
    install_with_constraints(
        session,
        "flake8",
        "flake8-annotations",
        "flake8-bandit",
        "flake8-black",
        "flake8-bugbear",
        "flake8-docstrings",
        "flake8-isort",
        "darglint",
    )
    session.run("flake8", *args)


@nox.session(python=["3.8", "3.9", "3.10", "3.11", "3.12", "3.13"])
def mypy(session: Session) -> None:
    """Type-check using mypy."""
    args = session.posargs or locations
    install_with_constraints(session, "mypy")
    session.run("mypy", "--config-file", "mypy.ini", *args)


@nox.session(python=["3.8", "3.9", "3.10", "3.11", "3.12", "3.13"])
def tests(session: Session) -> None:
    """Run the test suite."""
    args = session.posargs or ["--cov", "-m", "not e2e"]
    install_with_constraints(
        session,
        "coverage[toml]",
        "pytest",
        "pytest-cov",
        "pytest-mock",
        install_self=True,
    )
    session.run("pytest", *args)


@nox.session(python=["3.8", "3.9", "3.10", "3.11", "3.12", "3.13"])
def xdoctest(session: Session) -> None:
    """Run examples with xdoctest."""
    args = session.posargs or ["all"]
    # Install main dependencies and package
    # Install xdoctest
    install_with_constraints(session, "xdoctest", install_self=True)
    session.run("python", "-m", "xdoctest", package, *args)


@nox.session(python="3.11")
def coverage(session: Session) -> None:
    """Upload coverage data."""
    install_with_constraints(session, "coverage[toml]", "codecov")
    session.run("coverage", "xml", "--fail-under=0")
    session.run("codecov", *session.posargs)


@nox.session(python="3.11")
def docs(session: Session) -> None:
    """Build the documentation."""
    # Install docs dependencies
    install_with_constraints(
        session, "sphinx", "sphinx-autodoc-typehints", install_self=True
    )
    session.run("sphinx-build", "docs", "docs/_build")
