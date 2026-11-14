"""Pinned SBML models: downloaded from their source at a fixed commit.

Published models are not copied into a repository. A `PinnedModel` names the
file by a URL with a commit and by the sha256 checksum of its content; `fetch`
downloads it into the cache on first use and verifies the checksum on every
use, so a simulation always runs the model it names.

    model = PinnedModel(
        name="apixaban",
        url="https://raw.githubusercontent.com/matthiaskoenig/apixaban-model/{ref}/models/apixaban_body_flat.xml",
        ref="d0e3277f3ebf389fe8ed8c9bc63be567ba929d3d",
        sha256="eeba5a98213ee4b32c2c2320b61c8409a5f427d59761c406738c6e21e741c6bb",
    )
    sbml_path = fetch(model)

A branch such as `main` instead of a commit is for trying out the current
state of a model: it has no checksum, is downloaded on every call and logs a
warning. Results which must be reproducible use a commit.
"""

import hashlib
import logging
import os
import re
import tempfile
import tomllib
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sbml2jax.cache import cache_dir

logger = logging.getLogger(__name__)

#: a full git commit hash
COMMIT = re.compile(r"^[0-9a-f]{40}$")
#: a sha256 checksum in hexadecimal
SHA256 = re.compile(r"^[0-9a-f]{64}$")
#: a name which is a directory of the cache
NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
#: placeholder of the ref in the URL
REF = "{ref}"
#: schemes a model is downloaded from; `file` serves tests and local mirrors
SCHEMES = ("https", "file")
#: seconds until a download gives up
TIMEOUT = 120


class PinnedModelError(RuntimeError):
    """A pinned model is not defined correctly or cannot be downloaded."""


class ChecksumError(PinnedModelError):
    """The content of a downloaded model has not the pinned checksum."""


@dataclass(frozen=True)
class PinnedModel:
    """An SBML file at a fixed version of its source.

    Attributes:
        name: short name of the model, the directory of the model in the cache.
        url: URL of the SBML file with the placeholder `{ref}`.
        ref: full commit hash, or a branch or tag to try out a model.
        sha256: checksum of the file; required for a commit, not allowed
            otherwise, because the content behind a branch changes.
    """

    name: str
    url: str
    ref: str
    sha256: str | None = None

    def __post_init__(self) -> None:
        if not NAME.match(self.name):
            raise PinnedModelError(f"Model name {self.name!r}: use letters, digits, '.', '_' and '-'")
        if REF not in self.url:
            raise PinnedModelError(f"Model {self.name!r}: the url has no {REF} placeholder: {self.url}")
        if not self.ref or "/" in self.ref or self.ref in (".", ".."):
            raise PinnedModelError(f"Model {self.name!r}: {self.ref!r} is not a commit, branch or tag name")
        scheme = urllib.parse.urlsplit(self.resolved_url).scheme
        if scheme not in SCHEMES:
            raise PinnedModelError(f"Model {self.name!r}: url scheme {scheme!r} is not one of {SCHEMES}")
        if not self.filename:
            raise PinnedModelError(f"Model {self.name!r}: the url names no file: {self.url}")
        if self.pinned and self.sha256 is None:
            raise PinnedModelError(
                f"Model {self.name!r} is pinned to commit {self.ref} but has no sha256. "
                f"Add the checksum of the file, see `sbml2jax.pinned.sha256sum`."
            )
        if self.sha256 is not None and not SHA256.match(self.sha256):
            raise PinnedModelError(
                f"Model {self.name!r}: sha256 must be 64 lowercase hexadecimal digits, got {self.sha256!r}"
            )
        if not self.pinned and self.sha256 is not None:
            raise PinnedModelError(
                f"Model {self.name!r}: ref {self.ref!r} is not a full commit hash, so its content can "
                f"change and a sha256 cannot hold. Use the 40 digit commit hash, or drop the sha256."
            )

    @property
    def pinned(self) -> bool:
        """Whether the ref is a commit, so the content can never change."""
        return COMMIT.match(self.ref) is not None

    @property
    def resolved_url(self) -> str:
        """The URL of the file at the ref."""
        return self.url.replace(REF, self.ref)

    @property
    def filename(self) -> str:
        """Name of the SBML file."""
        return urllib.parse.unquote(urllib.parse.urlsplit(self.resolved_url).path).rsplit("/", 1)[-1]

    def cache_path(self, cache: Path | None = None) -> Path:
        """Path of the file in the cache, `cache_dir()` by default."""
        return (cache or cache_dir()) / "models" / self.name / self.ref / self.filename


def sha256sum(path: Path) -> str:
    """Return the sha256 checksum of a file in hexadecimal."""
    digest = hashlib.sha256()
    with path.open("rb") as f_in:
        for chunk in iter(lambda: f_in.read(1 << 16), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch(model: PinnedModel, cache: Path | None = None) -> Path:
    """Return the path of the SBML file of `model`, downloaded on first use.

    A model pinned to a commit is downloaded once; the checksum is verified
    after the download and on every later call, and a cached file which no
    longer has it is downloaded again. A model on a branch or tag is
    downloaded on every call.

    Args:
        model: the model to fetch.
        cache: cache root, `cache_dir()` by default.

    Returns:
        The path of the SBML file in the cache.

    Raises:
        ChecksumError: if the downloaded file has not the pinned checksum.
        PinnedModelError: if the download fails.
    """
    path = model.cache_path(cache)
    if model.pinned:
        if path.is_file():
            if sha256sum(path) == model.sha256:
                return path
            logger.warning("Cached model %s has not the pinned checksum, downloading it again", path)
    else:
        logger.warning(
            "Model %r follows %r, which is not a commit: its content can change. "
            "Pin a commit and a sha256 for reproducible results.",
            model.name,
            model.ref,
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    # download next to the target and rename only a verified file into place,
    # so the cache never holds a partial or a wrong file
    handle, name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    partial = Path(name)
    try:
        with os.fdopen(handle, "wb") as f_out:
            _download(model, f_out)
        actual = sha256sum(partial)
        if model.pinned and actual != model.sha256:
            raise ChecksumError(
                f"Checksum mismatch for model {model.name!r}.\n"
                f"  url:      {model.resolved_url}\n"
                f"  expected: {model.sha256}\n"
                f"  actual:   {actual}\n"
                f"The file is not the one which was pinned. If the commit was changed on purpose, "
                f"update the sha256 to the actual value; otherwise check the url and the commit."
            )
        os.replace(partial, path)
    finally:
        partial.unlink(missing_ok=True)
    logger.info("Model %r (%s, sha256 %s) cached in %s", model.name, model.ref, actual, path)
    return path


def _download(model: PinnedModel, f_out: Any) -> None:
    """Write the content behind the URL of `model` to the open file `f_out`."""
    url = model.resolved_url
    logger.info("Downloading model %r from %s", model.name, url)
    try:
        # the scheme is restricted to SCHEMES in PinnedModel
        with urllib.request.urlopen(url, timeout=TIMEOUT) as response:  # noqa: S310
            for chunk in iter(lambda: response.read(1 << 16), b""):
                f_out.write(chunk)
    except (urllib.error.URLError, OSError) as err:
        raise PinnedModelError(f"Download of model {model.name!r} from {url} failed: {err}") from err


def read_pinned(path: Path) -> dict[str, PinnedModel]:
    """Read the pinned models of a TOML file.

    Every table of the file is a model with the keys `url`, `ref` and, for a
    commit, `sha256`. Further keys and sub-tables belong to the user of the
    file (for example the settings of a verification) and are ignored here.

    Args:
        path: the TOML file.

    Returns:
        The models by name, in the order of the file.

    Raises:
        PinnedModelError: if the file cannot be read or a model is incomplete.
    """
    try:
        with path.open("rb") as f_in:
            tables: Mapping[str, Any] = tomllib.load(f_in)
    except (OSError, tomllib.TOMLDecodeError) as err:
        raise PinnedModelError(f"Could not read the pinned models in {path}: {err}") from err
    models = {}
    for name, table in tables.items():
        if not isinstance(table, Mapping):
            raise PinnedModelError(f"{path}: {name!r} is not a table of a model")
        missing = [key for key in ("url", "ref") if not isinstance(table.get(key), str)]
        if missing:
            raise PinnedModelError(f"{path}: model {name!r} needs the text keys {missing}")
        sha256 = table.get("sha256")
        if sha256 is not None and not isinstance(sha256, str):
            raise PinnedModelError(f"{path}: the sha256 of model {name!r} must be text")
        models[name] = PinnedModel(name=name, url=table["url"], ref=table["ref"], sha256=sha256)
    return models
