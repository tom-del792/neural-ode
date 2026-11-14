"""Pinned models: download into the cache, checksum, errors.

No network: the "remote" is a directory with one subdirectory per ref, reached
through `file://` urls, and the model is the absorption model of the
documentation.
"""

import logging
import shutil
from pathlib import Path

import pytest

from sbml2jax import pinned
from sbml2jax.cache import CACHE_ENV, cache_dir
from sbml2jax.model import JaxModel
from sbml2jax.pinned import ChecksumError, PinnedModel, PinnedModelError, fetch, read_pinned, sha256sum

from .conftest import ABSORPTION

COMMIT = "0123456789abcdef0123456789abcdef01234567"
SHA256 = sha256sum(ABSORPTION)


@pytest.fixture
def remote(tmp_path: Path) -> Path:
    """Directory which serves the absorption model at the commit and on main."""
    root = tmp_path / "remote"
    for ref in (COMMIT, "main"):
        (root / ref).mkdir(parents=True)
        shutil.copy(ABSORPTION, root / ref / "absorption.xml")
    return root


@pytest.fixture
def cache(tmp_path: Path) -> Path:
    return tmp_path / "cache"


def url(remote: Path) -> str:
    return f"file://{remote.as_posix()}/{{ref}}/absorption.xml"


def model(remote: Path, ref: str = COMMIT, sha256: str | None = SHA256) -> PinnedModel:
    return PinnedModel(name="absorption", url=url(remote), ref=ref, sha256=sha256)


def test_fetch_downloads_into_cache(remote: Path, cache: Path) -> None:
    path = fetch(model(remote), cache)
    assert path == cache / "models" / "absorption" / COMMIT / "absorption.xml"
    assert path.read_bytes() == ABSORPTION.read_bytes()
    assert [p.name for p in path.parent.iterdir()] == ["absorption.xml"], "no partial download is left"
    assert "C" in JaxModel.from_sbml(path).module.OUTPUTS


def test_fetch_uses_cache_without_remote(remote: Path, cache: Path) -> None:
    path = fetch(model(remote), cache)
    shutil.rmtree(remote)
    assert fetch(model(remote), cache) == path


def test_cache_root_from_environment(remote: Path, cache: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(CACHE_ENV, str(cache))
    assert cache_dir() == cache
    assert fetch(model(remote)).is_relative_to(cache)


def test_wrong_checksum_raises_and_caches_nothing(remote: Path, cache: Path) -> None:
    wrong = model(remote, sha256="0" * 64)
    with pytest.raises(ChecksumError) as error:
        fetch(wrong, cache)
    message = str(error.value)
    assert "absorption" in message
    assert wrong.resolved_url in message
    assert f"expected: {'0' * 64}" in message
    assert f"actual:   {SHA256}" in message
    assert not wrong.cache_path(cache).exists()
    assert list(wrong.cache_path(cache).parent.iterdir()) == []


def test_changed_remote_raises(remote: Path, cache: Path) -> None:
    (remote / COMMIT / "absorption.xml").write_text("<sbml/>")
    with pytest.raises(ChecksumError):
        fetch(model(remote), cache)


def test_damaged_cache_is_downloaded_again(
    remote: Path, cache: Path, caplog: pytest.LogCaptureFixture
) -> None:
    path = fetch(model(remote), cache)
    path.write_text("damaged")
    with caplog.at_level(logging.WARNING, logger=pinned.__name__):
        assert fetch(model(remote), cache) == path
    assert "downloading it again" in caplog.text
    assert sha256sum(path) == SHA256


def test_damaged_cache_without_remote_raises(remote: Path, cache: Path) -> None:
    path = fetch(model(remote), cache)
    path.write_text("damaged")
    shutil.rmtree(remote)
    with pytest.raises(PinnedModelError, match="Download of model 'absorption'"):
        fetch(model(remote), cache)


def test_missing_remote_raises(remote: Path, cache: Path) -> None:
    shutil.rmtree(remote)
    with pytest.raises(PinnedModelError, match="Download of model 'absorption'") as error:
        fetch(model(remote), cache)
    assert not isinstance(error.value, ChecksumError)


def test_branch_warns_and_follows_the_remote(
    remote: Path, cache: Path, caplog: pytest.LogCaptureFixture
) -> None:
    main = model(remote, ref="main", sha256=None)
    assert not main.pinned
    with caplog.at_level(logging.WARNING, logger=pinned.__name__):
        path = fetch(main, cache)
    assert "not a commit" in caplog.text
    assert path == cache / "models" / "absorption" / "main" / "absorption.xml"
    changed = ABSORPTION.read_text().replace('value="0.02"', 'value="0.03"')
    assert changed != ABSORPTION.read_text()
    (remote / "main" / "absorption.xml").write_text(changed)
    assert fetch(main, cache).read_text() == changed


@pytest.mark.parametrize(
    ("changes", "match"),
    [
        ({"sha256": None}, "has no sha256"),
        ({"sha256": "abc"}, "64 lowercase hexadecimal"),
        ({"sha256": SHA256.upper()}, "64 lowercase hexadecimal"),
        ({"ref": "main"}, "not a full commit hash"),
        ({"ref": COMMIT[:7]}, "not a full commit hash"),
        ({"ref": ""}, "not a commit, branch or tag"),
        ({"ref": "../main"}, "not a commit, branch or tag"),
        ({"url": "https://example.org/model.xml"}, "placeholder"),
        ({"url": "http://example.org/{ref}/model.xml"}, "scheme"),
        ({"url": "https://example.org/{ref}/"}, "names no file"),
        ({"name": "../absorption"}, "Model name"),
        ({"name": ""}, "Model name"),
    ],
)
def test_invalid_definition_raises(changes: dict[str, str | None], match: str) -> None:
    fields = {
        "name": "absorption",
        "url": "https://example.org/{ref}/absorption.xml",
        "ref": COMMIT,
        "sha256": SHA256,
    }
    with pytest.raises(PinnedModelError, match=match):
        PinnedModel(**(fields | changes))


def test_read_pinned(tmp_path: Path) -> None:
    path = tmp_path / "models.toml"
    path.write_text(
        f"""
[first]
url = "https://example.org/{{ref}}/first.xml"
ref = "{COMMIT}"
sha256 = "{SHA256}"

[first.verification]
dose = 10.0

[second]
url = "https://example.org/{{ref}}/second.xml"
ref = "main"
"""
    )
    models = read_pinned(path)
    assert list(models) == ["first", "second"]
    assert models["first"] == PinnedModel("first", "https://example.org/{ref}/first.xml", COMMIT, SHA256)
    assert models["first"].resolved_url == f"https://example.org/{COMMIT}/first.xml"
    assert not models["second"].pinned


@pytest.mark.parametrize(
    ("text", "match"),
    [
        ('[model]\nurl = "https://example.org/{ref}/m.xml"\n', r"needs the text keys \['ref'\]"),
        ('[model]\nref = "main"\n', r"needs the text keys \['url'\]"),
        ('model = "https://example.org/{ref}/m.xml"\n', "not a table"),
        ('[model]\nurl = "https://example.org/{ref}/m.xml"\nref = "main"\nsha256 = 1\n', "must be text"),
        ("[model\n", "Could not read"),
    ],
)
def test_read_pinned_errors(tmp_path: Path, text: str, match: str) -> None:
    path = tmp_path / "models.toml"
    path.write_text(text)
    with pytest.raises(PinnedModelError, match=match):
        read_pinned(path)


def test_read_pinned_missing_file(tmp_path: Path) -> None:
    with pytest.raises(PinnedModelError, match="Could not read"):
        read_pinned(tmp_path / "missing.toml")


def test_models_of_the_repository_are_pinned() -> None:
    """The models file of the examples is valid without downloading anything."""
    from .conftest import MODELS_TOML

    models = read_pinned(MODELS_TOML)
    assert models
    assert all(m.pinned for m in models.values())
