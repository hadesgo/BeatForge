"""The CLI command a new user runs first."""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from beatforge.cli import app


def test_init_creates_a_loadable_project(tmp_path: Path) -> None:
    """``init`` has to leave behind a project that ``run`` can actually open.

    The template it writes is validated by ``load_project`` here, so a template that no
    longer parses fails the test rather than the user's first ``run``.
    """
    runner = CliRunner()
    target = tmp_path / "my-mv"

    result = runner.invoke(app, ["init", str(target)])

    assert result.exit_code == 0, result.output
    assert (target / "project.toml").is_file()
    assert (target / "media").is_dir()
    assert (target / "fonts").is_dir()
    assert (target / "lyrics.lrc").is_file()

    # The generated project has to load back: an unparsable template is the worst
    # possible first impression.
    from beatforge.config import load_project

    project = load_project(target / "project.toml")
    assert project.music.name == "music.mp3"
    assert project.media_dir.is_dir()
