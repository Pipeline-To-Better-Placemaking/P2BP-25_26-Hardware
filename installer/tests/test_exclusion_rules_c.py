"""Tests for the GLib-only exclusion rules under
`mv3dt_installer/assets/deepstream/exclusion/` (DETECTION-EXCLUSION-ZONES
section 9 item 1).

A small C driver is compiled against `mv3dt_exclusion_rules.c` with `gcc`
and `pkg-config glib-2.0`, then run to check config parsing (section 4.2)
and the inside-only rule (section 3.2). The whole module skips when `gcc`
or the GLib headers are absent.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

EXCLUSION_DIR = (
    Path(__file__).resolve().parents[1]
    / "mv3dt_installer"
    / "assets"
    / "deepstream"
    / "exclusion"
)
RULES_C = EXCLUSION_DIR / "mv3dt_exclusion_rules.c"


def _glib_flags() -> list[str] | None:
    if shutil.which("gcc") is None or shutil.which("pkg-config") is None:
        return None
    result = subprocess.run(
        ["pkg-config", "--cflags", "--libs", "glib-2.0"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return None
    return result.stdout.split()


GLIB_FLAGS = _glib_flags()
if GLIB_FLAGS is None:
    pytest.skip("gcc or pkg-config glib-2.0 unavailable", allow_module_level=True)


# Modes:
#   parse          read a config from stdin, print "total <n>" and one
#                  "source <s> <l> <t> <r> <b>" line per rectangle, or
#                  "error" when the text is not a key file
#   file <path>    as parse, from a file
#   box <l> <t> <r> <b> <bl> <bt> <bw> <bh>
#                  print "inside=<0|1> overlaps=<0|1>" for the box against
#                  the rectangle with edges l, t, r, b
DRIVER_C = r"""
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <glib.h>
#include "mv3dt_exclusion_rules.h"

static int
dump (Mv3dtExclusionRules * rules)
{
  if (!rules) {
    printf ("error\n");
    return 0;
  }
  printf ("total %u\n", rules->total);
  for (guint s = 0; s < MV3DT_EXCLUSION_MAX_SOURCES; s++) {
    const GArray *rects = mv3dt_exclusion_rules_for_source (rules, s);
    if (!rects)
      continue;
    for (guint i = 0; i < rects->len; i++) {
      const Mv3dtExclusionRect *r =
          &g_array_index (rects, Mv3dtExclusionRect, i);
      printf ("source %u %g %g %g %g\n", s, r->left, r->top, r->right,
          r->bottom);
    }
  }
  mv3dt_exclusion_rules_free (rules);
  return 0;
}

int
main (int argc, char **argv)
{
  if (argc >= 2 && strcmp (argv[1], "parse") == 0) {
    GString *text = g_string_new (NULL);
    char buf[4096];
    size_t n;
    while ((n = fread (buf, 1, sizeof buf, stdin)) > 0)
      g_string_append_len (text, buf, n);
    int rc = dump (mv3dt_exclusion_rules_load_data (text->str, text->len,
            NULL));
    g_string_free (text, TRUE);
    return rc;
  }
  if (argc == 3 && strcmp (argv[1], "file") == 0)
    return dump (mv3dt_exclusion_rules_load_file (argv[2], NULL));
  if (argc == 10 && strcmp (argv[1], "box") == 0) {
    Mv3dtExclusionRect r = { atof (argv[2]), atof (argv[3]),
      atof (argv[4]), atof (argv[5]) };
    gfloat l = atof (argv[6]), t = atof (argv[7]);
    gfloat w = atof (argv[8]), h = atof (argv[9]);
    GArray *rects = g_array_new (FALSE, FALSE, sizeof r);
    g_array_append_val (rects, r);
    printf ("inside=%d overlaps=%d\n",
        mv3dt_exclusion_rects_inside_any (rects, l, t, w, h) ? 1 : 0,
        mv3dt_exclusion_rects_overlap_any (rects, l, t, w, h) ? 1 : 0);
    g_array_free (rects, TRUE);
    return 0;
  }
  fprintf (stderr, "usage\n");
  return 2;
}
"""


@pytest.fixture(scope="module")
def driver(tmp_path_factory: pytest.TempPathFactory) -> Path:
    build = tmp_path_factory.mktemp("exclusion_driver")
    source = build / "driver.c"
    source.write_text(DRIVER_C, encoding="utf-8")
    binary = build / "driver"
    result = subprocess.run(
        [
            "gcc",
            "-std=c99",
            "-Wall",
            "-Werror",
            f"-I{EXCLUSION_DIR}",
            str(source),
            str(RULES_C),
            "-o",
            str(binary),
            *GLIB_FLAGS,
            "-lm",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    return binary


def _parse(driver: Path, config: str) -> tuple[list[str], str]:
    result = subprocess.run(
        [str(driver), "parse"], input=config, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    return result.stdout.splitlines(), result.stderr


def _box(driver: Path, rect: tuple, box: tuple) -> str:
    result = subprocess.run(
        [str(driver), "box", *(str(v) for v in (*rect, *box))],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout.strip()


# ---------------------------------------------------------------------------
# Config parsing (section 4.2)
# ---------------------------------------------------------------------------


def test_parses_multiple_rects_and_sources(driver: Path) -> None:
    lines, stderr = _parse(
        driver,
        "# header comment\n"
        "[source1]\n"
        "rects=530,175,245,225\n"
        "[source3]\n"
        "rects=0,0,10,20;100,200,50,60\n",
    )
    assert lines == [
        "total 3",
        "source 1 530 175 775 400",
        "source 3 0 0 10 20",
        "source 3 100 200 150 260",
    ]
    assert stderr == ""


@pytest.mark.parametrize(
    "bad",
    [
        "1,2,3",
        "1,2,3,4,5",
        "1,2,x,4",
        "1,2,,4",
        "1,2,0,4",
        "1,2,4,0",
        "1,2,-3,4",
        "1,2,3,-4",
        "1,2,nan,4",
        "1,2,1.5,4",
        "1.0,2,3,4",
        "-3,2,3,4",
        "+3,2,3,4",
        "1e2,2,3,4",
        "1 0,2,3,4",
        "0x10,2,3,4",
    ],
)
def test_bad_entry_rejects_its_group_only(driver: Path, bad: str) -> None:
    lines, stderr = _parse(
        driver,
        f"[source0]\nrects=10,10,20,20;{bad}\n[source2]\nrects=1,2,3,4\n",
    )
    assert lines == ["total 1", "source 2 1 2 4 6"]
    assert "bad rects in [source0]" in stderr


def test_whitespace_and_trailing_separator_are_accepted(driver: Path) -> None:
    lines, stderr = _parse(
        driver, "[source0]\nrects= 1 , 2 ,\t3, 4 ;  ; 10,20,30,40 ;\n"
    )
    assert lines == ["total 2", "source 0 1 2 4 6", "source 0 10 20 40 60"]
    assert stderr == ""


def test_frame_bounds_are_not_checked(driver: Path) -> None:
    lines, _ = _parse(driver, "[source0]\nrects=1900,1000,500,500\n")
    assert lines == ["total 1", "source 0 1900 1000 2400 1500"]


@pytest.mark.parametrize(
    "group", ["camera1", "source64", "source", "source-1", "source1x", "Source1"]
)
def test_other_group_names_are_ignored(driver: Path, group: str) -> None:
    lines, stderr = _parse(
        driver, f"[{group}]\nrects=1,2,3,4\n[source63]\nrects=5,6,7,8\n"
    )
    assert lines == ["total 1", "source 63 5 6 12 14"]
    assert f"ignoring group [{group}]" in stderr


def test_no_rects_yields_empty_rules(driver: Path) -> None:
    lines, _ = _parse(driver, "# Rendered with no boxes.\n")
    assert lines == ["total 0"]


def test_unparseable_text_is_an_error(driver: Path) -> None:
    lines, _ = _parse(driver, "this is not a key file\n")
    assert lines == ["error"]


def test_load_file(driver: Path, tmp_path: Path) -> None:
    config = tmp_path / "exclusion.conf"
    config.write_text("[source1]\nrects=530,175,245,225\n", encoding="utf-8")
    result = subprocess.run(
        [str(driver), "file", str(config)], capture_output=True, text=True
    )
    assert result.stdout.splitlines() == ["total 1", "source 1 530 175 775 400"]

    missing = subprocess.run(
        [str(driver), "file", str(tmp_path / "missing.conf")],
        capture_output=True,
        text=True,
    )
    assert missing.stdout.splitlines() == ["error"]


# ---------------------------------------------------------------------------
# Inside-only rule (section 3.2)
# ---------------------------------------------------------------------------

TV = (530, 175, 775, 400)


def test_box_touching_all_four_edges_is_inside(driver: Path) -> None:
    assert _box(driver, TV, (530, 175, 245, 225)) == "inside=1 overlaps=1"


def test_box_inside_with_margin_is_inside(driver: Path) -> None:
    assert _box(driver, TV, (600, 200, 90, 130)) == "inside=1 overlaps=1"


@pytest.mark.parametrize(
    "box",
    [
        (529, 200, 90, 130),  # past the left edge
        (600, 174, 90, 130),  # past the top edge
        (686, 200, 90, 130),  # past the right edge
        (600, 271, 90, 130),  # past the bottom edge
        (600, 250, 90, 400),  # a person in front of the TV
    ],
)
def test_box_overlapping_an_edge_is_not_inside(driver: Path, box: tuple) -> None:
    assert _box(driver, TV, box) == "inside=0 overlaps=1"


def test_box_fully_outside_is_neither(driver: Path) -> None:
    assert _box(driver, TV, (1000, 500, 100, 200)) == "inside=0 overlaps=0"
