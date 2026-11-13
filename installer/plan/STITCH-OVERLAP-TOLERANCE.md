# Stitch overlap tolerance — stop one person splitting into two (owner: DevD)

Status: amendment spec, **specified, prototype-validated on recorded data,
not implemented**. It changes stage 5c (sequential stitch) of
[`TRACKING-FUSION` section 5](TRACKING-FUSION-AND-CLEANING.md#5-fusion-stages),
adds two keys to the table in
[`TRACKING-FUSION` section 4.2](TRACKING-FUSION-AND-CLEANING.md#42-keys-and-strict-defaults)
and one counter to
[`TRACKING-FUSION` section 5.1](TRACKING-FUSION-AND-CLEANING.md#51-drop-and-merge-counters-required).
The rest of the fusion contract is **not** restated here.

A seated person whose tracker ID churns shows up in the fused output as two
people. Their tracklets alternate at the same spot, and the second chain of
fragments is never stitched to the first, because the `stitch_exclusive`
rule refuses any join between two people who overlap in time by more than
one bucket. **One 0.2 s overlap blocks the whole join.** This doc lets a
stitch through a short overlap when the two people are at the same place
for all of it.

---

## 1. Scope

In scope: the `stitch_exclusive` conflict test of stage 5c, two cleaning
keys and one counter. Out of scope:
[section 6](#6-out-of-scope--open-decisions).

---

## 2. Measured evidence

Recorded data from 2026-11-12, AMC run 2 calibration, live `cleaning.json`
with `smooth_window` 15, replayed through the installed fusion code.

| Fact | Value |
|---|---|
| Seated person, 17:40 to 18:10 | fused into person 1 (1602 s) and person 2 (216 s) |
| Person 2's pieces | 16 pieces of 0.2 to 15.8 s, each in a gap of person 1, 0.02 to 0.34 m from person 1 |
| Time persons 1 and 2 exist together | one 0.2 s bucket, 0.18 m apart |
| Why stage 5b does not join them | the overlap is under `dup_min_overlap_s` (1.0 s) |
| Why stage 5c does not join them | `stitch_exclusive` refuses any overlap longer than one bucket |
| `stitch_exclusive: false` instead | 17:40 to 18:10 collapses to 1 person, merging a second real person |

Prototype of [section 3](#3-the-rule), replayed over three windows:

| Window | Live code | Tolerance 1.0 s, 0.5 m | 1.0 s, 0.3 m | 2.0 s, 0.5 m |
|---|---|---|---|---|
| 17:00 to 17:40 | 2 people | 2 people | 2 people | 2 people |
| 17:40 to 18:10 | 3 people | **2 people** | 2 people | 2 people |
| 18:10 to 18:50 | 6 people | **5 people** | 9 people | 5 people |

---

## 3. The rule

**REQUIRED.** In stage 5c, when `stitch_exclusive` is true and the
candidate's person and the target person overlap in time by more than one
bucket (today's refusal), the stitch is **still allowed** when both hold:

1. **Short**: the buckets in which both people have a position add up to at
   most `stitch_overlap_tol_s`.
2. **Co-located**: in every one of those buckets, the two people's positions
   are within `stitch_overlap_dist_m` of each other.

A person's position in a bucket, for this test, is the mean of its member
tracklets' points in that bucket, before camera fusion (stage 6) and
smoothing (stage 7). Every other stage 5c condition is unchanged: the gap,
the distance limit and the choice of the nearest candidate.

**LOCKED: the tolerance never applies with `stitch_exclusive` false**; that
setting already allows every overlap.

### 3.1 Keys

| Key | Strict default | Unit | Stage | Meaning |
|---|---|---|---|---|
| `stitch_overlap_tol_s` | `1.0` | s | 5c | longest overlap a stitch may cross; `0` restores today's rule |
| `stitch_overlap_dist_m` | `0.5` | m | 5c | the two people must stay this close in every overlapping bucket |

Both are numbers of at least 0; any other value makes the file invalid as
[`TRACKING-FUSION` section 4.1](TRACKING-FUSION-AND-CLEANING.md#41-file-and-reload-required)
states. The relaxed preset inherits both.

### 3.2 Counter

`overlap_stitch_joins` (stage 5c, joins): stitches that relied on
[section 3](#3-the-rule). They are also counted in `stitch_joins`, so the
existing counter keeps its meaning.

### 3.3 Cost (REQUIRED)

The test needs each person's positions in the overlapping buckets. Keep
per-person position maps updated on every union rather than rebuilding them
from all tracklets on each candidate; a 30-minute window must not become
noticeably slower than today.

---

## 4. Verification

1. **Unit tests** in `tests/tracking/test_fuse.py`:
   - two alternating co-located chains with a 0.2 s overlap fuse into one
     person, and `overlap_stitch_joins` counts the join;
   - the same with the people 1.0 m apart during the overlap stays two
     people;
   - an overlap longer than `stitch_overlap_tol_s` stays two people;
   - `stitch_overlap_tol_s: 0` reproduces today's output exactly;
   - existing stitch, duplicate and shared-id tests pass unchanged.
2. **Cleaning tests** in `tests/tracking/test_cleaning.py`: defaults, preset
   inheritance and invalid values for both keys.
3. **Workstation, after release**: the
   [section 2](#2-measured-evidence) windows give 2, 2 and 5 people.

---

## 5. Amendments to TRACKING-FUSION

The spec unit applies these to
[`TRACKING-FUSION-AND-CLEANING.md`](TRACKING-FUSION-AND-CLEANING.md):

- section 4.2: the two [section 3.1](#31-keys) rows after
  `stitch_exclusive`;
- section 5, stage 5c (item 3): one sentence stating the exception, linking
  here;
- section 5.1: the `overlap_stitch_joins` row.

---

## 6. Out of scope / open decisions

- Short people. A 5 s person in the 18:10 to 18:50 window survives
  `min_person_s` (3 s); tuning that is separate.
- Any change to stages 5a and 5b.

Open decisions:

1. **Flagged:** the defaults come from one evening of one room. Revisit them
   when a space with more people is recorded.

---

## 7. Decomposition

| Unit | Branch | Files touched | Depends on | Wave |
|---|---|---|---|---|
| T1 Spec | `feat/installer-stitch-overlap-spec` | `installer/plan/STITCH-OVERLAP-TOLERANCE.md`, `installer/plan/TRACKING-FUSION-AND-CLEANING.md` (sections 4.2, 5, 5.1) | none | 1 |
| T2 Stitch tolerance | `feat/installer-stitch-overlap` | `assets/tracking/mv3dt_tracking/fuse.py`, `assets/tracking/mv3dt_tracking/cleaning.py`, `tests/tracking/test_fuse.py`, `tests/tracking/test_cleaning.py` | none | 1 |

Package paths are relative to `installer/mv3dt_installer/`; test paths are
relative to `installer/`. The two units touch disjoint files. Recommended
merge order: T1, T2, then a release.

---

## References

Evidence comes from the workstation's tracking database on 2026-11-12,
replayed through the installed `mv3dt_tracking` package and a prototype of
[section 3](#3-the-rule).

Repo files referenced:

- [`TRACKING-FUSION-AND-CLEANING.md`](TRACKING-FUSION-AND-CLEANING.md) —
  sections 4.1, 4.2, 5 and 5.1, the contract this doc amends.
- [`REPLAY-PRESENTATION.md`](REPLAY-PRESENTATION.md) — the Replay that
  surfaced the split person.
