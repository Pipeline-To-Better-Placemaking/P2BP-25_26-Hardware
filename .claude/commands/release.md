---
description: Cut the next patch release after /merge-prs — bump the version from the most recent v* tag, tag it, wait for the release build, and print the workstation update commands.
---

Cut the next release of `mv3dt-installer`. This command is the step after
`/merge-prs` (`.claude/commands/merge-prs.md`): once the PRs for a release
are on `main`, it turns `main` into a published GitHub Release. Publishing
is done by `.github/workflows/release.yml`, which builds the binary and
creates the release when a `v*` tag is pushed. This command prepares and
pushes that tag, then verifies the result. It takes no arguments. Follow
these steps in order and stop at the first failure.

1. **Preconditions.** From the repo root, run `git fetch origin --tags`,
   then confirm all of the following. If any fails, stop and report it; do
   not fix it silently.
   - The current branch is `main`, the working tree is clean
     (`git status --porcelain` prints nothing), and local `main` equals
     `origin/main`.
   - There are no open PRs (`gh pr list --state open`). If there are, list
     them and ask whether to release without them: the release commit
     records "no open PRs at the release cut" in `DEVELOPMENT-STATUS.md`.

2. **Find the most recent tag and check it has a release.**
   `LATEST=$(git tag -l 'v*' --sort=-v:refname | head -1)`.
   Run `gh release view "$LATEST" --json tagName,isDraft,assets`. It must
   exist, not be a draft, and list both `mv3dt-installer` and
   `mv3dt-installer.sha256`. **If it does not, stop and do not run:** the
   previous release never finished. Report `LATEST` and the latest run from
   `gh run list --workflow release.yml --limit 3` so the user can see why.
   Also stop if:
   - `__version__` in `installer/mv3dt_installer/__init__.py` does not
     equal `LATEST` without its `v` (the version was bumped without a tag,
     or the other way round);
   - `git log "$LATEST"..origin/main --oneline` is empty (nothing to
     release).

3. **Compute the next version.** Increment the patch number of `LATEST`:
   `vX.Y.Z` becomes `vX.Y.(Z+1)`. Call it `NEXT`. Minor and major bumps are
   out of scope for this command; they are done by hand.

4. **Bump the version.** Change exactly these lines, found by content, not
   by line number:
   - `installer/mv3dt_installer/__init__.py`: `__version__ = "<old>"`.
   - `installer/plan/DEVELOPMENT-STATUS.md`, five lines:
     - `Current release: **v<old>**.`
     - `` `__version__ = "<old>"` `` on the line under it;
     - the checklist item "`__version__` equals the most recent `v*` tag
       (`v<old>`)";
     - the checklist item "`gh pr list --state open` is empty at the
       v<old> release cut";
     - the `References` sentence "drawn from the repository through release
       `v<old>`".

   Leave every other mention of the old version alone. The workstation
   handoff prose and the history tables are updated in a separate `docs:`
   change (step 9), not here. Show the diff: it must be one line in
   `__init__.py` and five lines in `DEVELOPMENT-STATUS.md`. Confirm with
   `cd installer && python3 -c "import mv3dt_installer as m; print(m.__version__)"`.

5. **Commit.** Use the `git-commits` skill: message exactly
   `installer: release version <NEXT without v>`, with no trailer or
   footer. Commit only those two files.

6. **Push `main`, then the tag.** In this order:
   ```bash
   git push origin main
   git tag <NEXT>
   git push origin <NEXT>
   ```
   `main` goes first so the tagged commit is on `main` when the build runs.
   The tag is lightweight, like the recent `v*` tags. Push only that one
   tag, never `--tags`: GitHub starts no workflows when a single push
   carries more than three tags.

   If Claude Code's permission check denies either push, do not retry it
   or work around it. Stop, print the remaining commands from this step for
   the user to run with the `!` prefix so their output lands in the
   session, and continue at step 7 once they have run.

7. **Wait for the release build.** Find the run the tag push started:
   `gh run list --workflow release.yml --branch <NEXT> --limit 1 --json databaseId,status`
   (a tag push's run lists the tag as its branch; poll briefly until it
   appears). Then `gh run watch <id> --exit-status`. If it fails, show
   `gh run view <id> --log-failed | tail -60` and stop. Do not delete or
   move the tag, and do not re-push: report the failure and let the user
   decide.

8. **Verify the release.**
   - `gh release view <NEXT> --json tagName,isDraft,assets` shows
     `isDraft: false` and both assets.
   - `curl -fsSL https://github.com/<owner>/<repo>/releases/download/<NEXT>/mv3dt-installer.sha256`
     prints a checksum line (`<owner>/<repo>` from
     `gh repo view --json nameWithOwner`).

9. **Report.** Give the release URL, the merged PRs it contains
   (`git log <LATEST>..<NEXT> --oneline --first-parent`), and the
   workstation update block, with `<NEXT>` and `<owner>/<repo>` filled in:
   ```bash
   cd "$HOME/Downloads"
   rm -f -- mv3dt-installer mv3dt-installer.sha256
   curl -fLO https://github.com/<owner>/<repo>/releases/download/<NEXT>/mv3dt-installer
   curl -fLO https://github.com/<owner>/<repo>/releases/download/<NEXT>/mv3dt-installer.sha256
   sha256sum -c mv3dt-installer.sha256
   chmod +x mv3dt-installer
   ./mv3dt-installer --version
   sudo ./mv3dt-installer --verbose
   ```
   The `rm -f` removes only the two previous files in `~/Downloads`; it
   does not reset installation progress. Add a reset step (for example
   clearing `AMC_PROJECT_ID`) only when the release's changes need one, and
   say why.

   Then offer, without doing it unasked, to update the workstation handoff
   section of `installer/plan/DEVELOPMENT-STATUS.md` for `<NEXT>` as a
   separate `docs:` change.
