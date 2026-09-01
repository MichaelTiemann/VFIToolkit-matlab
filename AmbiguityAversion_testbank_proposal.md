# Proposal: Ambiguity Aversion test bank (CoreFHorzTests/withAmbiguityAversion)

2026-08-28, v3 (approved, with second-round red-pen updates: lowmemory=2 where relevant; a
pi_z-consistency warning; both open questions answered yes).
**STATUS: COMPLETE — GPU-GREEN 2026-09-01, first run. 156/156 zero-checks (138 exact zeros, 18
at the 1e-15..1e-14 ULP floor: the 10 ValueFnFromPolicy lines are matmul-vs-broadcast rounding;
the 8 cross-test-3 V lines are min() picking either prior's rounding where retirement makes V
flat in z — the Policy lines there are exact zeros). Error assert + all 8 warning checks passed;
6 figures produced. Ready to commit (bank + toolkit together).**

Implementation notes (2026-09-01), beyond Section 4 as written:
- The 6 plain raws were REWRITTEN from the modern standard raws (not renamed): the old ones were
  old-style (2-row Policy2 output, per-z EV recompute under lowmemory) and, in the z&e variants,
  had a real bug — the z-stage prior loop overwrote its own EV base after the first prior, so
  priors 2+ multiplied garbage. All 24 raws are now mechanical transforms of the modern standard
  family (donor + per-prior loop + min at the EV sites, nothing else), so drift-sweeps against
  the standard raws stay meaningful.
- GI design decision (corrected in review 2026-09-01): the interpolation is done CONDITIONAL ON
  THE PRIOR, in the same way GI is conditional on d, z and e — each prior's EV is interpolated
  over aprime, and the min over priors is taken afterwards. Same in ValueFnFromPolicy (each
  prior's EV read at the two L2 points with the L2 weights, then min). On grid points the two
  orders coincide, so GI level 1, plain, and all of DC (where EV is only ever read at grid
  points, pointwise) just use the pointwise worst-case EV — DC and DC+GI are thereby also
  'conditional on the prior' wherever that distinction has any content.
- Composition convention for z&e (from the old raws, kept): the iid-e worst case is taken first,
  then the markov-z worst case; the same n_ambiguity governs both.
- The setup subfn makes ONE recursive ExogShockSetup_FHorz call per prior (gridpiboth=2, which
  handles that prior's pi_z AND pi_e together), plus one normal gridpiboth=3 call for the grids
  and the regular pi's; recursion is guarded by blanking exoticpreferences in the temp options.
  ExogShockFn/EiidShockFn error under ambiguity; e-only ambiguity models now work (the old
  unconditional ambiguity_pi_z demand is gone). Test-first, like the ResidualAsset bank: build the whole
bank first, assuming every toolkit feature it exercises already exists; then implement the toolkit
side until the bank runs green. The bank stays uncommitted until the toolkit side passes (house
convention: a test-first bank commits together with its implementation).

Location: `/home/kmarshallbanana/Dropbox/VFI_Toolkit_Docs/vfitoolkitTests/CoreFHorzTests/withAmbiguityAversion/`
— sibling of `withEpsteinZinPreferences/` and `withQuasiHyperbolicDiscounting/`, sharing
`CoreFHorzTests_Setup/` and `CoreFHorz_ReturnFns/`, writing diary and pngs into `../TestOutput/`.

**Settled in review:**
- Ambiguity is over the transition probabilities **only**: every prior uses the same shock grid
  (the model z_grid/e_grid). No per-prior grids, no interpolation of V over shocks. The
  different-grids cross test is gone; a duplicated-priors (3 pi vs 9 pi) test replaces it.
- Naming: the main dispatcher is `ValueFnIter_FHorz_AmbiguityAversion.m`; all subfunctions (raws,
  level-2 dispatchers) use the shorter `AmbAverse`.
- Test 3 sign confirmed: the min binds on the pi with the *higher* probability on the worst grid
  point; that is the pi the standard-preferences comparison uses.

---

## 1. Where the toolkit stands today

What exists (all in `ValueFnIter/FHorz/ExoticPrefs/`):

- `ValueFnIter_FHorz_Ambiguity.m` dispatcher + 6 plain raws in `AmbiguityAversion/`:
  {nod, d} x {z, noz_e, z_e}. (noz+noe deliberately errors: "what is the point".)
- Interface: `vfoptions.exoticpreferences='AmbiguityAversion'`; `vfoptions.n_ambiguity` (scalar or
  age-vector); `vfoptions.ambiguity_pi_z` [N_z,N_z,max(n_ambiguity)] or `ambiguity_pi_z_J`;
  `vfoptions.ambiguity_pi_e_J` (indexed at jj+1 in the raws — Reading A, matching the pi_e timing
  convention). EV is computed per prior, then `min` over priors; the regular pi_z/pi_e arguments
  are unused by the solve.
- lowmemory=0/1 in every raw, and lowmemory=2 already in the two z&e raws (the
  docs/ExogenousShocks.md lowmemory table has the rules: z or e alone allows 0/1; z&e allows
  0/1/2).

What does not exist (the bank assumes all of it):

1. **The rename.** Dispatcher becomes `ValueFnIter_FHorz_AmbiguityAversion.m`; the 6 existing raws
   become `ValueFnIter_FHorz_AmbAverse_*_raw.m`; call site in `ValueFnIter_Case1_FHorz.m`
   updated. The vfoptions string stays `'AmbiguityAversion'`.
2. **Divide-and-conquer, grid-interp-layer, and DC+GI raws.** Today the dispatcher never reads
   `vfoptions.divideandconquer`/`gridinterplayer` — it *silently ignores* them and solves plain.
3. **`ValueFnFromPolicy_FHorz_AmbiguityAversion`.** `ValueFnFromPolicy_FHorz.m` dispatches for QH
   and EZ only; ambiguity Policy currently gets valued under the wrong (single-prior) expectations.
4. **ExogShockSetup integration.** Today the ambiguity pi's bypass `ExogShockSetup_FHorz`
   entirely: they get only ad-hoc validation inside the ambiguity dispatcher, so none of the
   accepted input shapes, age-broadcasting, stacked/joint handling, or timing/trim conventions
   apply to them (flat `ambiguity_pi_e` does not work at all — only pre-built `_J`; and the
   dispatcher demands ambiguity_pi_z unconditionally, so an e-only ambiguity model cannot even be
   declared cleanly). Design in Section 5.
5. **pi_z-consistency warning.** On the ValueFnIter path, check that the regular pi_z (which the
   agent distribution etc. will use) is equal to one of the ambiguity_pi_z priors, and throw a
   warning() if not; likewise pi_e against the ambiguity_pi_e priors. Lives in the setup
   subfunction of Section 5 (which only runs from the ValueFnIter call, and has both sides in _J
   form to compare).

## 2. Bank layout (mirrors the QH bank)

```
withAmbiguityAversion/
  CoreFHorzAmbiguityTests.m            % main script: diary, setup, priors, calls subcodes, saves figs
  withAmbiguityAversion_subcodes/
    AmbFHorz_nod_z_noe_nosemiz.m       % fig 1
    AmbFHorz_d_z_noe_nosemiz.m         % fig 2
    AmbFHorz_nod_noz_e_nosemiz.m       % fig 3
    AmbFHorz_d_noz_e_nosemiz.m         % fig 4
    AmbFHorz_nod_z_e_nosemiz.m         % fig 5
    AmbFHorz_d_z_e_nosemiz.m           % fig 6
    CrossTests/
      AmbFHorz_CrossTests_nod_nosemiz.m
      AmbFHorz_CrossTests_d_nosemiz.m
```

9 test files. Only 6 main subcodes (not the QH bank's 8) because noz+noe is an error for ambiguity;
the main script asserts that the error fires (one try/catch check).

Diary: `../TestOutput/CoreFHorzAmbiguityTestsdiary.txt`. Figures:
`CoreFHorzAmbiguityTests_FigN.png`, one per subcode (no Naive/Sophisticated split, so no 100+
figures). All zero-checks print `%.3e` (ULP-floor convention).

**Scope:** nosemiz, single standard endogenous asset only. Semiz and with2A waves can follow later,
mirroring how the QH bank was staged; they are out of scope here and their raw families are not
assumed to exist.

### Baseline priors (defined once in the main script)

`n_ambiguity=3`, shared by every main subcode:

- prior 1: baseline `pi_z` (from `CoreFHorz_setup`);
- prior 2: `0.9*pi_z + 0.1*[1,0,...,0]` — contamination toward the worst z';
- prior 3: `0.9*pi_z + 0.1*(1/N_z)` — contamination toward uniform.

Same recipe for `pi_e`. Priors 2 and 3 are not ranked against each other, so the binding prior can
switch across (a,z,j) and the min-over-priors machinery is genuinely exercised (not a relabelled
single-prior solve). The z_e subcodes make both z and e ambiguous. All priors share the model
grids throughout the bank.

### What each main subcode does

The QH template minus Valt (ambiguity returns just V, Policy):

1. Solve baseline; solve with `divideandconquer=1`; check V, Policy zero. (2 checks)
2. `lowmemory=1` on both; check vs their lowmemory=0 twins. (4)
3. `ValueFnFromPolicy_FHorz` vs V. (1)
4. Solve with `gridinterplayer=1, ngridinterp=5`; solve DC+GI; check V, Policy zero. (2)
5. `lowmemory=1` on GI and DC+GI. (4)
6. `ValueFnFromPolicy_FHorz` on the GI solve. (1)
6b. In the two z_e subcodes only: repeat the lowmemory checks at lowmemory=2 (baseline, DC, GI,
   DC+GI), per the docs/ExogenousShocks.md rule that z&e allows levels 0/1/2. (+8 checks each)
7. Big `a_grid` (n_a_big=1001): moments with vs without GI via
   StationaryDist/AllStats/LifeCycleProfiles, eyeball tables + one near-zero dist print, and the
   age-conditional figure. The distribution needs a "true" process — under ambiguity there isn't
   one, so the bank uses baseline `pi_z`/`pi_e` (the regular arguments). Everything downstream of
   Policy is standard code; no toolkit changes needed there.

~14 zero-checks x 6 subcodes, +8 lowmemory=2 checks in each z_e subcode = ~100 checks, plus the
eyeball moment tables and 6 figures.

## 3. The five cross tests

Each cross-test file runs all five tests twice over: once with the ambiguity on a **markov z**, once
on an **iid e** (separate code paths: `ambiguity_pi_z_J` vs `ambiguity_pi_e_J`, markov vs iid
expectation). Every comparison is against `exoticpreferences='None'` — exact zeros, no tolerances.

Prerequisite the setup already satisfies: V is increasing in z (and e), since shocks enter as
earnings `w*kappa_j*z` with increasing exp-grids. Tests 3a/3b rely on this monotonicity for the
worst-case prior to bind everywhere; tests 1, 2, 4 do not.

**Test 1 — three identical priors.** `n_ambiguity=3`, all three priors = baseline pi. Must equal
standard preferences with that pi. All three EVs are identical, min is trivial. Also catches
normalisation bugs (e.g. priors accidentally summed instead of min'd). Run twice — once passing
flat `ambiguity_pi_z` (/`ambiguity_pi_e`), once passing the pre-built `_J` form — with a zero
check between the two: a cheap plumbing gate on the setup subfunction of Section 5.

**Test 2 — duplicated priors, 3 pi vs 9 pi.** Take the three distinct baseline priors (pi_1, pi_2,
pi_3 above). Solve once with `n_ambiguity=3` and those three; solve again with `n_ambiguity=9`
where the nine slots hold five copies of pi_1 and two copies each of pi_2 and pi_3. The min over a
multiset ignores multiplicity, so the two solves must agree exactly. Catches any implementation
that weights priors by count (e.g. an accidental mean or sum over the prior dimension) and
exercises n_ambiguity truly varying in size.

**Tests 3a/3b — pi ordering.** Two priors, same grid, pi's identical except one shifts probability
delta (say 0.05) in every row from the **best** grid point into the **worst** grid point. That pi
is the unambiguously worse prior, and the answer must equal standard preferences under it:
EV_worse - EV_orig = delta*(V(z_worst)-V(z_best)) <= 0 pointwise, so the min always selects it,
exactly.
- 3a: worse pi in slot 1; 3b: worse pi in slot 2. Same answer both ways — demonstrating prior
  ordering is irrelevant. The twin looks trivial (min commutes) but is exactly the test that
  catches an implementation that special-cases prior 1.

**Test 4 — age-varying n_ambiguity via V_Jplus1.** With N_j=20 and half=10:
- Solve (a): full-horizon AA with `n_ambiguity=[3*ones(1,10), ones(1,10)]` and `ambiguity_pi_z_J`
  holding the three baseline priors in the first half and prior 1 (=baseline pi) in the second.
- Solve (b): full-horizon **standard preferences** (vNM) with baseline pi — this is what the
  second half of (a) is, since a single prior is just expected utility.
- Solve (c): AA over ages 1..10 only (`N_j=10`, age-dependent params trimmed per the jstar
  convention from the CoreFHorz V_Jplus1 tests) with `vfoptions.V_Jplus1 = V_b(:,:,11)`.
Checks: second half of (a) equals second half of (b); first half of (a) equals (c). In words: "AA
with 1 prior in the second half" equals "AA with V_Jplus1 taken from the vNM solution of the
second half". This is the only test that exercises `n_ambiguity(jj)` genuinely varying, and it is
the bank's gate on the V_Jplus1 branches of the AA raws (the copy-paste-drift hotspot).

Additionally each flavour checks the pi-consistency warning of Section 1 item 5: solve once with
the regular pi (equal to prior 1) — lastwarn must stay clean — and once with a regular pi that is
not among the priors — lastwarn must show the warning. (2 checks per flavour.)

Counts: T1 (4, incl. the flat-vs-_J twin) + T2 (2) + T3a (2) + T3b (2) + T4 (4) + warning (2)
= 16 checks per flavour, x 2 flavours (z, e) x 2 files (nod, d) = 64 zero-checks.

## 4. Toolkit implementation wave (what the bank forces into existence)

In dependency order:

1. **Rename:** `ValueFnIter_FHorz_Ambiguity.m` -> `ValueFnIter_FHorz_AmbiguityAversion.m`; the 6
   existing raws -> `ValueFnIter_FHorz_AmbAverse_{...}_raw.m` (folder name `AmbiguityAversion/`
   stays); update the call in `ValueFnIter_Case1_FHorz.m`. All new subfunctions below use
   `AmbAverse`.
2. **18 new raws**, mirroring the QH folder layout under
   `ValueFnIter/FHorz/ExoticPrefs/AmbiguityAversion/`:
   `DivideConquer/` (`AmbAverse_DC1_*` x6), `GridInterpLayer/` (`AmbAverse_GI1_*` x6),
   `DivideConquerGridInterpLayer/` (`AmbAverse_DC1_GI1_*` x6), each set covering
   {nod,d} x {z, noz_e, z_e}, each with a V_Jplus1 block and the lowmemory levels the
   docs/ExogenousShocks.md table allows (0/1, plus 2 in the z_e variants). The nearest donor
   family is QH-Sophisticated's DC1/GI1/DC1_GI1 raws (same "transform EV, then proceed as
   standard" shape — here the transform is the per-prior loop + min). V_Jplus1 blocks are
   copy-paste drift magnets: run the two mechanical sweeps (undefined `N_*`; V/Policy writes
   missing the loop variable) over all 18 before first run — and cross test 4 gates them at
   runtime.
3. **Dispatcher routing** in `ValueFnIter_FHorz_AmbiguityAversion.m` on
   divideandconquer/gridinterplayer (today: silently ignored), mirroring
   `ValueFnIter_FHorz_QuasiHyperbolic_DC/_DC_GI`. Level-2 sub-dispatchers
   `ValueFnIter_FHorz_AmbAverse_DC.m` / `_DC_GI.m` per the QH pattern.
4. **`ValueFnFromPolicy_FHorz_AmbiguityAversion.m`** + its dispatch branch in
   `ValueFnFromPolicy_FHorz.m`: value the given Policy under min-over-priors continuation. This is
   the bank's inverse check — in the ResidualAsset build it was the gate that caught what the
   structural checks missed.
5. **`ExogShockSetup_FHorz_AmbiguityAversion.m`** + the `if AmbiguityAversion` branch near the top of
   `ExogShockSetup_FHorz.m` — Section 5. This replaces the ad-hoc validation currently inside the
   ambiguity dispatcher, and is what makes flat `ambiguity_pi_e` (and all the standard pi input
   shapes) work.
6. **Doc comment** in the dispatcher stating the design decision: ambiguity is over pi only; all
   priors share the model shock grid. (A user who wants "grid ambiguity" can build it by hand as a
   union grid with zero-padded pi's.)

Estimated toolkit-side: ~22 new files, renames of 7 existing, edits to 3 files
(`ValueFnIter_Case1_FHorz.m`, `ValueFnFromPolicy_FHorz.m`, `ExogShockSetup_FHorz.m`).
GulPesendorfer and InfHorz are untouched.

## 5. Interaction with ExogShockSetup_FHorz

`ValueFnIter_Case1_FHorz` calls `ExogShockSetup_FHorz(n_z,z_grid,pi_z,N_j,Parameters,vfoptions,3)`
before the preference dispatch. That function owns everything about shock inputs: accepted shapes
(stacked column vs joint grid, age-independent vs age-dependent), broadcasting to `_J` form, the
timing/trim conventions (`pi_z_J` gets N_j-1 slices, or N_j when V_Jplus1 is in use; `pi_e_J` gets
N_j columns, or N_j+1 with V_Jplus1; Reading A), and ExogShockFn/EiidShockFn. The ambiguity pi's
currently bypass all of it.

**Design (as suggested in review):** the user inputs everything as normal into ExogShockSetup —
i.e. the call site is unchanged — and near the top of `ExogShockSetup_FHorz` a branch

```matlab
if strcmp(options.exoticpreferences,'AmbiguityAversion') && gridpiboth==3
    [z_gridvals_J,pi_z_J,options]=ExogShockSetup_FHorz_AmbiguityAversion(n_z,z_grid,pi_z,N_j,Parameters,options);
    return
end
```

hands off to a subfunction (`SubCodes/ExoShocks/ExogShockSetup_FHorz_AmbiguityAversion.m`) that sorts out
all the pi, then sends the grid back through a call of `ExogShockSetup_FHorz` to be handled as
normal. Concretely the subfunction:

1. Validates `n_ambiguity` (scalar or age-vector) and that `ambiguity_pi_z`/`ambiguity_pi_z_J`
   are present when there is a z, and the e analogues when there is an e — the checks now living
   in the ambiguity dispatcher move here (and the current unconditional demand for ambiguity_pi_z
   is relaxed so e-only ambiguity models work).
2. Loops over priors, and for prior k slices out its pi and runs it through a recursive
   `ExogShockSetup_FHorz(n_z,z_grid,ambiguity_pi_z(:,:,k),...,2)` call (gridpiboth=2: pi only),
   stacking the results into `options.ambiguity_pi_z_J` of size [N_z,N_z,N_j-1(or N_j),max(n_ambiguity)].
   Same per prior for e into `options.ambiguity_pi_e_J` [N_e,N_j(or N_j+1),max(n_ambiguity)].
   Because each prior goes through the standard pipeline, every accepted pi shape
   (age-independent, [.,.,N_j], [.,.,N_j-1], flat pi_e, [N_e,N_j], ...) and every timing/trim/
   V_Jplus1 convention applies to the ambiguity priors identically and for free — no second
   implementation of those rules. (The recursive calls must not re-enter the ambiguity branch:
   clear the exoticpreferences field, or pass a flag, in the options struct handed to them.)
3. Calls `ExogShockSetup_FHorz(n_z,z_grid,pi_z,...,1)` (gridpiboth=1: grid only) so
   `z_gridvals_J` and `options.e_gridvals_J` are produced exactly as normal — this is also where
   the shared-grid design lands physically: there is one grid pipeline, priors touch only pi.
   The regular `pi_z` argument can additionally be run through as normal (gridpiboth=2) so
   `pi_z_J` comes back well-formed for signature consistency, even though the AA solve never reads
   it (the agent-dist side gets its pi via simoptions' own ExogShockSetup call, unchanged).
4. Warns if the regular pi_z is not equal (to floating-point equality, per age) to one of the
   ambiguity_pi_z priors — the regular pi_z is what the agent distribution etc. will use, so it
   should be one of the priors the agent entertains; likewise pi_e vs the ambiguity_pi_e priors.
   warning(), not error: a researcher may deliberately want a true process outside the prior set.

Consequences worth noting:

- The raws keep their current indexing (`ambiguity_pi_z_J(:,:,jj,k)`, `ambiguity_pi_e_J` at jj+1);
  what changes is that the shapes they receive now follow the standard trim convention (N_j-1
  slices without V_Jplus1) instead of demanding a full-but-never-read N_j slices.
- `ExogShockFn`/`EiidShockFn` return (grid, pi) pairs per age; under ambiguity the pi half is
  ill-posed (which prior would it be?). Simplest rule, mirroring "ambiguity is over pi only":
  error if ExogShockFn/EiidShockFn is combined with AmbiguityAversion. Revisit only if a use case
  appears.
- Cross test 1's flat-vs-`_J` twin run and test 4's V_Jplus1 shapes are the bank's gates on this
  subfunction.
- Only the FHorz variant is needed for this bank; the `_PType`/`_TPath` ExogShockSetup variants
  are untouched until ambiguity reaches those features.

## 6. Order of work

1. Bank first: 9 test files + main script, priors and cross-test pi's fully written out.
   (~165 zero-checks, 6 figures + diary. Bank stays uncommitted.)
2. Toolkit wave in the order of Section 4; after each group, static sweeps (checkcode + the two
   V_Jplus1 drift greps — no octave).
3. GPU run of `CoreFHorzAmbiguityTests.m` on the GPU machine (this sandbox has no GPU); iterate to
   green; commit bank + toolkit together on master.

## 7. Formerly open questions (both answered yes in review)

1. Semiz and with2A deliberately deferred — agreed.
2. Cross tests run z-flavour and e-flavour; a combined z_e flavour is skipped as redundant (both
   code paths covered separately) — agreed.
