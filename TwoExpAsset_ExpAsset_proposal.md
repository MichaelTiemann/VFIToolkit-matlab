# Two Experience Assets ("with2A2") — ExpAsset family

Proposal for the first bank in the "add *with two experience assets* to every `ExpAsset*` test
bank" programme. Order within the bank is **test first, then toolkit** (the convention already used
for the `with2A1` and ResidAsset banks).

Scope of this document: `CoreFHorzExpAssetTests` + `ValueFnIter/FHorz/ExperienceAsset` and its
downstream. The sibling banks (`ExpAssetU`, `ExpAssete`, `ExpAssetz`, `ExpAssetze`,
`ExpAssetsemiz`) are sequenced at the end but not designed here.

---

## STATUS

**Test bank: BUILT, UNRUN (2026-09-02).** 76 new files in `CoreFHorzExpAssetTests`:
32 subcodes (figs 49-80), 32 ReturnFns, 8 cross-tests, 4 cross-test ReturnFns, plus the setup and
main-script blocks. 844 printed zero-checks. The 32 subcodes were derived mechanically from the
one-experience-asset tier and every changed line was diffed and classified against its source;
that check found two real generation faults in `jequaloneDist` (the a1+semiz tier starts at
a2_1 index 2, not 1, so a pattern-matched rule silently skipped all 8 of those files; and the
`noz_noe` files had the index rule applied twice). Both fixed and re-verified.

Nothing is committed: per the test-first-banks rule the bank ships in one commit with its toolkit
support. **Toolkit side (§3) not started.**

Deviation from the design below: §2.4 proposed a fourth cross-test family (`CrossTests8`, a fake
second experience asset driven by a decision variable). It was dropped in favour of a noa1 version
of the inert-second-asset test, which exercises the `N_a1==0` code path — a distinct branch in
every raw — for much less machinery. The three families that survived (inert-second, inert-first,
swap-symmetry) already pin down the dimension ordering, which was the point of the fourth.

---

## 1. What "two experience assets" means, and what already exists

An experience asset is an endogenous state whose next-period value is *not* chosen, but produced by
`vfoptions.aprimeFn`. "Two experience assets" means `length(n_a2)==2`: two such states side by
side, so the model has `a1` (standard, chosen) plus `a2_1` and `a2_2` (both experience).

**This is not greenfield.** Two prior commits already laid the convention and a working slice of
the implementation:

- `56267288` — *ExperienceAsset: support l_a2>=2 via integer-valued vfoptions.experienceasset*
- `1fe20768` — *Multi-dim experienceasset: two-dim a2 interp in aprimeFn matrices, DC2A/GI2A raws...*

There is also a working demo and a hand-written cross-test at
`~/Dropbox/Matlab_Codes/MyProjects/DoubleExpAsset/` (`LifeCycle_DoubleExpAsset.m`, `crosstest.m`)
and a `ze` sibling at `.../DoubleExpAssetze/`. The proposal below is deliberately built on that
existing convention rather than inventing a new one.

### The established user-facing convention (do not change it)

| Item | Convention |
|---|---|
| Switch | `vfoptions.experienceasset=2` (integer = number of a2 dims; `0` off, `1` legacy). Same for `simoptions`. |
| Grid split | last `vfoptions.experienceasset` entries of `n_a` are the experience assets. `a_grid=[a1_grid; a2_1_grid; a2_2_grid]`. |
| `aprimeFn` | `@(d2..., a2_1, a2_2, whicha, params...)` — a **`whicha` integer selector** sits between the a2 inputs and the params. Builder calls `arrayfun` once with `whicha=1` (returns `a2_1'`) and once with `whicha=2` (returns `a2_2'`). GPU `arrayfun` is scalar-output only, hence the selector. |
| `ReturnFn` | gains one argument: `(d..., a1prime, a1, a2_1, a2_2, z, e, params...)`. Same for `FnsToEvaluate`. |
| `Policy` | **unchanged** — still `d`'s and `a1prime` only. The a2 dims are never chosen. |
| Interpolation | per-dim factored, *not* Kron-folded: builders return `a2primeIndexes(k,...)` = lower index in dim `k`, `a2primeProbs(k,...)` = prob of lower. Consumers do **nested 2-corner interp with `skipinterp` at each level**, with per-contribution NaN cleanup so `0*(-Inf)` at a zero-prob corner does not poison the sum. |

### Audit: what is implemented today

Peel / plumbing — **done**:

- `SubCodes/NonStandardEndoStates/SetupNonStandardEndoStates_FHorz.m` slices `n_a2` by
  `vfoptions.experienceasset`, generically.
- `PolicyInd2Val/PolicyInd2Val_FHorz.m` (and `_InfHorz`, `_TPath`) subtract
  `vfoptions.experienceasset` rather than `1`.
- All six non-fastOLG return-fn builders in `ReturnFnMatrix/ExperienceAsset/` have `l_a2==2` arms:
  `..._Disc`, `..._Disc_noz`, `..._Disc_e`, `..._Disc_DC2A`, `..._Disc_DC2A_noz`, `..._Disc_DC2A_e`.
  These serve the plain **and** the DC/GI tiers (DC/GI differ only by `Level`/fine `a1prime` grid),
  so the return-matrix side needs no further work for this bank.
- `aprimeFnMatrix/CreateExperienceAssetFnMatrix{,_J}.m` and
  `aprimeFnMatrix/CreateaprimePolicyExperienceAsset{,_J}.m` have `l_a2==2` branches, including the
  `N_semizze>0` layouts (so the SemiExo raws have their builder support already).

Solvers — **almost nothing**. Of the **384** raws under `ValueFnIter/FHorz/ExperienceAsset/`,
exactly **6** carry an `l_a2==2` arm:

```
ValueFnIter_FHorz_ExpAsset_noz_raw.m
ValueFnIter_FHorz_ExpAsset_nod1_noz_raw.m
QuasiHyperbolic/ValueFnIter_FHorz_QuasiHyperbolicExpAsset{N,S}_{,nod1_}noz_raw.m   (4)
```

Downstream — **partial**:

| File | l_a2=2 status |
|---|---|
| `StationaryDist_FHorz_ExpAsset_noz.m` | done (Kron-folds per-dim to `Kaprimepts=4`); **errors** on `gridinterplayer=1` |
| `StationaryDist_FHorz_ExpAsset.m` (with z) | **not done** |
| `ValueFnFromPolicy_FHorz_ExpAsset.m` | done for the `N_z==0 && N_e==0` branch only |
| `ValueFnFromPolicy_FHorz_QuasiHyperbolic_ExpAsset.m` | same partial state |
| SemiExo / TPath / fastOLG | **not done** |

**The dangerous gap:** apart from one `error()` in `StationaryDist_FHorz_ExpAsset_noz.m`, there is
**no guard anywhere**. A user who sets `experienceasset=2` with `z`, or `e`, or `divideandconquer`,
gets either a cryptic shape error deep in a raw or — worse — silently wrong numbers. Closing that is
stage T0 below and should land regardless of how far the rest gets.

### The reference block to copy

`ValueFnIter_FHorz_ExpAsset_noz_raw.m:53-88` is the canonical `l_a2==2` EV block (and its twin in
the in-loop half of the same file). It appears **twice per raw** (terminal/`V_Jplus1` half, and
in-loop half) and is independent of the `lowmemory` rungs. Porting a raw is mechanical:

```matlab
% a2primeIndex/a2primeProbs shape [l_a2=2, N_d2, N_a2] per-dim. Nested 2-corner with skipinterp.
loIdx_1 = ...; loIdx_2 = ...; prob_1_exp = ...; prob_2_exp = ...;
aprime_ll/hl/lh/hh = a1prime_offsets + N_a1*(loIdx_1(+1) + n_a2_1*(loIdx_2(+1)-1) - 1);
V_ll/hl/lh/hh      = EVpre(...);
p1 = prob_1_exp; p1(V_ll==V_hl)=0;          % skipinterp, inner dim, low branch
EV_lo = p1.*V_ll + (1-p1).*V_hl;            % with per-term isnan(...)=0
... same for the high branch ...
p2 = prob_2_exp; p2(EV_lo==EV_hi)=0;        % skipinterp, outer dim
EV = p2.*EV_lo + (1-p2).*EV_hi;             % with per-term isnan(...)=0
```

The only thing that varies raw to raw is the trailing-dimension shape (`z`, `e`, `semiz`) that
`repmat`/`reshape` must respect — which is exactly where copy-paste drift historically bites (see
the `vjplus1-branch-drift` experience). Diff each ported block against its own in-loop twin, not
against the reference file.

---

## 2. Test bank design — `CoreFHorzExpAssetTests`

### 2.1 Naming

Suffix **`with2A2`**, mirroring the existing `with2A1` (two standard `a1` assets). So:

```
CoreFHorzExpAssetTests_subcodes/With2A2_subcodes/                    (a1 + 2 expassets)
CoreFHorzExpAssetTests_subcodes/With2A2_subcodes/Semiz_subcodes/
CoreFHorzExpAssetTests_subcodes/With2A2_subcodes/Noa1_subcodes/      (noa1 + 2 expassets)
CoreFHorzExpAssetTests_subcodes/With2A2_subcodes/Noa1_subcodes/Semiz_subcodes/
CoreFHorzExpAsset_ReturnFns/With2A2_ReturnFns/  (mirroring the same four-way split)
```

Subcode file names: `CoreFHorzExpAsset_<d1>_<z>_<e>_<semiz>_with2A2.m` and
`..._noa1_<semiz>_with2A2.m`.

### 2.2 Setup additions (`CoreFHorzExpAsset_setup.m`)

Append (leaving everything existing untouched, as the `with2A1` block did):

```matlab
% with2A2: a SECOND experience asset a2_2, appended after the existing experience asset a2_1.
% Grid layout: a = [a1, a2_1, a2_2].  vfoptions.experienceasset=2.
n_a2_1=9;                                  % first experience asset  (human capital)
n_a2_2=7;                                  % second experience asset (cumulated earnings)
a2_1_grid=linspace(0,10,n_a2_1)';
a2_2_grid=linspace(0,10,n_a2_2)';
n_a_2A2=[n_a(1),n_a2_1,n_a2_2];
a_grid_2A2=[a1_grid;a2_1_grid;a2_2_grid];
n_a_2A2_justexpasset=[n_a2_1,n_a2_2];      % noa1 tier
a_grid_2A2_justexpasset=[a2_1_grid;a2_2_grid];

% aprimeFn with the whicha selector. a2_2 is DELIBERATELY coupled to a2_1: an index/stride or
% dim-ordering bug then changes the answer instead of cancelling out.
vfoptionsbaseline.aprimeFn_2A2=@(d2,a2_1,a2_2,whicha,phi1,phi2,phi3,phi4) ...
    (whicha==1)*(phi1*(1-d2)+(1-phi2)*a2_1) + ...
    (whicha==2)*(phi3*d2*a2_1+(1-phi4)*a2_2);
Params.phi3=0.4;    % rate at which d2*a2_1 accumulates into a2_2
Params.phi4=0.05;   % depreciation of a2_2
```

Note the grid sizes drop from the `n_a_justexpasset=13` of the 1-expasset tier: `9x7=63` vs `13`,
so the state space grows ~5x. Given that figs 40/44-48 of the existing bank already OOM,
`n_a2_1=9, n_a2_2=7` (and smaller `n_a_big` variants for the `z_e` cases) is the starting point,
tunable down to `7x5` if the GPU box complains.

### 2.3 Tiers and figure numbering

The bank currently ends at fig 48. The new tier mirrors the bank's own noa1-then-a1 ordering:

| Figs | Tier | `n_a` | Endo states | DC/GI? |
|---|---|---|---|---|
| 49–56 | noa1 + 2A2, nosemiz | `[n_a2_1,n_a2_2]` | 2 | no (no `a1` to divide) |
| 57–64 | noa1 + 2A2, semiz | `[n_a2_1,n_a2_2]` | 2 | no |
| 65–72 | a1 + 2A2, nosemiz | `[n_a1,n_a2_1,n_a2_2]` | 3 | yes — DC1/GI1/DC1+GI1 on `a1` |
| 73–80 | a1 + 2A2, semiz | `[n_a1,n_a2_1,n_a2_2]` | 3 | yes |

Each block is the standard 8: `{nod1,d1} x {noz,z} x {noe,e}`.

**DC2A/GI2A x 2A2 (two standard + two experience assets = 4 endogenous states) is explicitly out of
scope.** It is a real combinatorial tier but it is not what "with two expasset" asks for, it would
OOM on the current box, and per the `three-endostate-plan-supersedes-dc2a-fold` note the DC2A
helpers hardcode one `a2` anyway.

32 new figures, 32 new subcodes, 32 new ReturnFns. Roughly in line with the noa1 tier (16) plus the
a1 tier (16); expect ~600–700 printed checks by the bank's usual density (~15 per nosemiz-simple
subcode, ~49 per `d1_z_e`, ~3–5 per noa1 subcode).

### 2.4 Cross-tests — the part that actually catches bugs

These are the reason to build the tier at all. Four new families, in
`CoreFHorzExpAssetTests_subcodes/CrossTests/`:

1. **`CrossTests5_*_with2A2` — inert second asset.** `a2_2' = a2_2` (identity), `a2_2` absent from
   the `ReturnFn`. `V`/`Policy` must be exactly independent of the `a2_2` dim and must equal the
   1-expasset model at every `a2_2` slice, **at machine precision**. This is Model A vs Model B of
   the existing `DoubleExpAsset/crosstest.m`, promoted into the bank.
2. **`CrossTests6_*_with2A2` — inert *first* asset.** Roles swapped: `a2_1' = a2_1` inert, `a2_2`
   carries the real law. Same machine-precision equality. Catches stride/dim-ordering bugs that
   test 1 is blind to (test 1 alone passes even if the two dims are transposed).
3. **`CrossTests7_*_with2A2` — swap symmetry.** Identical grids (`n_a2_1==n_a2_2`), identical laws,
   `ReturnFn` symmetric in `(a2_1,a2_2)`. Then `V` must equal `permute(V,[1 3 2 ...])` exactly.
   Cheap, and a strong check on the nested-interp ordering.
4. **`CrossTests8_*_with2A2` — fake second experience asset.** `a2_2' = d3` with `d3` a decision
   variable on the same grid as `a2_2`, degenerate so the model reduces to a standard 1-expasset
   model with an extra `d`. Mirrors the existing `CrossTests3` idea one level up.

Plus: extend the existing `CrossTests` (Markov-as-iid) and `CrossTests2` (semiz-as-Markov) to the
2A2 tier, `nod1`/`d1` x `nosemiz`/`semiz` — cheap, since they are parameter swaps on subcodes that
will already exist.

Every subcode also carries the bank-standard checks: `ValueFnFromPolicy` vs `V`, DC vs plain, GI vs
plain on the big grid, `SimPanel` vs `LifeCycleProfiles`, the `V_Jplus1` `jstar` pair, and the
`lowmemory` rungs where shocks exist.

### 2.5 Expected first-run state

Written test-first, essentially every one of the 32 figures errors out on the first run (only
figs 65 and 66 — a1, nosemiz, noz — touch the two raws that already have the `l_a2==2` arm, and
even those will fail at `StationaryDist`/`ValueFnFromPolicy` if `z` or GI is involved). That is the
intended state: the bank is the specification, and it is not committed until its toolkit support
exists (per the `test-first-banks-commit-with-implementation` rule).

---

## 3. Toolkit-side work plan

Staged so each stage is independently GPU-verifiable against a subset of the new figures.

### T0 — guards (do first; tiny; ships alone)

Add `error('... not yet supported with two experience assets (l_a2=2) ...')` at every entry that
would otherwise silently mis-solve. Per the `exoticpref-checks-in-dispatchers` convention these go
in the **dispatchers**, not the raws:

- `ValueFnIter/FHorz/ExperienceAsset/ValueFnIter_FHorz_ExpAsset.m`
- `.../ExperienceAsset/DivideConquer/ValueFnIter_FHorz_ExpAsset_DC.m` (+ GI, DC_GI dispatchers)
- `.../ExperienceAsset/ExpAssetSemiExo/ValueFnIter_FHorz_ExpAssetSemiExo.m` (+ its DC/GI/DC_GI)
- `.../ExperienceAsset/QuasiHyperbolic/ValueFnIter_FHorz_QuasiHyperbolicExpAsset.m`
- `StationaryDist/FHorz/ExpAsset/StationaryDist_FHorz_ExpAsset.m`
- `ValueFnFromPolicy/ExpAsset/ValueFnFromPolicy_FHorz_ExpAsset.m` (non-noz/noe branches)

Each guard is deleted as its stage below lands. This is worth doing even if the rest slips.

### T1 — plain raws (14 files)

`ValueFnIter/FHorz/ExperienceAsset/`: the 16 plain raws minus the 2 already done —
`_raw`, `_e_raw`, `_noz_e_raw`, `_nod1_raw`, `_nod1_e_raw`, `_nod1_noz_e_raw`,
`_noa1_raw`, `_noa1_noz_raw`, `_noa1_e_raw`, `_noa1_noz_e_raw`,
`_nod1_noa1_raw`, `_nod1_noa1_noz_raw`, `_nod1_noa1_e_raw`, `_nod1_noa1_noz_e_raw`.
Two blocks each. Unblocks figs 49–56 and the plain solves of 65–72.

### T2 — `StationaryDist` + `ValueFnFromPolicy` (with z / with e)

- `StationaryDist_FHorz_ExpAsset.m`: port the `Kaprimepts=2^l_a2` fold that already exists in the
  `_noz` sibling.
- `StationaryDist_FHorz_ExpAsset_noz.m`: replace the `gridinterplayer` `error()` with the real
  thing — the block hardcodes `1:2`/`3:4` for `Kaprimepts=2`; with 4 corners it becomes `1:4`/`5:8`
  and `K=8`. Same generalisation then applies to the with-z file.
- `ValueFnFromPolicy_FHorz_ExpAsset.m`: extend the `l_a2==2` arm from the `N_z==0 && N_e==0` branch
  to the other three branches.

Without T2 the bank's `ValueFnFromPolicy` / moment / GI checks cannot even print, so T1 and T2 are
effectively one GPU run.

### T3 — DC1 / GI1 / DC1+GI1 (24 files)

`DivideConquer/ValueFnIter_FHorz_ExpAsset_DC1_*_raw.m` (8),
`GridInterpLayer/..._GI1_*` (8), `DivideConquerGridInterpLayer/..._DC1_GI1_*` (8).
Return-matrix side is already done; only the EV block changes. Completes figs 65–72.

### T4 — `ExpAssetSemiExo` (40 files)

16 plain + 8 DC1 + 8 GI1 + 8 DC1_GI1. The `N_semizze>0` layouts in
`CreateaprimePolicyExperienceAsset_J.m` already exist, so this is the same port with one more
trailing dim. Plus the semiz path through `StationaryDist`. Completes figs 57–64 and 73–80.

### T5 — cleanup (bundle with whichever stage touches the file)

- `CreateaprimePolicyExperienceAsset.m:198` and `CreateaprimePolicyExperienceAsset_J.m:357` each
  define a **local subfunction `local_interp1d`**, against the toolkit's no-helper-functions house
  style. Inline it into each `l_a2==2` arm while the file is open.
- Same files use `d_shape={...}` / `policy_idx_args={...}` cell indirection in the `l_a2==2` arm —
  also off-idiom; write the branches out.

### Deliberately deferred

- **QuasiHyperbolic** (128 raws; 4 already done). The QH sub-bank
  (`withQuasiHyperbolicDiscounting/`) is its own bank and per `exotic-prefs-test-vf-policy-only`
  needs only V+Policy checks, so it is a much cheaper follow-on once T1–T4 are green.
- **TPath / fastOLG** — `CreateReturnFnMatrix_fastOLG_ExpAsset_Disc*` have no `l_a2==2` arm at all;
  that belongs with `CoreFHorzTPathExpAssetTests`.
- **DC2A/GI2A x 2A2** — see §2.3.

---

## 4. Risks and open decisions

1. **Memory.** 3 endogenous states plus `z` and `e` plus a fine `a1prime` grid for GI is the
   combination that already OOMs at `with2A1`. Expect figs 72 and 80 (`d1_z_e`) to need reduced
   grids or to be marked `CANNOT RUN` like figs 40/44–48. Decide up front whether to shrink
   `n_a2_2` to 5 for the `z_e` cases rather than discovering it mid-run.
2. **Off-grid clamping.** With `a2_2' = phi3*d2*a2_1 + (1-phi4)*a2_2`, some states push `a2_2'` past
   the top of its grid, where the builder clamps to `n_a2_2-1` with prob 0. That is worth
   exercising, but it makes the "inert" cross-tests the only machine-precision checks — the main
   figures are eyeball/consistency checks, as elsewhere in the bank.
3. **Zero-prob corners and `-Inf`.** The nested interp's per-contribution `isnan(...)=0` cleanup is
   load-bearing (`0*(-Inf)`). Any ported block that drops it produces `NaN` only in the corner
   region — which the aggregate checks may not surface. Worth a targeted check in the bank at a
   state known to be at the grid edge.
4. **`l_a2>2`.** Both builders hard-`error` above 2. Leave it there; three experience assets is a
   separate decision.
5. **Naming.** `with2A2` is proposed for symmetry with `with2A1`. If `with2ExpAsset` reads better
   to you, say so before the 32 files are named — renaming afterwards touches four directories,
   the ReturnFns, and the main script.

## 5. Sequencing across the rest of the family

After ExpAsset, in order of decreasing existing support:

1. **`ExpAssetze`** — already has 10 core + 16 QH raws with `l_a2==2` (the only family further along
   than ExpAsset on the solver side) and a hand-written `DoubleExpAssetze` crosstest to promote.
   Its bank has no coverage at all, so this is mostly test-side work.
2. **`ExpAssetz`**, **`ExpAssete`** — builders (`CreateExperienceAssetz/eFnMatrix`) are 1D-only;
   need the `l_a2==2` builder arm *and* the raws.
3. **`ExpAssetU`**, **`ExpAssetsemiz`** — same, plus the extra `u` / `semiz` dimension in the
   aprime matrices.

`vfittoolkitcoverage.md` should gain a "2A2" column in the ExpAsset-family table as each lands.
