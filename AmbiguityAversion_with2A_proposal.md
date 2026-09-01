# Proposal: AmbiguityAversion with2A tier

2026-09-01, v1 — APPROVED (including the companion fix; T3-2A stays plain-only; semiz stays
deferred).
**STATUS: bank BUILT 2026-09-01 (8 new files + with2A section in the main script + companion T4
DC/GI/DC+GI legs in the 2 existing 1A cross-test files; bank now totals 416 zero-checks: 156
main tier + 100 with2A variants + 112 with2A cross + 48 companion; unrun).
Toolkit wave BUILT 2026-09-01 (all of Section 3): 18 AmbAverse_{DC2A,GI2A,DC2A_GI2A} raws
(mechanical transforms with three DC2A-specific pre-normalization rules: V_Jplus1-named bases and
combined base-multiply EV lines; GI2A/DC2A_GI2A interp conditional on the prior via
min(interp1(a1_grid,reshape(ambEV,...),a1prime_grid),[],<priordim>) — the DC2A_GI2A form folds
DiscountFactorParamsVec inside the min, which commutes); 2A branches in the three level-2
dispatchers (level1n/ngridinterp guards, UnKron2/3 tails, length(n_a)>2 errors); FromPolicy GI2A
via the QH-style a2prime-fold into alower (per-prior lookups then unchanged). Static sweeps green.
GPU-GREEN 2026-09-02: 416/416 checks, 0 errors (374 exact zeros; 42 ULP-floor lines in three
benign classes — the two known ones plus a new expected one: GI2A-vs-DC2A_GI2A at ~1e-14 in the
six 2A subcodes, inherited from the donors' discount-before vs discount-after interpolation
asymmetry). One real bug found and fixed along the way, caught by the companion T4 GI leg:
interp1 returns the QUERY's shape when its value input collapses to a vector, which the e-only
GI1 raws' ambEV does when age-varying n_ambiguity hits 1 — fixed by transposing the query to a
column in all 24 GI1/DC1_GI1 interp sites (bit-identical for the matrix/N-D cases). Every
V_Jplus1 check is an exact zero at every tier, 1A and 2A. Ready to commit the whole
AmbiguityAversion family together.**

Extends the (complete, GPU-green) AmbiguityAversion family to TWO standard
endogenous assets — the `n_a=[n_a1,n_a2]` models that trigger the DC2A/GI2A/DC2A_GI2A code
paths. This closes the larger of the two deferrals recorded in
`AmbiguityAversion_testbank_proposal.md` (the other being semiz). Same working method as the
main tier: test bank written first, toolkit wave second, one GPU run to green, commit together.

All conventions carry over unchanged from the main tier: ambiguity is over pi only (shared
grids), e-min before z-min, GI/interpolation conditional on the prior with the min over priors
taken last, DC purely pointwise so the pointwise worst-case EV is already conditional-on-prior,
pi-consistency warning, `ExogShockSetup_FHorz_AmbiguityAversion` untouched (nothing in it
depends on n_a).

---

## 1. What already works today vs what blocks

Because the 24 main-tier raws were built from the modern generic donors, the **plain tier
already accepts `n_a=[n_a1,n_a2]`**: the plain AmbAverse raws are generic in `prod(n_a)`, the
main dispatcher's plain path and its UnKron1 calls are n_a-generic, and the non-GI half of
`ValueFnFromPolicy_FHorz_AmbiguityAversion` is generic too. So the with2A subcodes' baseline
solves should pass on day one.

What blocks, all by explicit error (nothing silent):

1. The three level-2 dispatchers (`ValueFnIter_FHorz_AmbAverse_DC/_GI/_DC_GI`) error on
   `~isscalar(n_a)` — placed there deliberately when the main tier was built.
2. No `AmbAverse_DC2A/GI2A/DC2A_GI2A` raws exist.
3. `ValueFnFromPolicy_FHorz_AmbiguityAversion` errors on `gridinterplayer==1 && ~isscalar(n_a)`
   (its GI block is GI1-only; needs the GI2A a2prime-folding block).

## 2. Bank layout

New `With2A_subcodes/` under `withAmbiguityAversion_subcodes/`, called from a new with2A
section at the bottom of `CoreFHorzAmbiguityTests.m` (mirroring the QH bank's section):

```
withAmbiguityAversion_subcodes/With2A_subcodes/
  AmbFHorz_nod_z_noe_nosemiz_with2A.m    % fig 7
  AmbFHorz_d_z_noe_nosemiz_with2A.m      % fig 8
  AmbFHorz_nod_noz_e_nosemiz_with2A.m    % fig 9
  AmbFHorz_d_noz_e_nosemiz_with2A.m      % fig 10
  AmbFHorz_nod_z_e_nosemiz_with2A.m      % fig 11
  AmbFHorz_d_z_e_nosemiz_with2A.m        % fig 12
  CrossTests/
    AmbFHorz_CrossTests_nod_nosemiz_with2A.m
    AmbFHorz_CrossTests_d_nosemiz_with2A.m
```

8 test files (6 variants — no noz+noe, as ever — plus 2 cross-test files). Setup mirrors the QH
2A section exactly: `n_a_2A=[n_a,4]`, `a2_grid=[0;1;2;3]` stacked under `a_grid`,
`Params.phi1=3; Params.phi2=0.1`, reusing the existing `With2A_ReturnFns/` (the 6 needed
variants all exist). Big-grid GI-convergence sections use `n_a_2A_big=[1001,4]` for the nod
variants and `n_a_notsobig=[501,4]` for the d variants (the QH bank's OOM dodge — copy it).
Ambiguity priors: the same three baseline priors from the main tier, unchanged.

### The 6 variant subcodes

Identical skeleton to figs 1-6 (which is itself the QH-minus-Valt skeleton): baseline vs DC2A,
lowmemory ladder (=1 everywhere, =2 in the z&e pair), ValueFnFromPolicy at plain and GI, GI2A
vs DC2A+GI2A, big-grid moments + figure. 4x14 + 2x22 = **100 zero-checks**, same as the main
tier, plus 6 figures.

### The 2 cross-test files

The main tier's cross tests already gate the pi-plumbing exhaustively, so the 2A cross tests
have one job: pin each new 2A raw family to something known-good. Per file (nod, d), each in a
markov-z and an iid-e flavour:

- **T1-2A — identical priors = vNM, at all four tiers.** Three identical priors vs
  `exoticpreferences='None'`, solved at plain, DC2A, GI2A and DC2A+GI2A, compared tier-by-tier.
  This is the acceptance test for each 2A raw as it lands: it pins the ambiguity raw to its own
  exponential donor exactly (same tier, same model), which is the sharpest oracle available.
  (4 tiers x 2 checks x 2 flavours = 16 checks.)
- **T3-2A — unambiguously worse pi binds** (both slot orders), plain tier only — the maxmin
  logic itself was proven at the main tier; this just confirms it survives the 2A state space.
  (8 checks.)
- **T4-2A — age-varying n_ambiguity via V_Jplus1, at all four tiers.** Same construction as the
  main tier's T4 (single prior in the second half = vNM there; first half = short solve seeded
  with the vNM V), but with the (a)/(b)/(c) solves run at each tier. This is deliberate: it
  gives the DC2A/GI2A/DC2A_GI2A raws' V_Jplus1 branches runtime coverage from day one —
  the drift audits keep showing that V_Jplus1 branches are where transformation faults hide.
  (4 tiers x 4 checks x 2 flavours = 32 checks.)

56 checks per file, **112 cross-test checks**, ~212 new checks in total.

### Recommended companion fix (main tier, 2 existing files)

Extend the existing 1A cross-test files' T4 with DC, GI and DC+GI solves of (a) and (c) — the
identical pattern as T4-2A above. This closes the one honest gap the coverage doc records for
the main tier ("the 18 DC/GI/DC+GI raws' V_Jplus1 branches have no runtime coverage yet") at
the cost of +24 checks per file. Cheap, and this wave is the natural time to do it.

## 3. Toolkit wave

1. **18 new raws**: `AmbAverse_{DC2A, GI2A, DC2A_GI2A}` x {nod,d} x {z, noz_e, z_e}, generated
   from the standard donors (`DivideConquer/DC2A/`, `GridInterpLayer/GI2A/`,
   `DivideConquerGridInterpLayer/DC2A/`) by the same mechanical transform as the main tier:
   signature + per-prior loop + min at the EV sites, nothing else, with hard assertions on the
   site counts and diff review per file. Two 2A-specific notes:
   - the DC2A donors use a combined base-and-multiply EV line
     (`EV=V(:,:,jj+1).*shiftdim(pi_z_J(:,:,jj)',-1);`) that the main-tier transformer did not
     need a pattern for — catalogue the 2A donors' exact site shapes first, as before;
   - GI2A/DC2A_GI2A interpolate over a1 only (`EVinterp=interp1(a1_grid,EV,a1prime_grid)`), so
     the conditional-on-prior fix is the same one-liner with the same min-last placement:
     `EVinterp=min(interp1(a1_grid,ambEV,a1prime_grid),[],<priordim>)`.
2. **Level-2 dispatchers**: replace the `~isscalar(n_a)` errors with 2A branches mirroring the
   standard `ValueFnIter_FHorz_DC/GI/DC_GI` dispatchers: the DC2A `level1n` handling
   (`[floor(sqrt(n_a(1))),n_a(2)]` default; DC in the first endo state only), the GI2A
   `isscalar(ngridinterp)` guard, `length(n_a)>2` stays an error (toolkit-wide: DC2A/GI2A
   hardcode one a2 — see the 3-endo-states finding), and the UnKron tails per the standard
   dispatchers (DC2A keeps UnKron1; GI2A/DC2A_GI2A use UnKron2 for nod / UnKron3 for d with the
   n_a1/n_a2 split).
3. **`ValueFnFromPolicy_FHorz_AmbiguityAversion`**: add the GI2A (`l_a>=2`) blocks from the
   `ValueFnFromPolicy_FHorz_GI` donor — a2prime folded into the linear index
   (`lower_lin=alower+n_a1*(a2prime-1)+zidxoffset`), then the per-prior L2-weighted lookup with
   the min over priors last, exactly as the GI1 blocks now do.
4. Doc comment updates in the dispatcher (with2A no longer deferred) and, after the green run,
   the proposal/coverage-doc bookkeeping (AA column 8 -> 16; the with2A deferral line closes;
   the V_Jplus1 gap note closes if the companion fix is taken).

Estimated: 18 new raw files, 3 dispatcher edits, 1 FromPolicy edit, 8 new + 1 edited bank
files (+2 edited cross-test files if the companion fix is taken). No changes to
ExogShockSetup, the plain raws, or the main dispatcher's plain path.

## 4. Order of work

1. Bank first (8 files + the with2A section of the main script), as ever uncommitted.
2. Toolkit wave in the Section 3 order; static sweeps (balance, name=file, age-index pairing,
   callee existence, diff-vs-donor review) after each group.
3. One GPU run of `CoreFHorzAmbiguityTests.m` (now figs 1-12); iterate to green; commit the
   whole AmbiguityAversion family — main tier + with2A — together on master.

## 5. Open questions

1. Take the companion fix (DC/GI/DC+GI legs added to the existing 1A cross-test T4)? Recommended.
2. T1-2A compares ambiguity-with-identical-priors to vNM at every tier; T4-2A covers V_Jplus1 at
   every tier. Is plain-only T3-2A enough, or do you want the worse-pi-binds test at all four
   tiers as well (+24 checks per file)?
3. Semiz remains deferred (it would need an AmbAverse semiz raw family and bothz handling —
   a separate proposal if wanted). Agreed?
