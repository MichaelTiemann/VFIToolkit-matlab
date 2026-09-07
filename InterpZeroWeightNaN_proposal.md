# Zero interpolation weights against infinite nodes produce NaN

Status: ExpAssetsemiz fully done -- 684 solver sites (120 files) and 8 ValueFnFromPolicy sites
(4 files), all on the general form (`ac60a3ae`). 5579 solver sites outstanding across four shapes;
see Scope. Found 2026-09-07 while chasing an `Inf` in the ExpAssetsemiz noa1 test bank.

## The defect

Every asset-interpolation site combines the value at two (or, in the grid-interp layer, four)
neighbouring grid nodes with weights that sum to one:

```matlab
entireEV=EV1.*aprimeProbs+EV2.*(1-aprimeProbs);
```

When a weight is exactly `0` and the corresponding node is `-Inf`, `0*(-Inf)` is `NaN`, and the
whole combined value becomes `NaN`. Exact zero weights are not rare -- they are the normal
representation of "a2prime lands exactly on a grid point", "off the bottom of the grid" (prob 1),
"off the top of the grid" (prob 0), and the `skipinterp` guard, which sets the weight to zero
precisely when the two nodes are equal.

`-Inf` in the value function is likewise ordinary: it is what a ReturnFn returns for an infeasible
choice, and it propagates to any state whose entire choice set is infeasible.

### Why it is not merely cosmetic

The NaN does not stay a NaN. One age later the raws hit

```matlab
EV=V_Jplus1.*shiftdim(pi_bothz',-1);
EV(isnan(EV))=0;
```

which exists to kill `0*(-Inf)` from zero transition probabilities, but also silently converts an
inherited NaN into a continuation value of **zero**. A dead end is then priced at 0. Where utility
is negative -- CRRA with sigma>1, as in most of the test banks -- zero is *better* than almost any
real continuation, so the policy is actively attracted to infeasible states. `max` ignores NaN, so
nothing errors and nothing warns.

### Observed instance

`CoreFHorzExpAssetsemizTests` noa1 tier, all 8 subcodes: `ValueFnFromPolicy, this should be zero: Inf`.
To reproduce on the smallest failing case (13x2x20): build the `nod1_noz_noe_noa1` model exactly as
that subcode does, solve it with `ValueFnIter_Case1_FHorz`, recompute with `ValueFnFromPolicy_FHorz`,
and print the finite / -Inf / NaN census of each plus the states where one side is infinite and the
other finite. That gave:

```
before fix   V  : 339 finite, 111 -Inf, 70 NaN     Vfp: 293 finite, 213 -Inf, 14 NaN
after  fix   V  : 293 finite, 227 -Inf,  0 NaN     Vfp: 293 finite, 212 -Inf, 15 NaN
             states with one side infinite and the other finite: 44 -> 0
```

The spurious value was exactly `-4.0000` = `F` at `c=uempbenefit=0.2` plus `beta*0`: the fingerprint
of a dead end priced at zero.

Note the checks themselves cannot see this. `max(abs(A-B))` **ignores NaN**, so wherever both sides
are NaN the comparison silently passes. Coverage over an array containing NaN is weaker than a
`0.000e+00` line suggests.

## The rule

**Zero each product term before summing. Never after the sum.** Zeroing after the sum destroys the
whole expectation, because one NaN term poisons the total. This is the single mistake behind every
variant below.

### Two-node combines

```matlab
entireEV=EV1.*aprimeProbs+EV2.*(1-aprimeProbs);
entireEV(aprimeProbs==0)=EV2(aprimeProbs==0); % includes the skipinterp positions
entireEV(aprimeProbs==1)=EV1(aprimeProbs==1);
```

No-op wherever both nodes are finite, so it is bit-for-bit identical in every model that has no
reachable infeasible states -- which is why the other banks are green and stay green. Costs two
logical masks the size of the array (1 byte/element against 8), so roughly 25% transient overhead;
worth watching in the with2A1 tiers that already OOM.

### Four-node (bilinear) combines, grid-interp layer

Any of the four terms can carry a zero weight, so masks do not generalise. Accumulate instead:

```matlab
% a zero weight against an infinite node gives 0*(-Inf)=NaN, so zero each term BEFORE summing
EVnext=wa1l.*wa2l.*EV_LL; EVnext(isnan(EVnext))=0;
EVterm=wa1l.*wa2u.*EV_LU; EVterm(isnan(EVterm))=0; EVnext=EVnext+EVterm;
EVterm=wa1u.*wa2l.*EV_UL; EVterm(isnan(EVterm))=0; EVnext=EVnext+EVterm;
EVterm=wa1u.*wa2u.*EV_UU; EVterm(isnan(EVterm))=0; EVnext=EVnext+EVterm;
```

Within a single term NaN can only come from `0*(+-Inf)`, since the weights are finite -- so the
per-term `isnan` is exactly "a zero-weight node contributes nothing". A term with a nonzero weight
against an infinite node stays `-Inf` and correctly poisons the sum.

### What NOT to do

`entireEV(isnan(entireEV))=0` after the sum. It removes the NaN but returns 0 -- pricing
infeasibility at zero, and, worse, returning 0 at feasible states too (see "Why the 1551 wrongfix
sites are worse than they look"). **1551 sites across shapes A and B already carry this idiom** and
need **correcting**, not copying: it was added piecemeal wherever someone hit the NaN, e.g. the
twelve `ExpAssetuSemiExo` raws in `c1d9dae7` ("NaN from 0*(-Inf) at skipinterp positions; treat as
zero contribution") and two ExpAssetsemiz GI ValueFnFromPolicy files, both since corrected. Its
presence is a marker of where the bug has already been noticed and mis-handled.

## Scope

Measured 2026-09-07 by scanning every `skipinterp` definition in the tree and classifying the
following twelve lines. **6263 sites.** They come in four shapes, and the shape decides the fix.
`grep`ping for the `EV1`/`EV2` form alone finds barely half of them.

Status column: **FIXED** = the general form (read the weighted node at `prob==0`/`prob==1`, or a
per-term `isnan` before summing); **wrongfix** = `X(isnan(X))=0` placed *after* the sum, which
removes the NaN but returns 0; **none** = nothing at all.

| shape | how it is written | sites | FIXED | wrongfix | none |
|---|---|---|---|---|---|
| A | `skipinterp=(EV1==EV2)` then `entireEV=EV1.*p+EV2.*(1-p)` | 3412 | 684 | 336 | 2392 |
| B | `skipinterp=(Vlower==Vupper)` then `EV=p.*Vlower+(1-p).*Vupper` | 2083 | 0 | 1215 | 868 |
| C | term-split: `EV1=EV(lower).*p; EV2=EV(upper).*(1-p)` | 640 | **640** | 0 | 0 |
| D | `skipinterpWG=logical(WGmatrix(i)==WGmatrix(i+1))` | 128 | 128 | 0 | 0 |

The 684 in A are the ExpAssetsemiz fix (`ac60a3ae`); shape C is complete as of 2026-09-07.
Everything else outstanding.

### By family

Note the QuasiHyperbolic bucket shadows the underlying asset family -- a QH ExpAssetz raw counts as
QuasiHyperbolic here, not ExpAssetz.

- **A**: QuasiHyperbolic 2256 (456 fixed, 224 wrongfix, 1576 none), ExperienceAsset 240,
  ExperienceAssetz 224, ExperienceAssetu 240 (96 wrongfix), ExperienceAssete 112,
  ExperienceAssetze 112 (16 wrongfix), ExperienceAssetsemiz 228 (all fixed).
- **B**: QuasiHyperbolic 1272 (760 wrongfix), ExperienceAssete 176, ExperienceAssetu 176,
  ExperienceAsset 180, TransitionPaths 144, ExperienceAssetz 64, ExperienceAssetze 36,
  InfHorz 14, AmbiguityAversion 8, RiskyAsset 8, EpsteinZin 4, InheritAsset 1.
- **C**: RiskyAsset 360, EpsteinZin 218, AmbiguityAversion 62 -- all fixed.
- **D**: EpsteinZin (RiskyAsset EZ SemiExo GI) 128.

### Shape D is the precedent -- it was already right

```matlab
WG1=WGmatrix(a2primeIndex).*aprimeProbsWG;
WG2=WGmatrix(a2primeIndex+1).*(1-aprimeProbsWG);
% If WG1 or WG2 is infinite, and probability is zero, we will get a nan, so get rid of these
WG1(isnan(WG1))=0;
WG2(isnan(WG2))=0;
WGmatrix=sum((WG1.*pi_u'),2)+sum((WG2.*pi_u'),2);
```

Per-term zeroing before the sum, cause named in the comment, arrived at independently. Copy this,
not the after-the-sum idiom.

### Shape C is the cheapest to fix

The terms are already separate, so it needs no restructuring and no masks -- two lines in the
shape-D style, placed before the terms are combined:

```matlab
EV1=reshape(EV1,[...]).*aprimeProbs;
EV2=reshape(EV2,[...]).*(1-aprimeProbs);
EV1(isnan(EV1))=0; % a zero weight against an infinite node gives 0*(-Inf)=NaN
EV2(isnan(EV2))=0;
```

### Shapes A and B take the two-mask form

Both combine node values rather than terms, so use the masks given under "The rule" above. B's
variable names differ (`Vlower`/`Vupper`, weight `aprimeProbs`) but the structure is identical.

### Why the 1551 "wrongfix" sites are worse than they look

`X(isnan(X))=0` after the sum does not merely mishandle infeasible states. At a genuine `prob==0`
position where the lower node is `-Inf` and **the upper node is finite**, the correct answer is just
the upper node's value; the product form gives `NaN` and the after-the-sum line then returns **0** --
a wrong answer at a state that was never infeasible. These sites need correcting, not leaving.

### Not yet surveyed

- **StationaryDist / SimulateTimeSeries.** They interpolate a2 the same way. Whether a NaN there
  matters depends on whether mass ever reaches an infeasible state; likely it cannot, but confirm
  rather than assume.
- **ValueFnFromPolicy** carries 60 outstanding combine sites in 21 files (RiskyAsset 20, ExpAssetU
  16, ExpAsset 8, ExpAssetz 7, ExpAssete 4, ExpAssetze 3); ExpAssetsemiz's 8 are done. Its GI files
  use a four-node bilinear combine -- see "Four-node (bilinear) combines" above.

## Sequencing

Revised around the shapes rather than the families, since a shape is one mechanical edit and a
family mixes several.

1. ~~ExpAssetsemiz, shape A, 684 sites.~~ DONE (`ac60a3ae`), both banks green.
2. ~~**Shape C, 640 sites / 198 files.**~~ DONE 2026-09-07, bank by bank, each green before the next:
   RiskyAsset main (360 sites/108 files, 1202 checks, max 4.547e-13), RiskyAsset EZ (218/62, 6010
   checks, max 2.132e-14), RiskyAsset AA (62/28, 220 checks, max 1.776e-14). No NaN, no Inf, no
   errors in any of the three.
3. **Shape B's 1215 wrongfix sites**, which are actively wrong at feasible states, then its 868
   bare ones. Heaviest in QuasiHyperbolic; rerun each QH bank plus TransitionPaths.
4. **Shape A's remaining 2728**, family by family, each bank rerun before moving on.
5. ValueFnFromPolicy's 60 sites, alongside their family.
6. Decide on StationaryDist / SimulateTimeSeries.

Shape D needs nothing.

## Lessons for the sweep

- **Use a loose regex and reconcile the found-count.** A strict end-of-line match missed 12 of the
  684 ExpAssetsemiz sites because those statements carried a trailing `% [N_d2, N_a2, N_bothz]`
  shape comment -- and all 12 were in the `noa1_e` raws, the tier that actually fires. A partial
  sweep would have left the most exposed files unfixed while reporting success.
- Verify after patching that every inserted line's guard variable is in scope, and that the diff is
  insertions only.
- `checkcode` via `matlab -singleCompThread -batch` runs in the sandbox and catches parse errors
  without a GPU.
- The finite / -Inf / NaN census is the diagnostic that cracked this. A plain `max(abs(A-B))` cannot
  see it, because `max` ignores NaN; count the categories on each side separately and compare.

## Test banks to rerun

ExpAssetsemiz and QH ExpAssetsemiz are done and green. Then, per sweep step: RiskyAsset,
RiskyAsset EZ and AmbiguityAversion (shape C); every QH bank plus the TransitionPaths banks
(shape B); then ExpAsset, ExpAssetU, ExpAssete, ExpAssetz, ExpAssetze and each QH counterpart
(shape A).

The TransitionPaths banks are the weak link: 144 shape-B sites sit under `TransitionPaths/FHorz`,
and most TPath check sites have never been run -- a wrong answer there is silent, not an error.

Expect the noa1 moment printouts to shift in any bank that had reachable infeasible states -- the
old numbers came from a policy attracted to dead ends priced at zero, so a change there is the fix
working, not a regression.
