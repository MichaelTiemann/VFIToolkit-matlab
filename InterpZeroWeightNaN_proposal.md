# Zero interpolation weights against infinite nodes produce NaN

Status: ExpAssetsemiz fully done -- 684 solver sites (120 files) and 8 ValueFnFromPolicy sites
(4 files), all on the general form. Rest of the toolkit outstanding. Found 2026-09-07 while chasing an `Inf` in the ExpAssetsemiz noa1 test bank.

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

`entireEV(isnan(entireEV))=0` after the sum. It removes the NaN but prices infeasibility at zero,
which is the second half of the bug. Twelve `ExpAssetuSemiExo` raws already carry this idiom (added
in `c1d9dae7`, comment "NaN from 0*(-Inf) at skipinterp positions; treat as zero contribution"), and
two ExpAssetsemiz GI ValueFnFromPolicy files did too. Those 24+2 sites need **correcting**, not
copying.

## Scope

Counts from a loose combine match (`p.*lo + (1-p).*up` in either operand order), excluding sites
already repaired.

### ValueFnIter + TransitionPaths -- outstanding

| family | sites | files |
|---|---|---|
| ExperienceAsset | 720 | 120 |
| ExperienceAssetz | 672 | 96 |
| ExperienceAssetu | 564 | 96 |
| ExperienceAssetze | 336 | 42 |
| ExperienceAssete | 280 | 40 |
| FHorz (misc) | 6 | 4 |
| **total** | **2578** | **398** |

`ExperienceAssetsemiz` (684 sites / 120 files) is already done and excluded.

### ValueFnFromPolicy -- outstanding

60 combine sites in 21 files: RiskyAsset 20, ExpAssetU 16, ExpAsset 8, ExpAssetz 7, ExpAssete 4,
ExpAssetze 3. ExpAssetsemiz (8 sites / 4 files) is done and excluded.

### Not yet surveyed

- **RiskyAsset solvers.** ~176 files contain `skipinterp` but none matched the `EV1`/`EV2` shape;
  they use `skipinterp=logical(WGmatrix...)` and `a2primeProbsWG`. Needs its own shape survey before
  patching -- do not assume the ExpAsset idiom.
- **StationaryDist / SimulateTimeSeries.** They interpolate a2 the same way. Whether a NaN there
  matters depends on whether mass ever reaches an infeasible state; likely it cannot, but confirm
  rather than assume.
- 3199 further `skipinterp` sites did not match either pattern inside a 12-line window. These are
  unclassified, **not** proven safe.

## Sequencing

1. ~~Upgrade the 684 ExpAssetsemiz solver sites to the two-line general form.~~ DONE. The first-pass
   guard keyed on `skipinterp` only covered positions where the two nodes are equal; it missed
   `aprimeProbs==1` against an infinite upper node and `==0` against an infinite lower one. It was
   sufficient for the observed model only because that model's infeasible region sits at `a2` index
   1, the bottom of the grid, so an infinite upper node cannot occur. Do not repeat the partial form
   elsewhere -- go straight to the general one.
2. Survey RiskyAsset's shape; extend the rule to it.
3. Sweep ValueFnIter + TransitionPaths, family by family, each family's test bank rerun before
   moving on.
4. Sweep the 60 ValueFnFromPolicy sites alongside their family.
5. Correct the 24 `ExpAssetuSemiExo` after-the-sum sites.
6. Decide on StationaryDist / SimulateTimeSeries.

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

ExpAssetsemiz and QH ExpAssetsemiz now. Then, per family swept: ExpAsset, ExpAssetU, ExpAssete,
ExpAssetz, ExpAssetze, RiskyAsset (+EZ), and the QH counterpart of each.

Expect the noa1 moment printouts to shift in any bank that had reachable infeasible states -- the
old numbers came from a policy attracted to dead ends priced at zero, so a change there is the fix
working, not a regression.
