# Proposal: AmbiguityAversion + RiskyAsset

2026-09-02, v1 — u treated as ambiguity (mandatory ambiguity_pi_u) per review.
**STATUS: bank BUILT 2026-09-02 (19 files, 240 zero-checks + 4 warning checks + 1 error assert).
Toolkit wave BUILT same day: 40 raws (segment transformer v3: stacked per-prior z/e-cores with
marker-paired lottery wraps; running argmin with component tracking so identical priors reproduce
the exponential donor bit-for-bit — T1 exact), 4 dispatchers (2A1 raw-calls become explicit
errors), riskyasset branches in ValueFnIter_FHorz_AmbiguityAversion (which now takes d_grid, not
d_gridvals) and ValueFnFromPolicy_FHorz_AmbiguityAversion, ExogShockSetup_FHorz_AmbiguityAversion
riskyasset block (mandatory ambiguity_pi_u + pi_u warning + no-shocks relaxation), and
ValueFnFromPolicy_FHorz_AmbAverse_RiskyAsset(+_GI) — note the base FromPolicy donor's z-cases use
a per-z'-lottery factorization; the AA versions instead mirror the raws' factorization (z-summed
per-prior EV through the lottery), which is ULP-equivalent for vNM and required for the mins.
Static sweeps green. GPU run 1 (2026-09-02): first 4 noa1 groups green (exact zeros, FromPolicy
at ULP), then crashed on an undefined `aprimeProbs` — the transformer's per-prior
`aprimeProbsK=aprimeProbs;` initializer assumed a pre-existing aprimeProbs, but in most riskyasset
donors the probs are built fresh inside the segment (repmat from a2primeProbs, or the
CreateRiskyAssetFnMatrix call itself). Fixed 2026-09-02: hoisted paramsVec+Create above the
prior loop in the 8 Create-inside blocks (noa1_noz_e pair, noz_e withA1 pair, GI1_noz_e pair's
V_Jplus1 blocks) and deleted the 50 dead initializers elsewhere (each verified
reassigned-before-use). Sweeps re-run clean. GPU run 2 (2026-09-02): 6 groups green (noz_e pair joined), then undefined
`EVpre` in the z&e raws — the e-min stage emitted its running worst case as `EV` while the z-stack
reads `EVpre` (the transformer picked the emit name per donor shape: EVpre/EVnext/EV; only the
EV-emitting blocks were wrong). Fixed same day in the 6 affected files (both blocks each:
Raw e/nod1_e/noa1_e/nod1_noa1_e, DC1_e, DC1_GI1_e); statement-aware use-before-def sweep over all
EV-family and lottery names now 0 problems, and per-file ambiguity_pi_e_J reshape-vs-indexing
audited consistent. GPU run 3 (2026-09-02): RAN TO COMPLETION — 206/240 exact zeros, benign ULP
classes (FromPolicy 1e-15..1e-14, T3 retirement-flat-V min-rounding), T1 BIT-EXACT at all four
methods in both d-variants, T2/T3u/T4 exact, warning + error asserts green. Two genuine faults,
both fixed same day: (i) Raw nod1_noz (withA1, u-only) squeeze-form u-min collapsed EV in place
(first prior's sum overwrote the pre-sum EV; priors 2..K then summed garbage) — only file with
the in-place shape, fixed with an EVpreu copy; (ii) FromPolicy GI ~1e-6 in d1+z subcodes: the GI
raws take ALL mins (z, u) and the d2-refinement max at the COARSE a1prime nodes and then
interpolate the worst-case EV (min-then-interp on a1), while FromPolicy did per-prior corner-mix
then min (interp-then-min). DESIGN NOTE, for review: the a1 GI layer deviates from the
interpolate-then-min ruling — moving the mins to the fine grid would force the d2 max to the
fine grid too (max must stay outside the mins), which differs mathematically from the vNM
donor's max-at-coarse-then-interp and would break T1's bit-exact oracle. The a2 lottery (the
riskyasset analog of the standard family's aprime interpolation) IS per-prior, honoring the
ruling where it binds. FromPolicy GI reworked to per-corner min pipelines (component-tracked)
mirroring the raws. GPU run 4 (2026-09-02): FULLY GREEN — 208/240 exact zeros, all 32 nonzeros
at or below 1.8e-14 in the documented benign ULP classes, no errors. COMPLETE, awaiting commit
(toolkit wave + bank together).**

Extends AmbiguityAversion (complete for the standard endogenous states,
`09cceeeb`) to the riskyasset family — and with it, ambiguity over the risky return
distribution itself, which is the classic economic application (ambiguous equity premium,
worst-case portfolio choice). Test-first as always: bank, then toolkit wave, one GPU run,
commit together.

## 1. Scope

- Tiers: **noa1** (the risky asset is the only endogenous state) and **withA1** (safe a1 +
  risky a2). noa1 has no divide-and-conquer and no grid interpolation (nothing to refine —
  structural, same as the vNM and EZ banks), so noa1 gets the base method only; withA1 gets
  all four methods (base/DC/GI/DC+GI).
- Combos: {nod1, d1} x {noz, z} x {noe, e} — all 8 per tier. Note noz+noe is VALID here,
  unlike the standard-asset family: the u shock always exists and can be ambiguous, so
  "ambiguity with no shocks" is no longer vacuous.
- Deferred: semiz (as everywhere in AA), with2A1.
- Raw count: 8 noa1 + 32 withA1 = **40 raws**, plus dispatcher/FromPolicy/setup plumbing.

## 2. Design decisions

1. **u is treated as AMBIGUITY, not as risk** (per review): under ambiguity aversion the
   agent does not know the risky return distribution, so `vfoptions.ambiguity_pi_u` — size
   [N_u, max(n_ambiguity)], age-independent like pi_u itself, no _J form — is MANDATORY
   whenever riskyasset is combined with AmbiguityAversion. The regular pi_u is only the true
   process (agent distribution etc.), exactly parallel to pi_z/pi_e. The mandatory rule is
   the same as the standard-asset tier's: every shock present must have its priors declared
   (ambiguity_pi_z when there is a z, ambiguity_pi_e when there is an e, ambiguity_pi_u
   always — u always exists in riskyasset). A shock the user wants unambiguous is declared
   with replicated priors, as anywhere else in the family.
2. **One `n_ambiguity` governs all three** — the priors are tuples (pi_u_k, pi_z_k, pi_e_k),
   exactly as the standard tier treats (pi_z_k, pi_e_k).
3. **Sequential mins, one per expectation stage, innermost first**: the iid-e worst case, then
   the markov-z worst case, then — after the aprime lottery evaluation — the u worst case.
   (Contrast EZ-riskyasset, which takes ONE joint certainty-equivalent over (u, zprime);
   under multiple priors the u- and z-priors are separate objects, so sequential mins are the
   natural extension of the family's e-then-z convention.)
4. **The aprime lottery is an interpolation, so it is conditional on the prior** (the standing
   ruling): the two-point lower/upper lookup and mix is done per z-prior — with a per-prior
   local copy of aprimeProbs for the skipinterp zeroing, so one prior's zeroing cannot leak to
   the next (the ExpAsset aprimeProbs-mutation bug class) — and the z-min is taken on the
   mixed values at the lottery points. Then each u-prior's pi_u-weighted sum over u, then the
   u-min. The d2 refinement (`max` over the riskyshare) comes after all the mins: the maxmin
   agent maximizes given the worst case.
5. **pi-consistency warning extends to pi_u**: the regular pi_u is what the agent distribution
   uses; warn if it is not one of the u-priors.

## 3. Bank (CoreFHorzRiskyAssetTests/withAmbiguityAversion/)

Mirrors the EZ mirror's placement inside the riskyasset bank; follows the modern
exotic-preference-mirror principle (V/Policy/ValueFnFromPolicy only — no dist, no figures).

```
withAmbiguityAversion/
  CoreFHorzRiskyAssetAmbiguityTests.m
  withAmbiguityAversion_subcodes/
    Noa1_subcodes/    AmbRiskyAsset_{nod1,d1}_{noz,z}_{noe,e}_nosemiz_noa1.m    (8)
    WithA1_subcodes/  AmbRiskyAsset_{nod1,d1}_{noz,z}_{noe,e}_nosemiz_withA1.m  (8)
    CrossTests/       AmbRiskyAsset_CrossTests_{nod1,d1}.m                      (2)
```

Priors: the three baseline z- and e-priors from the standard AA bank recipe, plus three
u-priors by the same recipe (baseline pi_u; 0.9*pi_u+0.1*(worst u); 0.9*pi_u+0.1*uniform).
Every subcode declares ambiguity_pi_u; z/e subcodes add their ambiguity_pi_z/ambiguity_pi_e.

- **noa1 subcodes**: base solve, ValueFnFromPolicy, lowmemory ladder (=1; =2 in z&e). 3-5
  checks each.
- **withA1 subcodes**: the standard AA skeleton — base vs DC, GI vs DC+GI, lowmemory ladders,
  FromPolicy at base and GI. 14 checks (22 in z&e).
- **Cross tests** (per d-variant file):
  - T1 identical priors (u and z; u and e) = vNM — noa1 base, and withA1 at ALL FOUR methods
    (the per-raw acceptance oracle).
  - T2 3-pi-vs-9-pi duplicated priors on pi_u (noz_noe noa1 model — pure return ambiguity).
  - T3u a/b: an unambiguously worse pi_u binds, either slot (noz_noe noa1) — the marquee
    economics: the ambiguity-averse investor behaves as if facing the worst return
    distribution. Exact because aprime is increasing in u and V is increasing in assets.
  - T3z a/b: worse pi_z binds (z noa1).
  - T4 age-varying n_ambiguity via V_Jplus1 — noa1 base and withA1 at all four methods.
  - T5 unambiguous-in-disguise: replicated u-priors with genuine z-priors equals z-only-style
    ambiguity (checks that declaring a shock unambiguous via replicated priors is exact), plus
    the missing-ambiguity_pi_u error assert.
  - pi_u-consistency warning fires/stays silent.

~156 subcode checks + ~90 cross-test checks = **~246 checks**, diary only.

## 4. Toolkit wave

1. **40 raws** from the vNM riskyasset donors (`RiskyAsset/Raw` 8 of 16, `DivideConquer`,
   `GridInterpLayer`, `DivideConquerGridInterpLayer` withA1 8 each), transform per Section 2:
   e-core and z-core per-prior loops as in the standard family, the z-min moved to after the
   per-prior lottery mix, and the pi_u line(s) (`EV=sum(EV1.*pi_u',2)+sum(EV2.*pi_u',2)` and
   the squeeze-form variant) replaced by a per-u-prior loop + min. Site catalogue with hard
   count assertions per file, diff review, as always.
2. **`ValueFnIter_FHorz_AmbAverse_RiskyAsset.m`** dispatcher mirroring
   `ValueFnIter_FHorz_RiskyAsset` (refine_d handling, tier/method routing, UnKron tails),
   called from a riskyasset branch in `ValueFnIter_FHorz_AmbiguityAversion` (which currently
   never sees riskyasset — Case1 errors first; that guard is removed for AmbiguityAversion,
   kept for GulPesendorfer).
3. **`ExogShockSetup_FHorz_AmbiguityAversion`**: riskyasset branch — require ambiguity_pi_u
   (u is ambiguity, not risk), validate/normalize it, extend the warning to pi_u; z/e keep
   their existing mandatory handling.
4. **FromPolicy**: riskyasset branch of `ValueFnFromPolicy_FHorz_AmbiguityAversion` (or a
   sibling subfn mirroring `ValueFnFromPolicy_FHorz_RiskyAsset`), same per-prior
   lookup-then-min at every stage.
5. Semiz/with2A1 remain errors.

## 5. Order of work

Bank first (19 files, uncommitted); toolkit wave; static sweeps; one GPU run of
`CoreFHorzRiskyAssetAmbiguityTests.m`; commit toolkit + bank together when green.
