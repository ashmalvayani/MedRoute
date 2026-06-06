"""
Thorough analysis of why self-consistency (3 rollouts + majority vote) barely helps
the routed pipeline compared to plain Qwen3-8B.
"""
import json
import sys
from collections import Counter
from pathlib import Path

# ---------------------------------------------------------------------------
# 1. Load routed self-consistency details
# ---------------------------------------------------------------------------
details_path = Path("result/self_consistency_routed/rollouts_3/"
                    "self_consistency_routed_2026-04-17-17-06-27_details.json")
with open(details_path) as f:
    routed_details = json.load(f)

summary_path = Path("result/self_consistency_routed/rollouts_3/"
                    "self_consistency_routed_2026-04-17-17-06-27.json")
with open(summary_path) as f:
    summary_data = json.load(f)

# First element is the summary
summary = summary_data[0]
print("=" * 70)
print("ROUTED SELF-CONSISTENCY OVERVIEW")
print("=" * 70)
print(f"  Majority-vote accuracy:    {summary['majority_vote_accuracy']:.4f} ({summary['total_correct']}/{summary['total_questions']})")
print(f"  Individual trace accuracy: {summary['individual_trace_accuracy']:.4f}")
print(f"  Num rollouts:              {summary['config']['num_rollouts']}")
print(f"  Temperature:               {summary['config']['temperature']}")
print()

# ---------------------------------------------------------------------------
# 2. Answer diversity analysis
# ---------------------------------------------------------------------------
print("=" * 70)
print("ANALYSIS 1: ANSWER DIVERSITY ACROSS 3 ROLLOUTS")
print("=" * 70)

all_agree = 0       # 3-0 split (unanimous)
two_one_split = 0   # 2-1 split
all_different = 0   # 1-1-1 split
parse_issues = 0

# Track sub-categories
unanimous_correct = 0
unanimous_wrong = 0
split_21_majority_correct = 0
split_21_majority_wrong = 0
all_diff_vote_correct = 0
all_diff_vote_wrong = 0

for q in routed_details:
    rollouts = q['Rollouts']
    gt = q['Ground_truth']
    voted = q['Voted_answer']
    letters = [r['letter'] for r in rollouts]

    unique = set(letters)
    counts = Counter(letters)

    if len(unique) == 1:
        all_agree += 1
        if letters[0] == gt:
            unanimous_correct += 1
        else:
            unanimous_wrong += 1
    elif len(unique) == 2:
        two_one_split += 1
        if voted == gt:
            split_21_majority_correct += 1
        else:
            split_21_majority_wrong += 1
    else:
        all_different += 1
        if voted == gt:
            all_diff_vote_correct += 1
        else:
            all_diff_vote_wrong += 1

total = len(routed_details)
print(f"\nTotal questions: {total}")
print(f"\n  All 3 agree (unanimous):  {all_agree:4d} ({100*all_agree/total:.1f}%)")
print(f"    -> Correct:             {unanimous_correct:4d} ({100*unanimous_correct/total:.1f}%)")
print(f"    -> Wrong:               {unanimous_wrong:4d} ({100*unanimous_wrong/total:.1f}%)")
print(f"\n  2-1 split:                {two_one_split:4d} ({100*two_one_split/total:.1f}%)")
print(f"    -> Majority correct:    {split_21_majority_correct:4d} ({100*split_21_majority_correct/total:.1f}%)")
print(f"    -> Majority wrong:      {split_21_majority_wrong:4d} ({100*split_21_majority_wrong/total:.1f}%)")
print(f"\n  All 3 different:          {all_different:4d} ({100*all_different/total:.1f}%)")
print(f"    -> Vote correct:        {all_diff_vote_correct:4d}")
print(f"    -> Vote wrong:          {all_diff_vote_wrong:4d}")

# ---------------------------------------------------------------------------
# 3. When traces disagree, does majority vote help or hurt?
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("ANALYSIS 2: DOES MAJORITY VOTE FIX OR BREAK THINGS?")
print("=" * 70)

# For disagreement cases (2-1 or 1-1-1), compare majority vote to individual traces
fix_count = 0   # at least one trace wrong, majority correct (net positive)
break_count = 0 # at least one trace correct, majority wrong (net negative)
no_change_good = 0  # all correct anyway (shouldn't happen with disagreement)
no_change_bad = 0   # all wrong, majority wrong too

for q in routed_details:
    rollouts = q['Rollouts']
    gt = q['Ground_truth']
    voted = q['Voted_answer']
    letters = [r['letter'] for r in rollouts]
    unique = set(letters)

    if len(unique) == 1:
        continue  # unanimous, no disagreement

    any_correct = any(l == gt for l in letters)
    all_correct = all(l == gt for l in letters)
    vote_correct = voted == gt

    if vote_correct and not all_correct:
        fix_count += 1
    elif not vote_correct and any_correct:
        break_count += 1
    elif not vote_correct and not any_correct:
        no_change_bad += 1
    else:
        no_change_good += 1

disagree_total = two_one_split + all_different
print(f"\nQuestions with disagreement: {disagree_total}")
print(f"  Vote FIXES  (wrong->right):  {fix_count:4d} ({100*fix_count/disagree_total:.1f}% of disagreements)")
print(f"  Vote BREAKS (right->wrong):  {break_count:4d} ({100*break_count/disagree_total:.1f}% of disagreements)")
print(f"  No change (all wrong):       {no_change_bad:4d} ({100*no_change_bad/disagree_total:.1f}% of disagreements)")
print(f"  No change (all correct):     {no_change_good:4d}")
print(f"\n  Net effect of voting on disagreements: +{fix_count - break_count} questions")

# ---------------------------------------------------------------------------
# 4. Routing trace consistency
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("ANALYSIS 3: ROUTING TRACE CONSISTENCY ACROSS ROLLOUTS")
print("=" * 70)

same_route = 0
diff_route = 0
route_variations = []

for q in routed_details:
    rollouts = q['Rollouts']
    routes = [tuple(r['routing_trace']) for r in rollouts]
    unique_routes = set(routes)
    if len(unique_routes) == 1:
        same_route += 1
    else:
        diff_route += 1
        route_variations.append(len(unique_routes))

print(f"\n  Same routing path all 3 rollouts:  {same_route:4d} ({100*same_route/total:.1f}%)")
print(f"  Different routing paths:           {diff_route:4d} ({100*diff_route/total:.1f}%)")

if route_variations:
    print(f"  When different: avg unique routes = {sum(route_variations)/len(route_variations):.2f}")

# Cross-tabulate: route consistency vs answer consistency
same_route_same_answer = 0
same_route_diff_answer = 0
diff_route_same_answer = 0
diff_route_diff_answer = 0

for q in routed_details:
    rollouts = q['Rollouts']
    routes = [tuple(r['routing_trace']) for r in rollouts]
    letters = [r['letter'] for r in rollouts]
    route_same = len(set(routes)) == 1
    answer_same = len(set(letters)) == 1

    if route_same and answer_same:
        same_route_same_answer += 1
    elif route_same and not answer_same:
        same_route_diff_answer += 1
    elif not route_same and answer_same:
        diff_route_same_answer += 1
    else:
        diff_route_diff_answer += 1

print(f"\n  Cross-tabulation (Route x Answer consistency):")
print(f"  {'':30s} Same Answer   Diff Answer")
print(f"  {'Same Route':30s} {same_route_same_answer:6d}        {same_route_diff_answer:6d}")
print(f"  {'Different Route':30s} {diff_route_same_answer:6d}        {diff_route_diff_answer:6d}")

# When routes differ, what kind of diversity?
print(f"\n  Key insight: Even with temp=0.7, {100*same_route/total:.1f}% of questions get identical routing.")
print(f"  Of {same_route} same-route questions, {same_route_diff_answer} ({100*same_route_diff_answer/same_route:.1f}% if same_route > 0) still get different answers.")

# ---------------------------------------------------------------------------
# 5. Error pattern analysis for questions wrong after majority vote
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("ANALYSIS 4: ERROR PATTERN ANALYSIS (Questions wrong after majority vote)")
print("=" * 70)

wrong_unanimous_same = 0      # All 3 wrong, same wrong answer
wrong_21_overrule = 0         # 2 wrong same answer, 1 right -> majority overrules correct
wrong_21_all_wrong = 0        # 2-1 split but all answers wrong (none match GT)
wrong_all_diff = 0            # All 3 different, none or minority correct
wrong_other = 0

for q in routed_details:
    voted = q['Voted_answer']
    gt = q['Ground_truth']
    if voted == gt:
        continue  # skip correct ones

    letters = [r['letter'] for r in q['Rollouts']]
    counts = Counter(letters)
    unique = set(letters)
    any_correct = any(l == gt for l in letters)

    if len(unique) == 1:
        wrong_unanimous_same += 1
    elif len(unique) == 2:
        if any_correct:
            wrong_21_overrule += 1
        else:
            wrong_21_all_wrong += 1
    elif len(unique) == 3:
        wrong_all_diff += 1
    else:
        wrong_other += 1

total_wrong = total - summary['total_correct']
print(f"\nTotal wrong after majority vote: {total_wrong}")
print(f"\n  All 3 unanimous WRONG (same answer):     {wrong_unanimous_same:4d} ({100*wrong_unanimous_same/total_wrong:.1f}%)")
print(f"  2-1 split, majority overrules correct:   {wrong_21_overrule:4d} ({100*wrong_21_overrule/total_wrong:.1f}%)")
print(f"  2-1 split, all wrong answers:            {wrong_21_all_wrong:4d} ({100*wrong_21_all_wrong/total_wrong:.1f}%)")
print(f"  All 3 different wrong answers:           {wrong_all_diff:4d} ({100*wrong_all_diff/total_wrong:.1f}%)")

print(f"\n  KEY: {100*wrong_unanimous_same/total_wrong:.1f}% of errors are unanimous wrong - self-consistency CANNOT fix these.")
print(f"       {100*wrong_21_overrule/total_wrong:.1f}% of errors are cases where voting HURTS (overrules correct minority).")

# ---------------------------------------------------------------------------
# 6. Entropy / diversity metrics
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("ANALYSIS 5: ANSWER DIVERSITY METRICS")
print("=" * 70)

import math

routed_entropies = []
for q in routed_details:
    letters = [r['letter'] for r in q['Rollouts']]
    counts = Counter(letters)
    n = len(letters)
    entropy = -sum((c/n) * math.log2(c/n) for c in counts.values())
    routed_entropies.append(entropy)

mean_entropy = sum(routed_entropies) / len(routed_entropies)
zero_entropy = sum(1 for e in routed_entropies if e == 0)
print(f"\n  Mean answer entropy across questions: {mean_entropy:.4f} (max possible = {math.log2(3):.4f})")
print(f"  Questions with zero entropy (unanimous): {zero_entropy} ({100*zero_entropy/total:.1f}%)")

# ---------------------------------------------------------------------------
# 7. Comparison: What would single-trace accuracy be?
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("ANALYSIS 6: SINGLE-TRACE vs MAJORITY VOTE COMPARISON")
print("=" * 70)

# Compute accuracy for each individual rollout index
for ri in range(3):
    correct = 0
    for q in routed_details:
        rollouts_by_ri = {r['ri']: r for r in q['Rollouts']}
        if ri in rollouts_by_ri and rollouts_by_ri[ri]['letter'] == q['Ground_truth']:
            correct += 1
    print(f"  Rollout {ri} accuracy: {correct}/{total} = {100*correct/total:.2f}%")

print(f"  Majority vote accuracy: {summary['total_correct']}/{total} = {100*summary['total_correct']/total:.2f}%")
print(f"  Gain from voting: +{100*(summary['total_correct']/total - summary['individual_trace_accuracy']):.2f}%")

# ---------------------------------------------------------------------------
# 8. Most common routing paths
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("ANALYSIS 7: MOST COMMON ROUTING PATHS")
print("=" * 70)

all_routes = []
for q in routed_details:
    for r in q['Rollouts']:
        all_routes.append(tuple(r['routing_trace']))

route_counts = Counter(all_routes)
print(f"\n  Total unique routing paths: {len(route_counts)}")
print(f"\n  Top 10 most common paths:")
for path, count in route_counts.most_common(10):
    print(f"    {' -> '.join(path):60s} {count:4d} ({100*count/len(all_routes):.1f}%)")

# ---------------------------------------------------------------------------
# 9. Specialist-level analysis: which specialists appear?
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("ANALYSIS 8: SPECIALIST FREQUENCY IN ROUTING")
print("=" * 70)

specialist_counts = Counter()
for q in routed_details:
    for r in q['Rollouts']:
        for s in r['routing_trace']:
            if s != 'DecisionMaker':
                specialist_counts[s] += 1

print(f"\n  Specialist appearances (excluding DecisionMaker):")
for spec, count in specialist_counts.most_common():
    print(f"    {spec:30s} {count:5d} ({100*count/len(all_routes):.1f}% of traces)")

# ---------------------------------------------------------------------------
# 10. Theoretical upper bound
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("ANALYSIS 9: THEORETICAL UPPER BOUND (Oracle voting)")
print("=" * 70)

oracle_correct = 0
for q in routed_details:
    letters = [r['letter'] for r in q['Rollouts']]
    if q['Ground_truth'] in letters:
        oracle_correct += 1

print(f"\n  Oracle accuracy (any trace correct): {oracle_correct}/{total} = {100*oracle_correct/total:.2f}%")
print(f"  Majority vote accuracy:             {summary['total_correct']}/{total} = {100*summary['total_correct']/total:.2f}%")
print(f"  Gap (room for improvement):         {100*(oracle_correct - summary['total_correct'])/total:.2f}%")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("SUMMARY: WHY SELF-CONSISTENCY BARELY HELPS THE ROUTED PIPELINE")
print("=" * 70)
print(f"""
Key findings:

1. EXTREME LACK OF DIVERSITY: {100*all_agree/total:.1f}% of questions get the SAME answer
   across all 3 rollouts. Self-consistency is useless for these questions.

2. ROUTING IS NEAR-DETERMINISTIC: {100*same_route/total:.1f}% of questions follow the
   exact same routing path across all 3 rollouts. The GNN router produces
   very similar routing decisions despite temperature=0.7, because the routing
   is based on learned graph structure, not sampling randomness.

3. WHEN ROUTES ARE THE SAME, ANSWERS ARE ALMOST ALWAYS THE SAME:
   Of {same_route} same-route questions, only {same_route_diff_answer} ({100*same_route_diff_answer/max(same_route,1):.1f}%) produce
   different answers. The specialist pipeline is also nearly deterministic.

4. MAJORITY VOTE HAS MINIMAL NET EFFECT: Among the {disagree_total} disagreement
   cases, voting fixes {fix_count} but breaks {break_count}, for a net gain of only
   {fix_count - break_count} questions ({100*(fix_count-break_count)/total:.2f}% absolute).

5. ERRORS ARE DOMINATED BY UNANIMOUS WRONG ANSWERS: {100*wrong_unanimous_same/total_wrong:.1f}%
   of all errors come from all 3 traces agreeing on the WRONG answer.
   Self-consistency fundamentally cannot help here.

6. THE ORACLE GAP IS SMALL: Even if we could perfectly pick the correct trace
   when available, we'd only reach {100*oracle_correct/total:.2f}% (vs {100*summary['total_correct']/total:.2f}% actual).
   The room for improvement via better voting is only {100*(oracle_correct-summary['total_correct'])/total:.2f}%.

ROOT CAUSE: The GNN router + specialist pipeline is essentially deterministic
for most questions. Temperature sampling doesn't create meaningful diversity
because the routing decisions dominate the output, and the router is not
stochastic enough. For plain LLM self-consistency, temperature directly
injects diversity into the generation process. For the routed pipeline,
temperature only affects within-specialist generation but the routing
structure constrains the answers to be very similar.

RECOMMENDATION: To make self-consistency effective for the routed pipeline,
one would need to introduce stochasticity at the ROUTING level (e.g.,
sampling from the routing probability distribution rather than argmax),
not just at the generation level.
""")
