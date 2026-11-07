from collections import Counter

from liedet import data


def test_facts_unique_and_answers_differ():
    facts = data.build_facts(seed=0)
    assert len({f.fact_id for f in facts}) == len(facts) >= 100
    assert all(f.true_answer != f.false_answer for f in facts)
    assert len({f.question for f in facts}) == len(facts)


def test_two_by_two_decouples_context_from_label():
    rows = data.build_rows(data.build_facts(0), variants=("answer",))
    c = Counter((r.context, r.label) for r in rows)
    assert c[("truth", 0)] == c[("truth", 1)] == c[("lie", 0)] == c[("lie", 1)]
    # label is determined only by whether the stated answer is the false one
    assert all(r.label == (0 if r.answer_is_true else 1) for r in rows)


def test_followup_rows_carry_followup_turn():
    rows = data.build_rows(data.build_facts(0), variants=("followup",))
    assert all(r.followup_user and r.followup_assistant for r in rows)
    assert all(r.followup_user is None for r in data.build_rows(data.build_facts(0), variants=("answer",)))


def test_split_is_by_fact_disjoint_and_stratified():
    facts = data.build_facts(0)
    tr, ca, te = data.split_by_fact(facts, seed=1)
    assert not (set(tr) & set(ca)) and not (set(tr) & set(te)) and not (set(ca) & set(te))
    assert set(tr) | set(ca) | set(te) == {f.fact_id for f in facts}
    fam = {f.fact_id: f.family for f in facts}
    for part in (tr, ca, te):
        assert {fam[i] for i in part} == {"geography", "chemistry", "science", "arithmetic"}


def test_split_is_deterministic():
    f = data.build_facts(0)
    assert data.split_by_fact(f, 3) == data.split_by_fact(f, 3)
