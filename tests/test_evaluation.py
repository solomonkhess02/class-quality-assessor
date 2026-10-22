import pytest

from cqa import evaluation as ev
from cqa.judge.grader import PointGrade, report_lines
from cqa.schemas import Assessment, CriterionResult


def test_pairwise_counts_only_different_labels_and_ties_half():
    true = {"a": "Excellent", "b": "Good", "c": "Good", "d": "Poor"}
    problems = {"a": 0, "b": 1, "c": 0, "d": 3}
    acc, n = ev.pairwise_accuracy(true, problems)
    # pairs: a-b right, a-c tie (0.5), a-d right, b-d right, c-d right; b-c has equal labels and is skipped
    assert n == 5
    assert acc == pytest.approx(4.5 / 5)


def test_pairwise_reversed_order_scores_zero():
    acc, n = ev.pairwise_accuracy({"x": "Excellent", "y": "Poor"}, {"x": 3, "y": 0})
    assert (acc, n) == (0.0, 1)


def test_label_metrics_bias_sign():
    true = {"a": "Excellent", "b": "Good", "c": "Poor"}
    harsh = ev.label_metrics(true, {"a": "Good", "b": "Average", "c": "Poor"})
    assert harsh["bias"] < 0 and harsh["exact"] == 1 and harsh["within_one"] == 3
    assert harsh["mae"] == pytest.approx(2 / 3)
    lenient = ev.label_metrics(true, {"a": "Excellent", "b": "Excellent", "c": "Good"})
    assert lenient["bias"] > 0


def test_criterion_recall_counts_unscored_as_missed():
    expected = {"k1": {"opening": "weak", "closure": "good"}, "k2": {"opening": "weak", "closure": "weak"}}
    scores = {"k1": {"opening": 2, "closure": 3}, "k2": {"opening": None, "closure": 4}}
    r = ev.criterion_recall(expected, scores)
    assert r["all"]["weak"] == (1, 3)  # k1 opening caught; k2 opening unscored; k2 closure scored 4
    assert r["all"]["good"] == (1, 1)
    assert r["opening"]["weak"] == (1, 2)
    assert ev.balanced_accuracy(r["all"]) == pytest.approx((1 / 3 + 1) / 2)


def test_feedback_recall_partial_is_half_and_split_by_source():
    fr = ev.feedback_recall([("T", "yes"), ("T", "partial"), ("V", "no"), ("A", "yes")])
    assert fr["all"] == (2.5, 4)
    assert fr["T"] == (1.5, 2)
    assert fr["V"] == (0.0, 1)


def test_run_agreement_and_score_range():
    runs = [{"a": "Good", "b": "Poor"}, {"a": "Good", "b": "Average"}, {"a": "Good", "b": "Poor"}]
    assert ev.run_agreement(runs) == (1, 2)
    scores = [{"a": {"x": 2, "y": None}}, {"a": {"x": 3, "y": 4}}, {"a": {"x": 2, "y": 4}}]
    assert ev.mean_score_range(scores) == 1.0  # y is unscored in one run and skipped


def test_trivial_baselines():
    true = {"a": "Excellent", "b": "Good", "c": "Average", "d": "Poor"}
    b = ev.trivial_baselines(true)
    assert b['Always "Poor"']["exact"] == 1 and b['Always "Poor"']["pairwise"] == 0.5
    assert b["Random label (expected)"]["exact"] == 1.0


def test_report_lines_carry_findings_not_quotes():
    a = Assessment(
        label="Good",
        weighted_score=3.0,
        confidence=0.9,
        criteria=[
            CriterionResult(
                id="closure_homework",
                score=2,
                observations=["No homework at 00:52:10."],
                improvement="Set one homework task.",
            )
        ],  # fmt: skip
        summary="Lively class.",
    )
    lines = report_lines(a)
    assert lines[0] == "Summary: Lively class."
    assert "closure_homework observation: No homework at 00:52:10." in lines
    assert PointGrade(point=1, verdict="partial").verified is False
