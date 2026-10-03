import json
from pathlib import Path

from src.agent import (
    ask_agent,
    detect_intent,
)
from src.rag import retrieve_context


# ============================================================
# Paths
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

TEST_FILE = (
    BASE_DIR
    / "tests"
    / "evaluation_questions.json"
)

REPORT_FILE = (
    BASE_DIR
    / "tests"
    / "evaluation_report.json"
)


# ============================================================
# Load Test Dataset
# ============================================================

def load_tests():

    if not TEST_FILE.exists():
        raise FileNotFoundError(
            f"Evaluation dataset not found: {TEST_FILE}"
        )

    with open(
        TEST_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(file)


# ============================================================
# Utility
# ============================================================

def percentage(
    passed: int,
    total: int
):

    if total == 0:
        return 0.0

    return round(
        (passed / total) * 100,
        2
    )


# ============================================================
# Intent Accuracy
# ============================================================

def evaluate_intent_accuracy(
    tests
):

    passed = 0
    results = []

    print(
        "\n================================"
    )

    print(
        "Intent Routing Evaluation"
    )

    print(
        "================================"
    )

    for test in tests:

        query = test[
            "query"
        ]

        expected = test[
            "expected_intent"
        ]

        predicted = (
            detect_intent(
                query
            ).value
        )

        success = (
            predicted == expected
        )

        if success:
            passed += 1

        results.append(
            {
                "query": query,
                "expected": expected,
                "predicted": predicted,
                "passed": success,
            }
        )

        status = (
            "PASS"
            if success
            else "FAIL"
        )

        print(
            f"[{status}] {query}"
        )

        print(
            f"Expected: {expected}"
        )

        print(
            f"Predicted: {predicted}\n"
        )

    total = len(
        tests
    )

    return {
        "passed": passed,
        "total": total,
        "score": percentage(
            passed,
            total
        ),
        "results": results,
    }


# ============================================================
# Retrieval Hit Rate
# ============================================================

def evaluate_retrieval(
    tests
):

    passed = 0
    results = []

    print(
        "\n================================"
    )

    print(
        "Retrieval Evaluation"
    )

    print(
        "================================"
    )

    for test in tests:

        query = test[
            "query"
        ]

        expected_source = test[
            "expected_source"
        ]

        result = retrieve_context(
            query=query,
            k=4,
        )

        sources = result[
            "sources"
        ]

        success = any(
            expected_source.lower()
            in source.lower()
            for source in sources
        )

        if success:
            passed += 1

        results.append(
            {
                "query": query,
                "expected_source": expected_source,
                "retrieved_sources": sources,
                "passed": success,
            }
        )

        status = (
            "PASS"
            if success
            else "FAIL"
        )

        print(
            f"[{status}] {query}"
        )

        print(
            f"Expected source: {expected_source}"
        )

        print(
            f"Retrieved: {sources}\n"
        )

    total = len(
        tests
    )

    return {
        "passed": passed,
        "total": total,
        "score": percentage(
            passed,
            total
        ),
        "results": results,
    }


# ============================================================
# Fact Accuracy
# ============================================================

def evaluate_fact_accuracy(
    tests
):

    passed = 0
    results = []

    print(
        "\n================================"
    )

    print(
        "Fact Accuracy Evaluation"
    )

    print(
        "================================"
    )

    for test in tests:

        query = test[
            "query"
        ]

        expected_text = (
            test["expected_text"]
            .lower()
        )

        try:

            result = ask_agent(
                query
            )

            answer = (
                result["answer"]
                .lower()
            )

            success = (
                expected_text
                in answer
            )

        except Exception as error:

            answer = (
                f"ERROR: {error}"
            )

            success = False

        if success:
            passed += 1

        results.append(
            {
                "query": query,
                "expected_text": expected_text,
                "answer": answer,
                "passed": success,
            }
        )

        status = (
            "PASS"
            if success
            else "FAIL"
        )

        print(
            f"[{status}] {query}"
        )

        print(
            f"Expected text: {expected_text}"
        )

        print(
            f"Answer: {answer}\n"
        )

    total = len(
        tests
    )

    return {
        "passed": passed,
        "total": total,
        "score": percentage(
            passed,
            total
        ),
        "results": results,
    }


# ============================================================
# Hallucination Refusal Rate
# ============================================================

def evaluate_hallucination_refusal(
    tests
):

    passed = 0
    results = []

    refusal_phrases = [
        "does not contain this information",
        "does not contain",
        "not available",
        "not provided",
        "insufficient information",
    ]

    print(
        "\n================================"
    )

    print(
        "Hallucination Evaluation"
    )

    print(
        "================================"
    )

    for test in tests:

        query = test[
            "query"
        ]

        try:

            result = ask_agent(
                query
            )

            answer = (
                result["answer"]
                .lower()
            )

            success = any(
                phrase in answer
                for phrase
                in refusal_phrases
            )

        except Exception as error:

            answer = (
                f"ERROR: {error}"
            )

            success = False

        if success:
            passed += 1

        results.append(
            {
                "query": query,
                "answer": answer,
                "passed": success,
            }
        )

        status = (
            "PASS"
            if success
            else "FAIL"
        )

        print(
            f"[{status}] {query}"
        )

        print(
            f"Answer: {answer}\n"
        )

    total = len(
        tests
    )

    return {
        "passed": passed,
        "total": total,
        "score": percentage(
            passed,
            total
        ),
        "results": results,
    }


# ============================================================
# Structured Output Validity
# ============================================================

def evaluate_structured_output(
    tests
):

    passed = 0
    results = []

    print(
        "\n================================"
    )

    print(
        "Structured Output Evaluation"
    )

    print(
        "================================"
    )

    required_keys = {

        "FACT_LOOKUP": {
            "answer",
            "confidence",
            "sources",
        },

        "CAMPAIGN_GENERATION": {
            "campaign_name",
            "objective",
            "target_audience",
            "key_message",
            "recommended_channels",
            "campaign_ideas",
            "reasoning",
            "general_recommendations",
            "confidence",
            "sources",
        },

        "SEGMENT_ANALYSIS": {
            "segment_name",
            "characteristics",
            "pain_points",
            "digital_behavior",
            "buying_motivation",
            "marketing_recommendations",
            "confidence",
            "sources",
        },
    }

    for test in tests:

        query = test[
            "query"
        ]

        expected_intent = test[
            "expected_intent"
        ]

        try:

            result = ask_agent(
                query
            )

            structured = result[
                "structured"
            ]

            actual_intent = result[
                "intent"
            ]

            keys = set(
                structured.keys()
            )

            expected_keys = (
                required_keys[
                    expected_intent
                ]
            )

            success = (
                actual_intent
                == expected_intent
                and
                expected_keys.issubset(
                    keys
                )
            )

        except Exception as error:

            structured = {
                "error": str(error)
            }

            actual_intent = "ERROR"

            success = False

        if success:
            passed += 1

        results.append(
            {
                "query": query,
                "expected_intent": expected_intent,
                "actual_intent": actual_intent,
                "structured": structured,
                "passed": success,
            }
        )

        status = (
            "PASS"
            if success
            else "FAIL"
        )

        print(
            f"[{status}] {query}"
        )

        print(
            f"Intent: {actual_intent}\n"
        )

    total = len(
        tests
    )

    return {
        "passed": passed,
        "total": total,
        "score": percentage(
            passed,
            total
        ),
        "results": results,
    }


# ============================================================
# Overall Evaluation
# ============================================================

def run_evaluation():

    tests = load_tests()

    intent_result = (
        evaluate_intent_accuracy(
            tests[
                "intent_tests"
            ]
        )
    )

    retrieval_result = (
        evaluate_retrieval(
            tests[
                "retrieval_tests"
            ]
        )
    )

    fact_result = (
        evaluate_fact_accuracy(
            tests[
                "fact_tests"
            ]
        )
    )

    hallucination_result = (
        evaluate_hallucination_refusal(
            tests[
                "hallucination_tests"
            ]
        )
    )

    structured_result = (
        evaluate_structured_output(
            tests[
                "structured_output_tests"
            ]
        )
    )

    metrics = {

        "intent_accuracy": (
            intent_result["score"]
        ),

        "retrieval_hit_rate": (
            retrieval_result["score"]
        ),

        "fact_accuracy": (
            fact_result["score"]
        ),

        "hallucination_refusal_rate": (
            hallucination_result[
                "score"
            ]
        ),

        "structured_output_validity": (
            structured_result[
                "score"
            ]
        ),
    }

    overall_score = round(
        sum(
            metrics.values()
        )
        / len(metrics),
        2
    )

    report = {
        "metrics": metrics,
        "overall_score": overall_score,
        "details": {
            "intent": intent_result,
            "retrieval": retrieval_result,
            "fact_accuracy": fact_result,
            "hallucination": hallucination_result,
            "structured_output": structured_result,
        },
    }

    # Save report
    REPORT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        REPORT_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            report,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print(
        "\n================================"
    )

    print(
        "MARKETMIND EVALUATION REPORT"
    )

    print(
        "================================"
    )

    print(
        "Intent Accuracy: "
        f"{metrics['intent_accuracy']}%"
    )

    print(
        "Retrieval Hit Rate: "
        f"{metrics['retrieval_hit_rate']}%"
    )

    print(
        "Fact Accuracy: "
        f"{metrics['fact_accuracy']}%"
    )

    print(
        "Hallucination Refusal Rate: "
        f"{metrics['hallucination_refusal_rate']}%"
    )

    print(
        "Structured Output Validity: "
        f"{metrics['structured_output_validity']}%"
    )

    print(
        "--------------------------------"
    )

    print(
        f"Overall Score: {overall_score}%"
    )

    print(
        "================================"
    )

    print(
        "\nDetailed report saved to:"
    )

    print(
        REPORT_FILE
    )


# ============================================================
# Run
# ============================================================

if __name__ == "__main__":
    run_evaluation()