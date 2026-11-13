import unittest
import ast
from src.meta_agent_madcoderme.utils import evaluate_condition, StateConditionEvaluator

class ConditionEvaluatorTests(unittest.TestCase):
    def setUp(self):
        self.state = {
            "resume_file": "path/to/resume.pdf",
            "parsed_resume": {"skills": ["Python", "Machine Learning"], "experience": 5},
            "suggestions": {"is_approved": True, "count": 3},
            "job_listings": [{"id": "job-1", "title": "AI Engineer"}, {"id": "job-2", "title": "SWE"}],
            "approved_job_analysis": [],
            "preferences": {"min_fit_score": 80, "auto_apply": False}
        }

    def test_null_comparisons(self):
        self.assertTrue(evaluate_condition("state.resume_file != null", self.state))
        self.assertFalse(evaluate_condition("state.resume_file == null", self.state))
        self.assertTrue(evaluate_condition("state.non_existent == null", self.state))

    def test_boolean_attributes(self):
        self.assertTrue(evaluate_condition("state.suggestions.is_approved == true", self.state))
        self.assertFalse(evaluate_condition("state.suggestions.is_approved == false", self.state))
        self.assertTrue(evaluate_condition("state.preferences.auto_apply == false", self.state))

    def test_literal_true_condition(self):
        self.assertTrue(evaluate_condition("true", self.state))
        self.assertTrue(evaluate_condition("True", self.state))

    def test_len_call_evaluations(self):
        self.assertTrue(evaluate_condition("len(state.job_listings) > 0", self.state))
        self.assertTrue(evaluate_condition("len(state.job_listings) == 2", self.state))
        self.assertFalse(evaluate_condition("len(state.approved_job_analysis) > 0", self.state))
        self.assertTrue(evaluate_condition("len(state.approved_job_analysis) == 0", self.state))

    def test_nested_numeric_comparisons(self):
        self.assertTrue(evaluate_condition("state.preferences.min_fit_score >= 80", self.state))
        self.assertTrue(evaluate_condition("state.preferences.min_fit_score <= 80", self.state))
        self.assertFalse(evaluate_condition("state.preferences.min_fit_score < 80", self.state))

if __name__ == "__main__":
    unittest.main()
