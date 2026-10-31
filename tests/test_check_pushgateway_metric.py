"""Validate freshness and job matching for the optional metrics gate."""

import unittest

from scripts.check_pushgateway_metric import fresh_chat_metric


class PushgatewayMetricTests(unittest.TestCase):
    def test_requires_metric_and_recent_push_for_the_job(self):
        body = (
            'openwebui_chat_completion_seconds{instance="",job="openwebui_chat_metrics",model="gpt-4o-mini"} 1.2\n'
            'push_time_seconds{instance="",job="openwebui_chat_metrics"} 200\n'
        )
        self.assertTrue(fresh_chat_metric(body, "openwebui_chat_metrics", 100))
        self.assertFalse(fresh_chat_metric(body, "openwebui_chat_metrics", 201))
        self.assertFalse(fresh_chat_metric(body, "other_job", 100))
        self.assertFalse(fresh_chat_metric(body.replace("openwebui_chat_completion_seconds", "other_metric"), "openwebui_chat_metrics", 100))

    def test_rejects_unrelated_job_instance_and_group(self):
        push = 'push_time_seconds{job="target",instance="a",cluster="one"} 200\n'
        for labels in (
            'job="other",instance="a",cluster="one"',
            'job="target",instance="b",cluster="one"',
            'job="target",instance="a",cluster="two"',
            'job="target",instance="a"',
            'model="test"',
        ):
            with self.subTest(labels=labels):
                body = f"openwebui_chat_completion_seconds{{{labels}}} 1.2\n" + push
                self.assertFalse(fresh_chat_metric(body, "target", 100))

    def test_multiple_groups_and_model_matching(self):
        body = (
            'openwebui_chat_completion_seconds{model="test",job="target",instance="a",cluster="one"} 1.2\n'
            'push_time_seconds{instance="a",cluster="one",job="target"} 200\n'
            'push_time_seconds{instance="b",cluster="one",job="target"} 50\n'
        )
        for exposition in (body, "\n".join(reversed(body.splitlines()))):
            self.assertTrue(fresh_chat_metric(exposition, "target", 100, "test"))
            self.assertFalse(fresh_chat_metric(exposition, "target", 100, "other"))

    def test_rejects_invalid_or_nonfinite_samples(self):
        for value in ("NaN", "+Inf", "-Inf", "invalid", "-1"):
            body = (
                f'openwebui_chat_completion_seconds{{job="target",instance=""}} {value}\n'
                'push_time_seconds{job="target",instance=""} 200\n'
            )
            self.assertFalse(fresh_chat_metric(body, "target", 100))

    def test_escaped_label_values(self):
        body = (
            'openwebui_chat_completion_seconds{job="target",instance="",model="a\\\\b"} 1\n'
            'push_time_seconds{job="target",instance=""} 200\n'
        )
        self.assertTrue(fresh_chat_metric(body, "target", 100, "a\\b"))


if __name__ == "__main__":
    unittest.main()
