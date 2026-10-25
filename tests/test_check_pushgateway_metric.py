"""Validate freshness and job matching for the optional metrics gate."""

import unittest

from scripts.check_pushgateway_metric import fresh_chat_metric


class PushgatewayMetricTests(unittest.TestCase):
    def test_requires_metric_and_recent_push_for_the_job(self):
        body = (
            'openwebui_chat_completion_seconds{model="gpt-4o-mini"} 1.2\n'
            'push_time_seconds{instance="",job="openwebui_chat_metrics"} 200\n'
        )
        self.assertTrue(fresh_chat_metric(body, "openwebui_chat_metrics", 100))
        self.assertFalse(fresh_chat_metric(body, "openwebui_chat_metrics", 201))
        self.assertFalse(fresh_chat_metric(body, "other_job", 100))
        self.assertFalse(fresh_chat_metric(body.replace("openwebui_chat_completion_seconds", "other_metric"), "openwebui_chat_metrics", 100))


if __name__ == "__main__":
    unittest.main()
