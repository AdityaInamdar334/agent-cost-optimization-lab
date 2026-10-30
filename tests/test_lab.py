import unittest

from agent_cost_lab.pricing import Rates, Usage, price
from agent_cost_lab.cache_experiment import Cache, prefix_key, run_scenario
from agent_cost_lab.trace_analysis import analyze, percentile
from agent_cost_lab.quality_eval import metrics, paired_bootstrap_recall


class PricingTests(unittest.TestCase):
    def test_four_exclusive_token_buckets(self):
        usage = Usage(regular_input=100000, cache_write=200000,
                      cache_read=300000, output=400000)
        self.assertAlmostEqual(price(usage, Rates()), 7.14)

    def test_invalid_tokens(self):
        with self.assertRaises(ValueError):
            Usage(cache_read=-1)


class CacheTests(unittest.TestCase):
    def test_expiry_and_refresh(self):
        cache = Cache(ttl_seconds=100)
        self.assertFalse(cache.access('x', 0))
        self.assertTrue(cache.access('x', 99))
        self.assertTrue(cache.access('x', 150))
        self.assertFalse(cache.access('x', 250))

    def test_prefix_keys_are_tenant_scoped(self):
        obj = {'instruction': 'one'}
        self.assertNotEqual(prefix_key('tenant-a', 'm', obj), prefix_key('tenant-b', 'm', obj))
        self.assertEqual(prefix_key('tenant-a', 'm', obj), prefix_key('tenant-a', 'm', obj))

    def test_stable_prefix_benefits_only_when_live(self):
        before = run_scenario(False, jobs=32, repository_count=16, interval_seconds=45)
        after = run_scenario(True, jobs=32, repository_count=16, interval_seconds=45)
        self.assertEqual(before['cache_hits'], 0)
        self.assertEqual(after['cache_hits'], 31)
        self.assertLess(after['total_usd'], before['total_usd'])

    def test_no_cache_benefit_after_ttl(self):
        after = run_scenario(True, jobs=3, interval_seconds=301, ttl_seconds=300)
        self.assertEqual(after['cache_hits'], 0)


class AnalyzerTests(unittest.TestCase):
    def test_percentiles(self):
        self.assertEqual(percentile([1, 2, 3], .5), 2)
        self.assertEqual(percentile([1, 3], .5), 2)

    def test_trace_aggregation(self):
        rows = [
            {'job_id': 'a', 'model': 'demo', 'reviewer': 'quality', 'tier': 'full',
             'duration_ms': 200, 'usage': {'regular_input': 100, 'output': 10}},
            {'job_id': 'a', 'model': 'demo', 'reviewer': 'quality', 'tier': 'full',
             'duration_ms': 100, 'usage': {'cache_read': 100, 'output': 10}},
        ]
        out = analyze(rows)
        self.assertEqual(out['jobs'], 1)
        self.assertEqual(out['requests'], 2)
        self.assertEqual(out['token_totals']['cache_read'], 100)


class QualityTests(unittest.TestCase):
    def setUp(self):
        self.cases = [
            {'gold': [{'id': 'a', 'category': 'security'},
                      {'id': 'b', 'category': 'performance'}],
             'baseline': ['a', 'b', 'noise'], 'optimized': ['a']},
            {'gold': [{'id': 'c', 'category': 'security'}],
             'baseline': ['c'], 'optimized': ['c']},
        ]

    def test_precision_recall(self):
        a = metrics(self.cases, 'baseline')
        b = metrics(self.cases, 'optimized')
        self.assertEqual(a['tp'], 3)
        self.assertEqual(a['fp'], 1)
        self.assertEqual(b['fn'], 1)
        self.assertEqual(b['recall_by_category']['performance'], 0.0)

    def test_paired_bootstrap(self):
        out = paired_bootstrap_recall(self.cases, 'baseline', 'optimized', iterations=100)
        self.assertEqual(out['iterations'], 100)
        self.assertEqual(len(out['ci95']), 2)


if __name__ == '__main__':
    unittest.main()

# The following additional tests are discovered as part of a separate class.
class ExtraScenarioTests(unittest.TestCase):
    def test_workflow_fewer_calls(self):
        from agent_cost_lab.workflow_experiment import model_workflow
        self.assertGreater(model_workflow(16)['usd'], model_workflow(10)['usd'])

    def test_prefix_breakpoint_stability(self):
        from agent_cost_lab.prefix_audit import inspect_request
        shared = {'role': 'system', 'content': 'same'}
        a = {'messages': [shared, {'role':'user', 'content':'repo-a'}]}
        b = {'messages': [shared, {'role':'user', 'content':'repo-b'}]}
        stable_a = inspect_request(a, 1, 't', 'm')
        stable_b = inspect_request(b, 1, 't', 'm')
        self.assertEqual(stable_a['stable_fingerprint'], stable_b['stable_fingerprint'])
        self.assertNotEqual(stable_a['full_fingerprint'], stable_b['full_fingerprint'])

class PairedTraceTests(unittest.TestCase):
    def test_matched_costs(self):
        from agent_cost_lab.compare_traces import compare
        a = [{'job_id':'x','usage':{'regular_input':1000}},
             {'job_id':'y','usage':{'regular_input':1000}}]
        b = [{'job_id':'x','usage':{'cache_read':1000}},
             {'job_id':'y','usage':{'cache_read':1000}}]
        report = compare(a, b, iterations=200)
        self.assertEqual(report['paired_jobs'],2)
        self.assertAlmostEqual(report['total_cost_reduction_pct'],90.0)

    def test_drops_unpaired(self):
        from agent_cost_lab.compare_traces import compare
        a = [{'job_id':'x','usage':{'regular_input':1000}},
             {'job_id':'y','usage':{'regular_input':1000}}]
        b = [{'job_id':'x','usage':{'regular_input':500}}]
        report = compare(a, b, iterations=10)
        self.assertEqual(report['unpaired_before_jobs'],1)
