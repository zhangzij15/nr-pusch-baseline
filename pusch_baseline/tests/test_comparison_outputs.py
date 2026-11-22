"""Tests for the three-codebook comparison artifacts."""

import csv
from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from pusch_baseline.comparison_outputs import _comparison_figure, write_comparison
from pusch_baseline.outputs import _render_plots
from run_uplink import _report


class ComparisonOutputsTests(unittest.TestCase):
    def setUp(self):
        self.runs = {}
        for name, factor in (("full", 3), ("partial_noncoherent", 2), ("noncoherent", 1)):
            self.runs[name] = {
                "config": {"pusch": {"rank": 1}},
                "metadata": {"codebook": name},
                "points": [{
                    "snr_db": snr, "bler": bler,
                    "thp_including_srs_mbps": factor * 4,
                    "thp_wall_clock_mbps": factor * 2,
                    "srs_grants": [{"slot_index": 1, "rank": 1, "tpmi": factor}],
                } for snr, bler in ((0.0, 0.0), (-8.0, 0.25))],
            }

    def test_figures_plot_three_ordered_curves_with_exact_zero_bler(self):
        thp_lines = _comparison_figure(self.runs, "thp", "thp_wall_clock_mbps").axes[0].lines
        self.assertEqual(len(thp_lines), 3)
        self.assertEqual(list(thp_lines[0].get_xdata()), [-8.0, 0.0])
        self.assertEqual([list(line.get_ydata()) for line in thp_lines], [[6, 6], [4, 4], [2, 2]])
        bler_lines = _comparison_figure(self.runs, "bler", "bler").axes[0].lines
        self.assertEqual(len(bler_lines), 3)
        self.assertEqual(list(bler_lines[0].get_ydata()), [0.25, 0.0])
        self.assertEqual(len({line.get_marker() for line in bler_lines}), 3)

    def test_write_creates_pngs_and_preserves_points_and_grants(self):
        with TemporaryDirectory() as temp:
            directory = write_comparison(Path(temp) / "comparison", self.runs)
            self.assertEqual(
                {path.name for path in directory.iterdir()},
                {"THP_vs_SNR.png", "BLER_vs_SNR.png", "results.csv", "summary.json"},
            )
            for name in ("THP_vs_SNR.png", "BLER_vs_SNR.png"):
                self.assertEqual((directory / name).read_bytes()[:8], b"\x89PNG\r\n\x1a\n")
            summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["runs"], self.runs)
            self.assertEqual(summary["throughput_field"], "thp_wall_clock_mbps")
            with (directory / "results.csv").open(encoding="utf-8", newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 6)
            self.assertEqual(rows[0]["codebook"], "full")
            self.assertEqual(json.loads(rows[0]["srs_grants"]), self.runs["full"]["points"][0]["srs_grants"])

    def test_non_tdd_comparison_uses_configured_uplink_duty_cycle(self):
        for run in self.runs.values():
            for point in run["points"]:
                del point["thp_wall_clock_mbps"]
                point["ul_duty_cycle"] = 0.5
                point["thp_duty_cycle_mbps"] = point["thp_including_srs_mbps"] * 0.5

        with TemporaryDirectory() as temp:
            directory = write_comparison(Path(temp) / "comparison", self.runs)
            summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["throughput_field"], "thp_duty_cycle_mbps")

    def _target_points(self):
        points = [
            {"snr_db": -8.0, "bler": 0.3, "target_status": "above_target", "thp_at_target_mbps": None},
            {"snr_db": -4.0, "bler": 0.1, "target_status": "at_target", "thp_at_target_mbps": 7.0},
            {"snr_db": 0.0, "bler": 0.2, "target_status": "calibration_not_converged",
             "thp_at_target_mbps": None},
            {"snr_db": 4.0, "bler": 0.0, "target_status": "at_mcs_ceiling", "thp_at_target_mbps": None},
        ]
        return [{"target_bler": 0.1, "bler_tolerance": 0.05, "num_tb": 10,
                 "calibration_status": "converged", "tb_errors": int(point["bler"] * 10),
                 "thp_wall_clock_mbps": 8.0 + index, "thp_scheduled_mbps": 12.0,
                 "throughput_ci95_mbps": [7.0 + index, 9.0 + index],
                 "ul_duty_cycle": 1.0, **point} for index, point in enumerate(points)]

    def test_target_throughput_and_bler_keep_all_measured_points(self):
        for run in self.runs.values():
            run["points"] = self._target_points()

        axes = _comparison_figure(self.runs, "thp", "thp_wall_clock_mbps").axes[0]
        thp = axes.lines
        self.assertEqual(len(thp), 3)
        self.assertEqual(list(thp[0].get_ydata()), [8.0, 9.0, 10.0, 11.0])
        self.assertIn("BLER target met", [text.get_text() for text in axes.get_legend().texts])
        bler = _comparison_figure(self.runs, "bler", "bler").axes[0].lines
        self.assertEqual(list(bler[0].get_ydata()), [0.3, 0.1, 0.2, 0.0])
        self.assertEqual(list(bler[-1].get_ydata()), [0.1, 0.1])

        with TemporaryDirectory() as temp:
            directory = write_comparison(Path(temp) / "comparison", self.runs)
            summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["throughput_field"], "thp_wall_clock_mbps")
            self.assertIsNone(summary["runs"]["full"]["points"][0]["thp_at_target_mbps"])

    def test_single_target_plots_and_cli_report(self):
        from matplotlib.backends.backend_agg import FigureCanvasAgg

        points = self._target_points()
        observed_axes = []

        def collect(canvas, _stream):
            observed_axes.append(canvas.figure.axes[0])

        with patch.object(FigureCanvasAgg, "print_png", new=collect):
            _render_plots(points, "both")

        bler, thp = observed_axes
        self.assertEqual(list(bler.lines[0].get_ydata()), [0.3, 0.1, 0.2, 0.0])
        self.assertEqual(list(bler.lines[-1].get_ydata()), [0.1, 0.1])
        self.assertEqual(len(thp.lines), 1)
        self.assertEqual(list(thp.lines[0].get_ydata()), [8.0, 9.0, 10.0, 11.0])
        self.assertEqual(thp.collections[0].get_segments()[0].tolist(), [[-8.0, 7.0], [-8.0, 9.0]])
        self.assertEqual(thp.collections[1].get_offsets().tolist(), [[-4.0, 9.0]])

        printed = StringIO()
        from pusch_baseline.__main__ import _report as report_module

        with redirect_stdout(printed):
            _report(points[0], name="full")
            _report(points[1], name="full")
            report_module(points[0])
        outside, valid, module_outside = printed.getvalue().splitlines()
        self.assertIn("THP 8.000 Mbps (above_target)", outside)
        self.assertIn("THP 9.000 Mbps (at_target)", valid)
        self.assertIn("THP 8.000 Mbps (above_target)", module_outside)

    def test_throughput_differences_resample_the_same_periods(self):
        for offset, run in enumerate(self.runs.values()):
            for point in run["points"]:
                point["slot_seconds"] = 0.001
                point["thp_wall_clock_mbps"] = 2.0 + offset
                point["throughput_srs_clusters"] = [
                    {"srs_slot_index": 0, "num_slots": 4, "delivered_payload_bits": 4000 * offset},
                    {"srs_slot_index": 4, "num_slots": 2, "delivered_payload_bits": 12000 + 2000 * offset},
                ]
        with TemporaryDirectory() as temp:
            directory = write_comparison(Path(temp) / "comparison", self.runs)
            summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
            paired = summary["paired_throughput_differences"]
            self.assertEqual(len(paired), 6)
            first = paired[0]
            self.assertEqual((first["codebook_a"], first["codebook_b"]), ("full", "partial_noncoherent"))
            self.assertEqual(first["difference_mbps"], -1.0)
            self.assertEqual(first["difference_ci95_mbps"], [-1.0, -1.0])
            self.assertEqual(first["num_srs_clusters"], 2)
            with (directory / "throughput_differences.csv").open(encoding="utf-8", newline="") as stream:
                self.assertEqual(len(list(csv.DictReader(stream))), 6)


if __name__ == "__main__":
    unittest.main()
