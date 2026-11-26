"""End-to-end regression checks for the native Sionna PUSCH link."""

from dataclasses import replace
from pathlib import Path
import unittest
from unittest.mock import Mock

from pusch_baseline import SimulationConfig, load_config, simulate


CONFIGS = Path(__file__).resolve().parents[1] / "configs"


class SimulationTests(unittest.TestCase):
    def test_awgn_curve_counts_batches_and_throughput(self):
        config = load_config(CONFIGS / "awgn_siso.json")
        config = replace(
            config,
            sweep=replace(
                config.sweep,
                snr_db=(-30.0, 30.0),
                num_tb_per_snr=7,
                batch_size=3,
                seed=144,
                target_block_errors=3,
                min_tb_per_snr=4,
                ul_duty_cycle=0.4,
            ),
        )
        observations = []
        result = simulate(config, on_point=observations.append)
        low, high = result["points"]
        self.assertEqual(observations, result["points"])
        self.assertEqual((low["snr_db"], high["snr_db"]), (-30.0, 30.0))
        self.assertEqual((low["num_tb"], low["tb_errors"]), (6, 6))
        self.assertEqual(low["stop_reason"], "target_block_errors")
        self.assertEqual(low["bler"], 1.0)
        self.assertEqual(low["thp_scheduled_mbps"], 0.0)
        self.assertEqual((high["num_tb"], high["tb_errors"]), (7, 0))
        self.assertEqual(high["stop_reason"], "max_tb")
        self.assertEqual(high["bler"], 0.0)
        self.assertGreater(high["bler_zero_error_upper95"], 0.0)
        self.assertEqual(high["slot_seconds"], 0.0005)
        peak = high["tb_size_bits"] / (0.0005 * 1e6)
        self.assertAlmostEqual(high["thp_scheduled_mbps"], peak)
        self.assertAlmostEqual(high["thp_duty_cycle_mbps"], 0.4 * peak)

    def test_rank_two_tdl_with_ls_and_short_symbol_grant(self):
        config = SimulationConfig()
        config = replace(
            config,
            carrier=replace(config.carrier, n_size_grid=12),
            pusch=replace(
                config.pusch,
                num_antenna_ports=2,
                rank=2,
                mcs_index=6,
                symbol_allocation=(0, 10),
            ),
            channel=replace(
                config.channel,
                kind="tdl",
                profile="A",
                num_rx_ant=4,
                estimator="ls",
                delay_spread_ns=30.0,
            ),
            sweep=replace(
                config.sweep, snr_db=(30.0,), num_tb_per_snr=3, batch_size=3, seed=144
            ),
        )
        result = simulate(config)
        point = result["points"][0]
        grant = result["metadata"]["pusch"]
        self.assertEqual(result["metadata"]["num_rx_ant"], 4)
        self.assertEqual(grant["allocated_prbs"], 12)
        self.assertEqual(grant["dmrs_ports"], [0, 1])
        self.assertEqual(grant["coded_bits"], 2 * 2 * 12 * 12 * 8)
        self.assertEqual((point["num_tb"], point["tb_errors"]), (3, 0))
        self.assertEqual(point["tb_size_bits"], grant["tb_size_bits"])
        self.assertAlmostEqual(
            point["thp_scheduled_mbps"], point["tb_size_bits"] / (0.0005 * 1e6)
        )

    def test_seed_replays_an_awgn_waterfall_point(self):
        config = load_config(CONFIGS / "awgn_siso.json")
        config = replace(
            config,
            sweep=replace(
                config.sweep, snr_db=(1.0,), num_tb_per_snr=6, batch_size=3, seed=142
            ),
        )
        first = simulate(config)
        second = simulate(config)
        self.assertEqual(first["points"], second["points"])

    def test_external_channel_uses_configured_power_for_grid_and_perfect_csi(self):
        import numpy as np
        import tensorflow as tf

        from pusch_baseline.config import AdaptationSettings
        from pusch_baseline.link import PuschLink

        base = SimulationConfig()
        config = replace(
            base,
            carrier=replace(base.carrier, n_size_grid=12),
            pusch=replace(base.pusch, num_antenna_ports=4, rank=1, mcs_index=5, precoding="codebook"),
            channel=replace(base.channel, kind="custom", num_rx_ant=4, estimator="perfect"),
        )
        cases = [(None, tpmi) for tpmi in range(28)]
        cases.extend((AdaptationSettings(tx_power_mode="per_port"), tpmi) for tpmi in (0, 4, 12))
        for adaptation, tpmi in cases:
            with self.subTest(adaptation=adaptation, tpmi=tpmi):
                cfg = replace(config, pusch=replace(config.pusch, tpmi=tpmi), adaptation=adaptation)
                link = PuschLink(cfg, external_channel=True)
                self.assertEqual(link.grant_metadata["qam_bits_per_symbol"], 2)
                grid = link.resource_grid
                h = tf.broadcast_to(
                    tf.reshape(tf.eye(4, dtype=tf.complex64), (1, 1, 4, 1, 4, 1, 1)),
                    (1, 1, 4, 1, 4, grid.num_ofdm_symbols, grid.fft_size),
                )
                link._apply_ofdm = Mock(wraps=link._apply_ofdm)
                link._receiver = Mock(wraps=link._receiver)
                noise = link.noise_variance(45)
                self.assertEqual(link.count_block_errors(1, noise, channel_response=h), 0)
                x, applied_channel, applied_noise = link._apply_ofdm.call_args.args
                self.assertIs(applied_channel, h)
                self.assertIs(applied_noise, noise)
                data_mask = grid.pilot_pattern.mask.numpy()[0, 0] == 0
                per_port_power = (abs(x.numpy()[0, 0][:, data_mask]) ** 2).mean(axis=-1)
                active_ports = 1 if tpmi < 4 else 2 if tpmi < 12 else 4
                expected_per_port = 1 / active_ports if adaptation is None else 0.25
                amplitude_scale = (4 * expected_per_port)**0.5
                self.assertAlmostEqual(link.grant_metadata["expected_tx_data_re_power"], active_ports / 4)
                self.assertAlmostEqual(link.grant_metadata["effective_tx_data_re_power"],
                                       active_ports * expected_per_port)
                self.assertEqual(np.count_nonzero(per_port_power), active_ports)
                np.testing.assert_allclose(per_port_power[per_port_power > 0], expected_per_port, atol=1e-6)
                _, receiver_noise, receiver_channel = link._receiver.call_args.args
                self.assertIs(receiver_noise, noise)
                np.testing.assert_allclose(receiver_channel.numpy(), (h * amplitude_scale).numpy())

    def test_short_grant_double_dmrs_separates_time_cdm_ports(self):
        import tensorflow as tf

        config = SimulationConfig()
        config = replace(
            config,
            carrier=replace(config.carrier, n_size_grid=12),
            pusch=replace(
                config.pusch,
                num_antenna_ports=2,
                rank=2,
                mcs_index=6,
                symbol_allocation=(0, 8),
                dmrs=replace(
                    config.pusch.dmrs,
                    length=2,
                    additional_position=1,
                    num_cdm_groups_without_data=1,
                    port_set=(0, 4),
                ),
            ),
            channel=replace(config.channel, kind="custom", num_rx_ant=2, normalize=False),
            sweep=replace(
                config.sweep, snr_db=(35.0,), num_tb_per_snr=3, batch_size=3, seed=144
            ),
        )

        def static_mimo_cir(_config, _resource_grid):
            gains = tf.constant([[1.0, 0.5], [0.4, 1.0]], dtype=tf.complex64)
            gains = tf.reshape(gains, (1, 1, 2, 1, 2, 1, 1))

            def cir(batch_size, num_time_steps, _sampling_frequency):
                a = tf.broadcast_to(gains, (batch_size, 1, 2, 1, 2, 1, num_time_steps))
                tau = tf.zeros((batch_size, 1, 1, 1), dtype=tf.float32)
                return a, tau

            return cir

        result = simulate(config, channel_model_factory=static_mimo_cir)
        self.assertEqual(result["metadata"]["pusch"]["dmrs_slot_symbols"], [2, 3])
        point, = result["points"]
        self.assertEqual((point["num_tb"], point["tb_errors"]), (3, 0))

    def test_cli_fit_budget_accepts_short_tb_and_warmup_overrides(self):
        import io
        from contextlib import redirect_stderr, redirect_stdout

        from pusch_baseline.__main__ import main as cli_main

        long_cfg = str(CONFIGS / "cdl_c_4t64r_codebooks_long.json")
        out = io.StringIO()
        with redirect_stdout(out):
            self.assertEqual(cli_main(["--config", long_cfg, "--num-tb", "20", "--check"]), 0)
            self.assertEqual(cli_main(["--config", long_cfg, "--warmup-tb", "15", "--num-tb", "15", "--check"]), 0)
        err = io.StringIO()
        with redirect_stderr(err):
            self.assertEqual(cli_main(["--warmup-tb", "10", "--check"]), 2)
        self.assertIn("requires adaptation.olla", err.getvalue())

    def test_worker_thread_configuration_applies_to_fresh_tensorflow_runtime(self):
        import json
        import os
        import subprocess
        import sys

        from pusch_baseline.simulation import apply_tensorflow_threads, configure_worker_env

        # Calling after TF is already initialized in-process must not raise RuntimeError.
        saved = {k: os.environ.get(k) for k in ("TF_NUM_INTRAOP_THREADS", "TF_NUM_INTEROP_THREADS")}
        try:
            os.environ["TF_NUM_INTRAOP_THREADS"] = "2"
            os.environ["TF_NUM_INTEROP_THREADS"] = "1"
            apply_tensorflow_threads()
        finally:
            for k, val in saved.items():
                if val is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = val

        # In a fresh child process, configure_worker_env + apply_tensorflow_threads
        # must set both TF's runtime thread pools and BLAS single-thread env vars.
        env = {k: v for k, v in os.environ.items()
               if k not in ("TF_NUM_INTRAOP_THREADS", "TF_NUM_INTEROP_THREADS",
                            "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")}
        code = (
            "import json, os\n"
            "from pusch_baseline.simulation import apply_tensorflow_threads, configure_worker_env\n"
            "configure_worker_env(2)\n"
            "apply_tensorflow_threads()\n"
            "import tensorflow as tf\n"
            "print(json.dumps({\n"
            "    'intra': tf.config.threading.get_intra_op_parallelism_threads(),\n"
            "    'inter': tf.config.threading.get_inter_op_parallelism_threads(),\n"
            "    'omp': os.environ.get('OMP_NUM_THREADS'),\n"
            "}))\n"
        )
        proc = subprocess.run(
            [sys.executable, "-B", "-c", code],
            cwd=Path(__file__).resolve().parents[2],
            env=env,
            check=True,
            capture_output=True,
            text=True,
        )
        reported = json.loads(proc.stdout.strip().splitlines()[-1])
        self.assertEqual(reported["intra"], min(os.cpu_count() or 1, 2))
        self.assertEqual(reported["inter"], 1)
        self.assertEqual(reported["omp"], "1")

    def test_explicit_worker_thread_flags_override_inherited_thread_limits(self):
        import argparse
        import os
        from unittest.mock import patch

        from pusch_baseline.simulation import add_worker_thread_arguments, configure_worker_env

        parser = argparse.ArgumentParser()
        add_worker_thread_arguments(parser)
        args = parser.parse_args(["--tf-intra-threads", "1", "--tf-inter-threads", "3"])
        with patch.dict(os.environ, {"TF_NUM_INTRAOP_THREADS": "4", "TF_NUM_INTEROP_THREADS": "2"}):
            configure_worker_env(16, intra_threads=2, inter_threads=1,
                                 tf_intra_threads=args.tf_intra_threads, tf_inter_threads=args.tf_inter_threads)
            self.assertEqual(os.environ["TF_NUM_INTRAOP_THREADS"], "1")
            self.assertEqual(os.environ["TF_NUM_INTEROP_THREADS"], "3")

        args = parser.parse_args([])
        with patch.dict(os.environ, {"TF_NUM_INTRAOP_THREADS": "4", "TF_NUM_INTEROP_THREADS": "2"}):
            configure_worker_env(8, intra_threads=2, inter_threads=1,
                                 tf_intra_threads=args.tf_intra_threads, tf_inter_threads=args.tf_inter_threads)
            self.assertEqual(os.environ["TF_NUM_INTRAOP_THREADS"], "4")
            self.assertEqual(os.environ["TF_NUM_INTEROP_THREADS"], "2")


if __name__ == "__main__":
    unittest.main()
