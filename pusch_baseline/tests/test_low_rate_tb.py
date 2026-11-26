from dataclasses import replace
from hashlib import sha256
from pathlib import Path
import unittest

import tensorflow as tf
from sionna.phy import config as sionna_config

from pusch_baseline import load_config
from pusch_baseline.link import PuschLink, _pusch_config
from pusch_baseline.low_rate_tb import low_rate_plan, low_rate_tb


CONFIG = Path(__file__).parents[1] / "configs" / "cdl_c_4t64r_codebooks_long.json"


class LowRateTbTests(unittest.TestCase):
    def test_shared_codec_accepts_separately_compiled_slot_links(self):
        config = load_config(CONFIG)
        config = replace(config, adaptation=None, channel=replace(config.channel, kind="custom", num_rx_ant=1))
        h = tf.ones((2, 1, 1, 1, 4, 14, 576), tf.complex64)
        sionna_config.seed = 42
        for mcs in (0, 4):
            cache = {}
            for slot in (2, 7):
                with self.subTest(mcs=mcs, slot=slot):
                    link = PuschLink(
                        replace(config, carrier=replace(config.carrier, slot_number=slot),
                                pusch=replace(config.pusch, mcs_index=mcs)),
                        external_channel=True, tb_decoder_cache=cache, allow_low_rate=True,
                    )
                    execute = tf.function(link.count_block_errors_tensor, autograph=False)
                    self.assertEqual(int(execute(2, link.noise_variance(30), channel_response=h).numpy()), 0)

    def test_table_one_low_rates_decode_on_the_native_pusch_grid(self):
        config = load_config(CONFIG)
        config = replace(config, adaptation=None, channel=replace(config.channel, kind="custom", num_rx_ant=1))
        h = tf.ones((1, 1, 1, 1, 4, 14, 576), tf.complex64)
        sionna_config.seed = 42
        cache = {}
        for mcs in (0, 1, 2, 3, 4, 5):
            with self.subTest(mcs=mcs):
                link = PuschLink(
                    replace(config, pusch=replace(config.pusch, mcs_index=mcs)),
                    external_channel=True, tb_decoder_cache=cache, allow_low_rate=True,
                )
                self.assertEqual(link.grant_metadata["coded_bits"], 13824)
                self.assertEqual(link.count_block_errors(1, link.noise_variance(30), channel_response=h), 0)

    def test_rate_matched_bits_match_matlab_ulsch(self):
        # MATLAB R2024b nrULSCH with QPSK, one layer, E=13824 and RV=0.
        hashes = {
            0: "a26e0f641b51d7be01488f720060d5666b3bcca0c997ab469833ea4e17a46ce9",
            1: "a61736a5a6effc864a319635479d935907e046de79bafbf5cda081ffeb45634e",
            2: "489dd981ac9688f62ddf91b20098f24cc58a13a5a9e4dca1b0bdb4dd80936f31",
            4: "8afb66e0c8181fc05c462f9dca02e87e580c2bed6d6b6990399255dea9b604b8",
        }
        config = load_config(CONFIG)
        for mcs, expected in hashes.items():
            with self.subTest(mcs=mcs):
                pc = _pusch_config(replace(config, pusch=replace(config.pusch, mcs_index=mcs)))
                plan = low_rate_plan(pc)
                codec, _ = low_rate_tb(pc, plan, config.precision, None)
                index = tf.range(plan.tb_size)
                payload = tf.cast((7 * index + index % 11) % 17 < 8, tf.float32)[None, None]
                # Remove the PUSCH-specific scrambling, which nrULSCH leaves to nrPUSCH.
                coded = codec._scrambler(codec.encode(payload)).numpy().astype("uint8")
                self.assertEqual(sha256(coded.tobytes()).hexdigest(), expected)

    def test_fixed_grant_simulate_supports_low_rate_mcs_across_all_tables(self):
        from pusch_baseline import SimulationConfig, simulate

        base = SimulationConfig()
        base = replace(
            base,
            carrier=replace(base.carrier, n_size_grid=12),
            sweep=replace(base.sweep, snr_db=(25.0,), num_tb_per_snr=2, batch_size=2, seed=7),
        )
        for mcs_table in (1, 2, 3):
            with self.subTest(mcs_table=mcs_table):
                cfg = replace(base, pusch=replace(base.pusch, mcs_table=mcs_table, mcs_index=0))
                run = simulate(cfg)
                point, = run["points"]
                self.assertEqual((point["num_tb"], point["tb_errors"]), (2, 0))
