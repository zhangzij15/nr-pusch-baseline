"""NR RV0 coding for grants requiring explicit base-graph or rate matching."""

from dataclasses import dataclass

import tensorflow as tf
from sionna.phy.fec.crc import CRCDecoder, CRCEncoder
from sionna.phy.fec.ldpc import LDPC5GDecoder, LDPC5GEncoder
from sionna.phy.fec.scrambling import Descrambler, TB5GScrambler
from sionna.phy.nr.utils import calculate_tb_size


@dataclass(frozen=True)
class LowRatePlan:
    tb_size: int
    cb_size: int
    tb_crc_length: int
    cb_crc_length: int
    cw_lengths: tuple[int, ...]
    base_graph: str


def _base_graph_two(payload_size, rate):
    return payload_size <= 292 or (payload_size <= 3824 and rate <= 0.67) or rate <= 0.25


def low_rate_plan(pc, *, allow_low_rate=True):
    tbs, cb_size, _, tb_crc, cb_crc, cw_lengths = calculate_tb_size(
        target_tb_size=pc.tb_size, num_coded_bits=pc.num_coded_bits,
        target_coderate=pc.tb.target_coderate, modulation_order=pc.tb.num_bits_per_symbol,
        num_layers=pc.num_layers,
    )
    k = int(cb_size)
    lengths = tuple(int(n) for n in cw_lengths.numpy())
    effective_rate = k / max(lengths)
    # Sionna 1.2.2 gives its LDPC constructor the code-block size (including
    # CRC) and k/E. TS 38.212 §6.2.2 uses transport payload A and target R.
    native_bg2 = _base_graph_two(k, effective_rate)
    bg2 = _base_graph_two(int(tbs), float(pc.tb.target_coderate))
    below_native_limit = effective_rate < (1 / 5 if bg2 else 1 / 3)
    if not below_native_limit and native_bg2 == bg2:
        return None
    if below_native_limit and not allow_low_rate:
        raise ValueError("This grant requires low-rate RV0 rate matching")
    return LowRatePlan(int(tbs), k, int(tb_crc), int(cb_crc), lengths, "bg2" if bg2 else "bg1")


class _Rv0MotherEncoder(LDPC5GEncoder):
    def __init__(self, plan, precision):
        # Initialize the native lifting and parity matrices at a supported
        # length. Its no-interleaver encoder can then return every required
        # distinct bit after the initial 2Z puncture and removal of filler.
        supported_length = plan.cb_size * (5 if plan.base_graph == "bg2" else 3)
        super().__init__(plan.cb_size, min(supported_length, max(plan.cw_lengths)),
                         bg=plan.base_graph, precision=precision)
        mother_length = self.n_ldpc - (self.k_ldpc - self.k) - 2 * self.z
        self._n = min(mother_length, max(plan.cw_lengths))
        self._coderate = self.k / self._n


class LowRateTb:
    """Use native Sionna CRC, LDPC and scrambling with an uncapped RV0 buffer."""

    def __init__(self, plan, modulation_order, rnti, scrambling_id, precision):
        self.tb_size = plan.tb_size
        self._num_blocks = len(plan.cw_lengths)
        self._cb_payload = plan.cb_size - plan.cb_crc_length
        self._tb_with_crc = plan.tb_size + plan.tb_crc_length
        self._cw_lengths = plan.cw_lengths
        self._ldpc_encoder = _Rv0MotherEncoder(plan, precision)
        self._ldpc_decoder = LDPC5GDecoder(
            self._ldpc_encoder, prune_pcm=False, precision=precision,
        )
        tb_crc_polynomial = "CRC16" if plan.tb_crc_length == 16 else "CRC24A"
        self._tb_crc = CRCEncoder(tb_crc_polynomial, precision=precision)
        # CRCDecoder checks the full codeword. Its generator matrix has a
        # different input length and must not be shared with the transmitter.
        self._tb_crc_decoder = CRCDecoder(
            CRCEncoder(tb_crc_polynomial, precision=precision), precision=precision,
        )
        self._cb_crc = CRCEncoder("CRC24B", precision=precision) if plan.cb_crc_length else None
        self._cb_crc_decoder = (
            CRCDecoder(CRCEncoder("CRC24B", precision=precision), precision=precision)
            if plan.cb_crc_length else None
        )
        self._scrambler = TB5GScrambler(n_rnti=rnti, n_id=scrambling_id, channel_type="PUSCH",
                                          binary=True, precision=precision)
        self._descrambler = Descrambler(self._scrambler, binary=False, precision=precision)
        self._selection = []
        for length in self._cw_lengths:
            interleave, _ = self._ldpc_encoder.generate_out_int(length, modulation_order)
            # 38.212 §5.4.2.1 and §5.4.2.2. Once filler is removed the
            # zero-offset circular buffer repeats with this native length.
            self._selection.append(tf.constant(interleave % self._ldpc_encoder.n, dtype=tf.int32))

    def encode(self, bits):
        crc_bits = self._tb_crc(bits)
        blocks = tf.reshape(crc_bits, (-1, 1, self._num_blocks, self._cb_payload))
        if self._cb_crc is not None:
            blocks = self._cb_crc(blocks)
        encoded = self._ldpc_encoder(blocks)
        matched = [tf.gather(encoded[:, :, row], indices, axis=-1)
                   for row, indices in enumerate(self._selection)]
        return self._scrambler(tf.concat(matched, axis=-1))

    def decode(self, logits):
        blocks = tf.split(self._descrambler(logits), self._cw_lengths, axis=-1)
        combined = [
            tf.transpose(tf.math.unsorted_segment_sum(tf.transpose(block, (2, 0, 1)), indices,
                                                     self._ldpc_encoder.n), (1, 2, 0))
            for block, indices in zip(blocks, self._selection)
        ]
        decoded = self._ldpc_decoder(tf.stack(combined, axis=-2))
        if self._cb_crc_decoder is not None:
            decoded, _ = self._cb_crc_decoder(decoded)
        bits, crc_ok = self._tb_crc_decoder(tf.reshape(decoded, (-1, 1, self._tb_with_crc)))
        return bits, tf.squeeze(tf.cast(crc_ok, tf.bool), axis=-1)


def low_rate_tb(pc, plan, precision, cache):
    qam = int(pc.tb.num_bits_per_symbol)
    rnti = pc.n_rnti
    nid = pc.carrier.n_cell_id if pc.tb.n_id is None else pc.tb.n_id
    key = ("rv0", plan, qam, rnti, nid, precision)
    if cache is not None and key in cache:
        return cache[key]
    codec = LowRateTb(plan, qam, rnti, nid, precision)
    decoder = codec.decode
    if cache is not None:
        decoder = tf.function(decoder, autograph=False, reduce_retracing=True)
        cache[key] = (codec, decoder)
    return codec, decoder
