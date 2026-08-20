"""Transfer function correctness: round trips and published anchor points."""

import numpy as np
import pytest

from vct.core.transfer import (TRANSFERS, get_transfer, hlg_decode, hlg_encode,
                               pq_decode, pq_encode)


@pytest.mark.parametrize("tf_id", sorted(TRANSFERS))
def test_encode_decode_round_trip(tf_id):
    """decode(encode(x)) == x over each curve's working domain."""
    tf = TRANSFERS[tf_id]
    # LOG curves carry many stops above white; SDR curves only go to 1.0.
    top = 60.0 if tf.kind == "log" else 1.0
    x = np.concatenate([
        np.linspace(0.0, 0.01, 40),
        np.linspace(0.01, 1.0, 200),
        np.linspace(1.0, top, 100),
    ])
    back = tf.decode(tf.encode(x))
    assert np.allclose(back, x, atol=1e-5, rtol=1e-4), \
        f"{tf_id} round trip max error {np.abs(back - x).max():.2e}"


@pytest.mark.parametrize("tf_id", sorted(TRANSFERS))
def test_monotonic_and_anchored(tf_id):
    tf = TRANSFERS[tf_id]
    x = np.linspace(0.0, 4.0, 500)
    y = np.asarray(tf.encode(x))
    assert np.all(np.diff(y) >= -1e-6), f"{tf_id} encode is not monotonic"
    assert np.isfinite(y).all(), f"{tf_id} encode produced non-finite values"


@pytest.mark.parametrize("tf_id,expected", [
    ("slog3", 420.0 / 1023.0),     # Sony S-Log3 Technical Summary
    ("vlog", 0.42335),             # Panasonic V-Log reference manual
    ("logc3", 0.391006),           # ARRI LogC3 EI 800
    ("clog2", 0.403550),           # Canon Log 2
    ("clog3", 0.343390),           # Canon Log 3
    ("nlog", 0.363636),            # Nikon N-Log
    ("applelog", 500.0 / 1023.0),  # Apple Log white paper: 18% grey at code 500
    ("bmdfilm5", 0.383547),        # Blackmagic Film Gen 5
])
def test_published_mid_grey(tf_id, expected):
    """Every LOG curve must put 18% scene grey where its vendor says it does.

    Tolerance is one 10-bit code value, which is the precision these are
    published to.
    """
    got = float(get_transfer(tf_id).encode(np.array(0.18)))
    assert abs(got - expected) < 1.0 / 1023.0, \
        f"{tf_id}: 18% grey at {got:.5f}, expected {expected:.5f}"


def test_apple_log_ninety_percent_white():
    """The white paper's second anchor: 90% white at 10-bit code 697."""
    got = float(get_transfer("applelog").encode(np.array(0.9)))
    assert abs(got * 1023.0 - 697.0) < 1.0


def test_pq_absolute_luminance():
    """PQ is absolute: known signal levels must land on known nits."""
    # ITU-R BT.2408 diffuse white, 203 cd/m2, is at ~58% PQ signal.
    assert abs(float(pq_encode(np.array(203.0 / 10000.0))) - 0.5806) < 0.001
    # Peak of the PQ range is 10 000 cd/m2 at signal 1.0.
    assert abs(float(pq_decode(np.array(1.0))) - 1.0) < 1e-6
    assert abs(float(pq_decode(np.array(0.0)))) < 1e-9


def test_hlg_reference_white():
    """HLG reference white sits at 75% signal (ITU-R BT.2100)."""
    scene = float(hlg_decode(np.array(0.75)))
    assert abs(scene - 0.26496) < 1e-4
    # And the OOTF at a 1000 cd/m2 display puts it on BT.2408's 203 cd/m2.
    assert abs(1000.0 * scene ** 1.2 - 203.0) < 1.0
    assert abs(float(hlg_encode(np.array(1.0))) - 1.0) < 1e-6


def test_srgb_and_bt709_breakpoints():
    assert abs(float(get_transfer("srgb").encode(np.array(1.0))) - 1.0) < 1e-6
    assert abs(float(get_transfer("bt709").encode(np.array(1.0))) - 1.0) < 1e-6
    assert abs(float(get_transfer("srgb").decode(np.array(0.5))) - 0.2140) < 1e-3


def test_aliases_resolve():
    assert get_transfer("smpte2084").id == "pq"
    assert get_transfer("arib-std-b67").id == "hlg"
    assert get_transfer("iec61966-2-1").id == "srgb"
    with pytest.raises(KeyError):
        get_transfer("not-a-curve")


def test_negative_values_survive():
    """Noise below black must not become NaN and blow up a whole frame."""
    x = np.array([-0.05, -0.01, 0.0])
    for tf in TRANSFERS.values():
        assert np.isfinite(np.asarray(tf.encode(x))).all(), tf.id
        assert np.isfinite(np.asarray(tf.decode(x))).all(), tf.id
