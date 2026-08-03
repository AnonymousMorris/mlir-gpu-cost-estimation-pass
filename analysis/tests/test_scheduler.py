from __future__ import annotations

import pytest

from scheduler import schedule_work


CATEGORIES = {
    "fp32": 10.0,
    "fp64": 0.0,
    "sfu": 2.0,
    "tensor": 4.0,
    "memory": 20.0,
}


def test_small_grid_runs_in_one_wave():
    work, schedule = schedule_work(
        CATEGORIES,
        program_count=4,
        num_ctas=1,
        num_sms=30,
    )

    assert work == CATEGORIES
    assert schedule.total_blocks == 4
    assert schedule.blocks_per_wave == 4
    assert schedule.waves == 1


def test_grid_larger_than_gpu_scales_work_by_discrete_waves():
    work, schedule = schedule_work(
        CATEGORIES,
        program_count=31,
        num_ctas=1,
        num_sms=30,
    )

    assert work == {category: value * 2 for category, value in CATEGORIES.items()}
    assert schedule.total_blocks == 31
    assert schedule.blocks_per_wave == 30
    assert schedule.waves == 2


def test_ctas_per_program_contribute_to_total_block_count():
    _, schedule = schedule_work(
        CATEGORIES,
        program_count=16,
        num_ctas=2,
        num_sms=30,
    )

    assert schedule.total_blocks == 32
    assert schedule.waves == 2


@pytest.mark.parametrize(
    ("program_count", "num_ctas", "num_sms"),
    [(0, 1, 30), (1, 0, 30), (1, 1, 0)],
)
def test_asserts_positive_schedule_inputs(program_count, num_ctas, num_sms):
    with pytest.raises(AssertionError):
        schedule_work(
            CATEGORIES,
            program_count=program_count,
            num_ctas=num_ctas,
            num_sms=num_sms,
        )
