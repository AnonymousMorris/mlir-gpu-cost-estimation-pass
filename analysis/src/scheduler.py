from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Schedule:
    model: str
    program_count: int
    num_ctas: int
    total_blocks: int
    num_sms: int
    blocks_per_wave: int
    waves: int


def schedule_work(
    per_block_work: dict[str, float],
    *,
    program_count: int,
    num_ctas: int,
    num_sms: int,
) -> tuple[dict[str, float], Schedule]:
    assert program_count > 0
    assert num_ctas > 0
    assert num_sms > 0

    total_blocks = program_count * num_ctas
    waves = total_blocks // num_sms + (total_blocks % num_sms != 0)
    scheduled_work = {
        category: work * waves for category, work in per_block_work.items()
    }
    schedule = Schedule(
        model="one_block_per_sm",
        program_count=program_count,
        num_ctas=num_ctas,
        total_blocks=total_blocks,
        num_sms=num_sms,
        blocks_per_wave=min(total_blocks, num_sms),
        waves=waves,
    )
    return scheduled_work, schedule
