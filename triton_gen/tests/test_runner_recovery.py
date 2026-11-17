import json
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock

import pytest
import torch
from triton.runtime.errors import OutOfResources

from benchmarking import runner, storage


@pytest.fixture
def sweep(tmp_path, monkeypatch):
    def test_kernel():
        pass

    test_kernel.arg_names = ("n", "BLOCK_SIZE")
    cases = [((n,), {"BLOCK_SIZE": 32}, (1,)) for n in (1, 2, 3)]
    module = SimpleNamespace(KERNEL=test_kernel, iter_args=lambda device: iter(cases))
    modules = [module]
    paths = storage.OutputPaths(tmp_path)
    messages = []

    class Progress:
        def __init__(self, items, **kwargs):
            self.items = items

        def __iter__(self):
            return iter(self.items)

        def set_description(self, description):
            pass

        def write(self, message):
            messages.append(message)

    fake_triton = SimpleNamespace(
        runtime=SimpleNamespace(
            driver=SimpleNamespace(
                active=SimpleNamespace(get_active_torch_device=lambda: "cuda:0")
            )
        )
    )
    result = runner.BenchmarkResult(
        compiled_kernel=SimpleNamespace(
            name="compiled_kernel", asm={"ttgir": "module { test }"}
        ),
        time_ms=0.01,
        time_p20_ms=0.009,
        time_p80_ms=0.011,
        time_cv=0.2,
    )
    benchmark = Mock(return_value=result)
    clear_cache = Mock()
    monkeypatch.setattr(runner, "triton", fake_triton)
    monkeypatch.setattr(runner, "tqdm", Progress)
    monkeypatch.setattr(runner, "benchmark_kernel", benchmark)
    monkeypatch.setattr(runner, "clear_cuda_cache", clear_cache)

    def run(**kwargs):
        runner.run_sweep(
            modules, warmup_ms=100, rep_ms=1000, paths=paths, **kwargs
        )

    return SimpleNamespace(
        paths=paths,
        cases=cases,
        modules=modules,
        result=result,
        benchmark=benchmark,
        clear_cache=clear_cache,
        messages=messages,
        run=run,
    )


def read_records(sweep):
    return json.loads(sweep.paths.result_path.read_text())["test_kernel"]


@pytest.mark.parametrize(
    "error",
    [OutOfResources(4096, 1024, "shared memory"), torch.cuda.OutOfMemoryError("full")],
)
def test_skipped_case_is_checkpointed_and_not_retried(sweep, error):
    sweep.benchmark.side_effect = [error, RuntimeError("interrupted")]
    with pytest.raises(RuntimeError, match="interrupted"):
        sweep.run()

    checkpoint = json.loads(sweep.paths.resume_path.read_text())
    assert (checkpoint["next_kernel_index"], checkpoint["next_case_index"]) == (0, 1)
    skipped = checkpoint["result"]["test_kernel"][0]
    assert skipped["status"] == "skipped"
    assert skipped["error"] == f"{type(error).__name__}: {error}"
    for field in (
        "compiled_name", "ttgir_filename", "time_ms", "time_p20_ms",
        "time_p80_ms", "time_cv",
    ):
        assert skipped[field] is None
    assert not sweep.paths.ttgir_dir.exists()
    assert f"skipped: {type(error).__name__}: {error}\n" in sweep.messages
    assert sweep.clear_cache.call_count == 2

    sweep.benchmark.reset_mock(side_effect=True)
    sweep.run()

    records = read_records(sweep)
    assert [record["status"] for record in records] == ["skipped", "ok", "ok"]
    assert records[0] == skipped
    assert sweep.benchmark.call_count == 2
    assert len(list(sweep.paths.ttgir_dir.iterdir())) == 2
    assert not sweep.paths.resume_path.exists()


def interrupt_after_first_case(sweep):
    sweep.benchmark.side_effect = [sweep.result, RuntimeError("interrupted")]
    with pytest.raises(RuntimeError, match="interrupted"):
        sweep.run()
    sweep.benchmark.reset_mock(side_effect=True)


def test_interrupted_sweep_resumes_without_duplicate_records(sweep):
    interrupt_after_first_case(sweep)
    sweep.run()

    records = read_records(sweep)
    assert [record["scalar_args"]["n"] for record in records] == [1, 2, 3]
    assert sweep.benchmark.call_count == 2
    assert sweep.clear_cache.call_count == 4
    assert "Resuming at kernel 1/1, case 2" in sweep.messages
    assert "compiled_kernel_1_BLOCK_SIZE=32" in sweep.messages
    assert not sweep.paths.resume_path.exists()


def test_resume_across_kernel_boundary_preserves_completed_kernel(sweep):
    def second_kernel():
        pass

    second_kernel.arg_names = ("n", "BLOCK_SIZE")
    sweep.modules.append(SimpleNamespace(
        KERNEL=second_kernel,
        iter_args=lambda device: iter([
            ((4,), {"BLOCK_SIZE": 32}, (1,)),
            ((5,), {"BLOCK_SIZE": 32}, (1,)),
        ]),
    ))
    sweep.benchmark.side_effect = [sweep.result] * 4 + [RuntimeError("interrupted")]
    with pytest.raises(RuntimeError, match="interrupted"):
        sweep.run()

    checkpoint = json.loads(sweep.paths.resume_path.read_text())
    assert (checkpoint["next_kernel_index"], checkpoint["next_case_index"]) == (1, 1)
    sweep.benchmark.reset_mock(side_effect=True)
    sweep.run()

    result = json.loads(sweep.paths.result_path.read_text())
    assert result["test_kernel"] == checkpoint["result"]["test_kernel"]
    assert [record["scalar_args"]["n"] for record in result["second_kernel"]] == [4, 5]
    sweep.benchmark.assert_called_once()
    assert "Resuming at kernel 2/2, case 2" in sweep.messages
    assert not sweep.paths.resume_path.exists()


def test_default_paths_write_to_results(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    storage.write_result({"test_kernel": []})
    filename = storage.write_ttgir("kernel", "module {}")
    storage.save_resume_state({}, ["test_kernel"], 100, 1000, 0, 0)

    result = json.loads((tmp_path / "results/result.json").read_text())
    assert result == {"test_kernel": []}
    assert (tmp_path / "results/ttgir" / filename).read_text() == "module {}"
    assert storage.load_resume_state(["test_kernel"], 100, 1000)["next_case_index"] == 0


def test_resume_rejects_changed_case(sweep):
    interrupt_after_first_case(sweep)
    sweep.cases[0] = ((99,), {"BLOCK_SIZE": 32}, (1,))

    with pytest.raises(RuntimeError, match="changed since the checkpoint"):
        sweep.run()

    sweep.benchmark.assert_not_called()
    assert sweep.paths.resume_path.exists()


def test_resume_rejects_shortened_case_iterator(sweep):
    interrupt_after_first_case(sweep)
    sweep.cases.clear()

    with pytest.raises(RuntimeError, match="has only 0 cases"):
        sweep.run()

    sweep.benchmark.assert_not_called()
    assert sweep.paths.resume_path.exists()


def test_restart_discards_checkpoint_and_repeats_all_cases(sweep):
    interrupt_after_first_case(sweep)
    sweep.run(restart=True)

    assert sweep.benchmark.call_count == 3
    assert len(read_records(sweep)) == 3
    assert not sweep.paths.resume_path.exists()


def test_final_write_failure_preserves_checkpoint_and_previous_result(
    sweep, monkeypatch,
):
    sweep.paths.result_path.write_text('{"previous": []}')
    replace = storage.os.replace

    def fail_final_write(source, destination):
        if destination == sweep.paths.result_path:
            raise OSError("disk unavailable")
        replace(source, destination)

    with monkeypatch.context() as patch:
        patch.setattr(storage.os, "replace", fail_final_write)
        with pytest.raises(OSError, match="disk unavailable"):
            sweep.run()

    checkpoint = json.loads(sweep.paths.resume_path.read_text())
    assert (checkpoint["next_kernel_index"], checkpoint["next_case_index"]) == (1, 0)
    assert json.loads(sweep.paths.result_path.read_text()) == {"previous": []}

    sweep.benchmark.reset_mock()
    sweep.run()

    sweep.benchmark.assert_not_called()
    assert len(read_records(sweep)) == 3
    assert "Resuming final result write" in sweep.messages
    assert not sweep.paths.resume_path.exists()
    assert not sweep.paths.result_path.with_suffix(".json.new").exists()


@pytest.mark.parametrize(
    "error", [OSError("write failed"), torch.cuda.OutOfMemoryError("write")],
)
def test_ttgir_write_errors_propagate_instead_of_becoming_skips(
    sweep, monkeypatch, error,
):
    monkeypatch.setattr(storage, "write_ttgir", Mock(side_effect=error))

    with pytest.raises(type(error), match=str(error)):
        sweep.run()

    sweep.clear_cache.assert_called_once_with()
    assert not sweep.paths.resume_path.exists()
    assert not sweep.paths.result_path.exists()


def test_checkpoint_write_failure_retries_only_uncommitted_case(sweep, monkeypatch):
    save_resume_state = storage.save_resume_state

    def fail_second_case(*args, **kwargs):
        if args[-2:] == (0, 2):
            raise OSError("checkpoint unavailable")
        save_resume_state(*args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(storage, "save_resume_state", fail_second_case)
        with pytest.raises(OSError, match="checkpoint unavailable"):
            sweep.run()

    checkpoint = json.loads(sweep.paths.resume_path.read_text())
    assert checkpoint["next_case_index"] == 1
    assert len(checkpoint["result"]["test_kernel"]) == 1
    sweep.benchmark.reset_mock()
    sweep.run()

    assert sweep.benchmark.call_count == 2
    assert [record["scalar_args"]["n"] for record in read_records(sweep)] == [1, 2, 3]
    assert len(list(sweep.paths.ttgir_dir.iterdir())) == 3
    assert not sweep.paths.resume_path.exists()


@pytest.mark.parametrize(
    "times, expected_cv",
    [([0.009, 0.01, 0.011], 0.2), ([0.0, 0.0, 0.0], 0.0), ([0.02, 0.01, 0.0], 0.0)],
)
def test_benchmark_returns_named_timings(monkeypatch, times, expected_cv):
    kernel = MagicMock()
    launch = kernel.__getitem__.return_value
    compiled = launch.return_value
    synchronize = Mock()

    def do_bench(fn, **kwargs):
        launch.assert_called_once_with(123, BLOCK_SIZE=32)
        synchronize.assert_called_once_with()
        assert kwargs == {"warmup": 25, "rep": 250, "quantiles": [0.2, 0.5, 0.8]}
        fn()
        return times

    monkeypatch.setattr(
        runner, "triton", SimpleNamespace(testing=SimpleNamespace(do_bench=do_bench))
    )
    monkeypatch.setattr(runner.torch.cuda, "synchronize", synchronize)

    result = runner.benchmark_kernel(kernel, (1,), (123,), {"BLOCK_SIZE": 32}, 25, 250)

    assert result.compiled_kernel is compiled
    assert (result.time_p20_ms, result.time_ms, result.time_p80_ms) == tuple(times)
    assert result.time_cv == pytest.approx(expected_cv)
    assert launch.call_count == 2
