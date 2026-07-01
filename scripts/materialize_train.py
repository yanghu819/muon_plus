#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path


UPSTREAM_SHA = "df78af0db523d8bceb25af4919a3e3e7082b80f3"


def replace_once(text: str, old: str, new: str) -> str:
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"expected one occurrence of {old!r}, found {count}")
    return text.replace(old, new, 1)


def patch_common_paths(text: str, train_pattern: str, val_pattern: str) -> str:
    replacements = [
        ("input_bin : str = 'data/fineweb10B/fineweb_train_*.bin'", f"input_bin : str = {train_pattern!r}"),
        ("input_val_bin : str = 'data/fineweb10B/fineweb_val_*.bin'", f"input_val_bin : str = {val_pattern!r}"),
        ('train_files = "data/fineweb10B/fineweb_train_*.bin"', f"train_files = {train_pattern!r}"),
        ('val_files = "data/fineweb10B/fineweb_val_*.bin"', f"val_files = {val_pattern!r}"),
    ]
    for old, new in replacements:
        if old in text:
            text = replace_once(text, old, new)
    return text


def patch_smoke_1(text: str) -> str:
    replacements = [
        ("batch_size : int = 8*64", "batch_size : int = 4"),
        ("device_batch_size : int = 64", "device_batch_size : int = 4"),
        ("sequence_length : int = 1024", "sequence_length : int = 128"),
        ("num_iterations : int = 6200", "num_iterations : int = 2"),
        ("warmdown_iters : int = 1800", "warmdown_iters : int = 1"),
        ("val_loss_every : int = 100", "val_loss_every : int = 1"),
        ("val_tokens : int = 10485760", "val_tokens : int = 4096"),
        (
            "GPT(GPTConfig(vocab_size=num_vocab, n_layer=12, n_head=12, n_embd=768))",
            "GPT(GPTConfig(vocab_size=num_vocab, n_layer=2, n_head=2, n_embd=128))",
        ),
        ("model = torch.compile(model)", "model = model  # smoke: skip torch.compile startup cost"),
    ]
    for old, new in replacements:
        text = replace_once(text, old, new)
    return text


def patch_resumable_full_1(text: str) -> str:
    save_every_replaced = False
    for old in (
        "save_every : int = 0 # every how many steps to save the checkpoint? 0 for only at the end",
        "save_every : int = 0",
    ):
        if old in text:
            text = replace_once(
                text,
                old,
                "save_every : int = int(os.environ.get(\"SAVE_EVERY\", \"100\")) # every how many steps to save the checkpoint? 0 for only at the end",
            )
            save_every_replaced = True
            break
    if not save_every_replaced:
        raise SystemExit("could not find save_every hyperparameter")
    for old, new in (
        ("num_iterations : int = 6200", "num_iterations : int = int(os.environ.get(\"NUM_ITERATIONS\", \"6200\"))"),
        ("warmdown_iters : int = 1800", "warmdown_iters : int = int(os.environ.get(\"WARMDOWN_ITERS\", \"1800\"))"),
        ("val_loss_every : int = 100", "val_loss_every : int = int(os.environ.get(\"VAL_LOSS_EVERY\", \"100\"))"),
        ("val_tokens : int = 10485760", "val_tokens : int = int(os.environ.get(\"VAL_TOKENS\", \"10485760\"))"),
    ):
        if old in text:
            text = replace_once(text, old, new)
    text = replace_once(
        text,
        "elif it < args.num_iterations - args.warmdown_iters:",
        "elif args.warmdown_iters <= 0 or it < args.num_iterations - args.warmdown_iters:",
    )
    text = replace_once(
        text,
        "    def next_batch(self):\n"
        "        B = self.B\n"
        "        T = self.T\n"
        "        buf = self.tokens[self.current_position : self.current_position+B*T+1]\n",
        "    def seek_batch(self, batch_index):\n"
        "        stride = self.B * self.T * self.num_processes\n"
        "        offset = self.process_rank * self.B * self.T\n"
        "        shard_sizes = [_peek_data_shard(fname) for fname in self.files]\n"
        "        shard_batches = [max(0, (size - offset - stride - 1) // stride + 1) for size in shard_sizes]\n"
        "        total_batches = sum(shard_batches)\n"
        "        assert total_batches > 0\n"
        "        batch_index = batch_index % total_batches\n"
        "        for shard, count in enumerate(shard_batches):\n"
        "            if batch_index < count:\n"
        "                self.current_shard = shard\n"
        "                self.current_position = offset + batch_index * stride\n"
        "                self.tokens = _load_data_shard(self.files[self.current_shard])\n"
        "                return self.next_batch()\n"
        "            batch_index -= count\n"
        "        raise RuntimeError(\"failed to seek dataloader batch\")\n\n"
        "    def next_batch(self):\n"
        "        B = self.B\n"
        "        T = self.T\n"
        "        buf = self.tokens[self.current_position : self.current_position+B*T+1]\n",
    )
    text = replace_once(
        text,
        "schedulers = [torch.optim.lr_scheduler.LambdaLR(opt, get_lr) for opt in optimizers]",
        "schedulers = [torch.optim.lr_scheduler.LambdaLR(opt, get_lr) for opt in optimizers]\n\n"
        "def _rebuild_preconditioner_views_after_load(opt):\n"
        "    if not hasattr(opt, \"_finalize_precond_buffers_\"):\n"
        "        return\n"
        "    if not getattr(opt, \"_precond_attached\", False):\n"
        "        return\n"
        "    saved_inv = {}\n"
        "    for group in opt.param_groups:\n"
        "        for p in group[\"params\"]:\n"
        "            st = opt.state.get(p, {})\n"
        "            inv = st.get(\"precond_inv_apply\")\n"
        "            if torch.is_tensor(inv):\n"
        "                saved_inv[p] = inv.detach().clone()\n"
        "    opt._precond_ready = False\n"
        "    opt._refresh_map = []\n"
        "    opt._refresh_K = None\n"
        "    opt._apply_plan = None\n"
        "    opt._finalize_precond_buffers_()\n"
        "    for p, inv in saved_inv.items():\n"
        "        st = opt.state.get(p, {})\n"
        "        if \"precond_inv_apply\" in st:\n"
        "            st[\"precond_inv_apply\"].copy_(inv)\n\n"
        "resume_training_time_ms = float(os.environ.get(\"RESUME_TRAIN_TIME_MS\", \"0\"))\n"
        "resume_step = int(os.environ.get(\"RESUME_STEP\", \"0\"))\n"
        "resume_checkpoint = os.environ.get(\"RESUME_CHECKPOINT\", \"\")\n"
        "if resume_checkpoint:\n"
        "    ckpt = torch.load(resume_checkpoint, map_location=\"cuda\")\n"
        "    resume_step = int(ckpt[\"step\"])\n"
        "    resume_training_time_ms = float(ckpt.get(\"training_time_ms\", resume_training_time_ms))\n"
        "    raw_model.load_state_dict(ckpt[\"model\"])\n"
        "    for opt, state in zip(optimizers, ckpt[\"optimizers\"]):\n"
        "        opt.load_state_dict(state)\n"
        "        _rebuild_preconditioner_views_after_load(opt)\n"
        "    for sched, state in zip(schedulers, ckpt.get(\"schedulers\", [])):\n"
        "        sched.load_state_dict(state)\n"
        "    if resume_step > 0 and not ckpt.get(\"schedulers\"):\n"
        "        for sched in schedulers:\n"
        "            sched.step(resume_step)\n"
        "    print(f\"resuming from {resume_checkpoint} at step {resume_step}\")",
    )
    text = replace_once(
        text,
        "log = dict(step=step, code=code, model=raw_model.state_dict(), optimizers=[opt.state_dict() for opt in optimizers])",
        "log = dict(step=step, code=code, model=raw_model.state_dict(), optimizers=[opt.state_dict() for opt in optimizers], schedulers=[sched.state_dict() for sched in schedulers], training_time_ms=training_time_ms)",
    )
    training_time_replaced = False
    for old, new in (
        (
            "training_time_ms = 0\n# start the clock",
            "training_time_ms = resume_training_time_ms\n# start the clock",
        ),
        (
            "training_time_ms = 0\ntorch.cuda.synchronize()",
            "training_time_ms = resume_training_time_ms\ntorch.cuda.synchronize()",
        ),
    ):
        if old in text:
            text = replace_once(text, old, new)
            training_time_replaced = True
            break
    if not training_time_replaced:
        raise SystemExit("could not find initial training_time_ms assignment")
    text = replace_once(
        text,
        "train_loader.reset()\nfor step in range(args.num_iterations + 1):",
        "train_loader.reset()\n"
        "if resume_step > 0:\n"
        "    x, y = train_loader.seek_batch(resume_step * train_accumulation_steps - 1)\n"
        "for step in range(resume_step, args.num_iterations + 1):",
    )
    return text


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--mode", choices=["smoke", "full"], required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()

    source = Path(args.source)
    text = source.read_text()

    if args.mode == "smoke":
        if not args.method.endswith("1"):
            raise SystemExit("smoke mode currently supports the *_1 single-GPU scripts")
        data_root = Path(args.run_dir) / "data" / "fineweb10B"
    else:
        data_root = Path(args.repo_root) / "data" / "fineweb10B"

    text = patch_common_paths(
        text,
        str(data_root / "fineweb_train_*.bin"),
        str(data_root / "fineweb_val_*.bin"),
    )
    if args.mode == "smoke":
        text = patch_smoke_1(text)
    elif args.method.endswith("1"):
        text = patch_resumable_full_1(text)

    banner = (
        f"# Materialized by scripts/materialize_train.py from {source.name}\n"
        f"# Upstream Newton-Muon SHA: {UPSTREAM_SHA}\n"
        f"# Mode: {args.mode}; method: {args.method}\n\n"
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(banner + text)


if __name__ == "__main__":
    main()
