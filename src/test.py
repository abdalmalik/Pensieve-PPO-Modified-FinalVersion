import argparse
import os
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np

import fixed_env as env
import load_trace
import ppo2 as network


S_INFO = 6
S_LEN = 8
A_DIM = 6
ACTOR_LR_RATE = 0.0001
VIDEO_BIT_RATE = [300, 750, 1200, 1850, 2850, 4300]  # Kbps
BUFFER_NORM_FACTOR = 10.0
CHUNK_TIL_VIDEO_END_CAP = 48.0
M_IN_K = 1000.0
REBUF_PENALTY = 4.3
SMOOTH_PENALTY = 1
DEFAULT_QUALITY = 1
RANDOM_SEED = 42
LOG_FILE_PREFIX = "log_sim_ppo"
TEST_TRACES = "./test/"


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a Pensieve PPO checkpoint on fixed test traces.")
    parser.add_argument("nn_model", help="Path to the .pth checkpoint to evaluate.")
    parser.add_argument(
        "--traces-dir",
        default=TEST_TRACES,
        help="Directory containing the fixed evaluation traces.",
    )
    parser.add_argument(
        "--output-dir",
        default="./test_results",
        help="Directory where per-trace evaluation logs will be written.",
    )
    parser.add_argument(
        "--policy",
        choices=["legacy-gumbel", "argmax", "safe-step"],
        default="safe-step",
        help="Inference policy used during evaluation.",
    )
    parser.add_argument(
        "--buffer-reserve",
        type=float,
        default=5.0,
        help="Safety reserve subtracted from the current buffer when policy=safe-step.",
    )
    parser.add_argument(
        "--min-safety-budget",
        type=float,
        default=2.0,
        help="Minimum download-time budget used by the safety layer when policy=safe-step.",
    )
    parser.add_argument(
        "--max-upstep",
        type=int,
        default=1,
        help="Maximum upward bitrate jump per chunk when policy=safe-step.",
    )
    return parser.parse_args()


def build_log_path(output_dir: Path, trace_name: str) -> Path:
    return output_dir / f"{LOG_FILE_PREFIX}_{Path(trace_name).name}"


def select_bitrate(policy, action_prob, state, next_video_chunk_sizes, buffer_size, last_bit_rate, buffer_reserve, min_safety_budget, max_upstep):
    if policy == "legacy-gumbel":
        noise = np.random.gumbel(size=len(action_prob))
        return int(np.argmax(np.log(action_prob) + noise))

    target_idx = int(np.argmax(action_prob))
    if policy == "argmax":
        return target_idx

    throughput = max(float(state[2, -1]), 1e-6)
    next_chunk_sizes_mb = np.array(next_video_chunk_sizes) / M_IN_K / M_IN_K
    budget = max(buffer_size - buffer_reserve, min_safety_budget)

    feasible = [idx for idx, chunk_size_mb in enumerate(next_chunk_sizes_mb) if (chunk_size_mb / throughput) <= budget]
    safe_idx = feasible[-1] if feasible else 0
    chosen_idx = min(target_idx, safe_idx)

    if max_upstep is not None:
        chosen_idx = min(chosen_idx, int(last_bit_rate) + int(max_upstep))

    return int(np.clip(chosen_idx, 0, A_DIM - 1))


def main():
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    np.random.seed(RANDOM_SEED)
    assert len(VIDEO_BIT_RATE) == A_DIM

    all_cooked_time, all_cooked_bw, all_file_names = load_trace.load_trace(args.traces_dir)
    net_env = env.Environment(all_cooked_time=all_cooked_time, all_cooked_bw=all_cooked_bw)

    actor = network.Network(state_dim=[S_INFO, S_LEN], action_dim=A_DIM, learning_rate=ACTOR_LR_RATE)
    actor.load_model(args.nn_model)
    print("Testing model restored.")
    print(
        "Policy:",
        args.policy,
        f"(buffer_reserve={args.buffer_reserve}, min_safety_budget={args.min_safety_budget}, max_upstep={args.max_upstep})",
    )

    log_path = build_log_path(output_dir, all_file_names[net_env.trace_idx])
    log_file = log_path.open("w", encoding="utf-8")

    time_stamp = 0
    last_bit_rate = DEFAULT_QUALITY
    bit_rate = DEFAULT_QUALITY

    action_vec = np.zeros(A_DIM)
    action_vec[bit_rate] = 1

    s_batch = [np.zeros((S_INFO, S_LEN))]
    a_batch = [action_vec]
    r_batch = []
    entropy_record = []
    entropy_ = 0.5
    video_count = 0

    while True:
        delay, sleep_time, buffer_size, rebuf, video_chunk_size, next_video_chunk_sizes, end_of_video, video_chunk_remain = net_env.get_video_chunk(bit_rate)

        time_stamp += delay
        time_stamp += sleep_time

        reward = VIDEO_BIT_RATE[bit_rate] / M_IN_K - REBUF_PENALTY * rebuf - SMOOTH_PENALTY * np.abs(VIDEO_BIT_RATE[bit_rate] - VIDEO_BIT_RATE[last_bit_rate]) / M_IN_K
        r_batch.append(reward)
        last_bit_rate = bit_rate

        log_file.write(
            str(time_stamp / M_IN_K)
            + "\t"
            + str(VIDEO_BIT_RATE[bit_rate])
            + "\t"
            + str(buffer_size)
            + "\t"
            + str(rebuf)
            + "\t"
            + str(video_chunk_size)
            + "\t"
            + str(delay)
            + "\t"
            + str(entropy_)
            + "\t"
            + str(reward)
            + "\n"
        )
        log_file.flush()

        state = np.array(s_batch[-1], copy=True) if len(s_batch) else np.zeros((S_INFO, S_LEN))
        state = np.roll(state, -1, axis=1)
        state[0, -1] = VIDEO_BIT_RATE[bit_rate] / float(np.max(VIDEO_BIT_RATE))
        state[1, -1] = buffer_size / BUFFER_NORM_FACTOR
        state[2, -1] = float(video_chunk_size) / float(delay) / M_IN_K
        state[3, -1] = float(delay) / M_IN_K / BUFFER_NORM_FACTOR
        state[4, :A_DIM] = np.array(next_video_chunk_sizes) / M_IN_K / M_IN_K
        state[5, -1] = np.minimum(video_chunk_remain, CHUNK_TIL_VIDEO_END_CAP) / float(CHUNK_TIL_VIDEO_END_CAP)

        action_prob = actor.predict(np.reshape(state, (1, S_INFO, S_LEN)))
        bit_rate = select_bitrate(
            policy=args.policy,
            action_prob=action_prob,
            state=state,
            next_video_chunk_sizes=next_video_chunk_sizes,
            buffer_size=buffer_size,
            last_bit_rate=last_bit_rate,
            buffer_reserve=args.buffer_reserve,
            min_safety_budget=args.min_safety_budget,
            max_upstep=args.max_upstep,
        )

        s_batch.append(state)
        entropy_ = -np.dot(action_prob, np.log(action_prob))
        entropy_record.append(entropy_)

        if end_of_video:
            log_file.write("\n")
            log_file.close()

            last_bit_rate = DEFAULT_QUALITY
            bit_rate = DEFAULT_QUALITY

            del s_batch[:]
            del a_batch[:]
            del r_batch[:]

            action_vec = np.zeros(A_DIM)
            action_vec[bit_rate] = 1

            s_batch.append(np.zeros((S_INFO, S_LEN)))
            a_batch.append(action_vec)
            entropy_record = []

            video_count += 1
            if video_count >= len(all_file_names):
                break

            log_path = build_log_path(output_dir, all_file_names[net_env.trace_idx])
            log_file = log_path.open("w", encoding="utf-8")

    print(f"Saved evaluation logs to: {output_dir}")


if __name__ == "__main__":
    main()
