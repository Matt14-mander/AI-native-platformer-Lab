"""Read-only source audits and deterministic metrics/profile report construction."""

import hashlib
from pathlib import Path

from .metrics import build_metrics, digest
from .profile import build_profiles
from .telemetry import read_session


def source_hashes(path):
    return {name: hashlib.sha256((path / name).read_bytes()).hexdigest()
            for name in ("manifest.json", "events.jsonl")}


def build_report(paths, *, window=10):
    paths = [Path(path).resolve() for path in paths]
    if not 1 <= len(paths) <= 128 or len(set(paths)) != len(paths):
        raise ValueError("provide 1..128 distinct telemetry session directories")
    before = [source_hashes(path) for path in paths]
    sessions = [read_session(path) for path in paths]
    metrics = build_metrics(sessions)
    profiles = build_profiles(metrics, window=window)
    sources = [{"path": str(path), "sha256": hashes,
                "session_id": session["manifest"]["session_id"],
                "actor_type": session["manifest"]["actor_type"],
                "complete": session["complete"]}
               for path, hashes, session in zip(paths, before, sessions, strict=True)]
    if before != [source_hashes(path) for path in paths]:
        raise ValueError("source session changed during analysis; retry after recording closes")
    return {
        "schema_version": 1, "kind": "play_metrics_and_profiles_v1",
        "build_id": digest({"sources": [s["sha256"] for s in sources],
                            "window": window, "metrics_version": 1, "profile_version": 1}),
        "sources": sources, "source_files_unchanged": True,
        "ordering": "--input order is oldest to newest; no filesystem timestamp inference",
        "metrics": metrics, "profiles": profiles,
    }


def markdown(report):
    def fmt(value):
        return "未知/不适用" if value is None else f"{value:.3f}"

    lines = [
        "# 游玩行为指标与玩家画像 v1", "",
        (f"输入 {len(report['sources'])} 个会话，{len(report['metrics']['attempts'])} 次尝试；"
        f"生成 {len(report['profiles'])} 份按身份与环境隔离的真人画像。"), "",
        ("通关率仅统计完整会话中自然结束的尝试；重开、退出和记录不完整不算死亡。"
        "各地图与协议分别展示，不作跨难度的技能排名。"), "",
        "## 分组行为指标", "",
        "| 玩家 / 来源 | 地图 | 播放模式 | 有效尝试 | 通关率 | 收集率 | 空中占比 | 回撤 px |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    # Escape untrusted local identifiers rather than allowing them to alter Markdown structure.
    def label(value):
        return str(value).replace("|", "\\|").replace("\n", " ").replace("\r", " ")

    for cohort in report["metrics"]["cohorts"]:
        summary = cohort["summary"]
        lines.append(
            f"| {label(cohort['identity']['id'])} / {cohort['actor_type']} "
            f"| {label(', '.join(cohort['level_ids']))} "
            f"| {cohort['playback_mode']['speed']}x / headless={cohort['playback_mode']['headless']} "
            f"| {summary['valid_terminal_attempts']} | {fmt(summary['success_rate'])} "
            f"| {fmt(summary['collection']['weighted_ratio'])} "
            f"| {fmt(summary['motion']['airborne_ratio'])} "
            f"| {fmt(summary['motion']['backward_distance_px'])} |"
        )
    lines += ["", "## 玩家画像", ""]
    if not report["profiles"]:
        lines += ["没有可建立真人画像的身份记录。模型和重放数据仍保留在分组指标中。", ""]
    for profile in report["profiles"]:
        gate = profile["readiness"]
        lines += [
            f"### {label(profile['identity']['id'])}（{profile['identity']['scope']}）", "",
            (f"状态：**{profile['status']}**；最近有效尝试 {gate['valid_attempts']} 次，"
            f"独立地图 {gate['independent_map_count']} 张。门槛为至少 5 次、2 张图，"
            "仅表示工程证据量，不是统计置信度。"), "",
            (f"环境：`{profile['environment_sha256']}`。窗口最多 "
            f"{profile['window']['maximum_attempts']} 次；输入顺序决定新旧。"), "",
            (f"运动数据：{profile['dimensions']['motion_behavior']['available_attempts']} 次可用，"
            f"{profile['dimensions']['motion_behavior']['missing_attempts']} 次缺失。"), "",
            ("收集偏好、跳跃错误率和战斗技能保留 unknown。完整逐地图数据、"
            "证据 attempt_id 与排除原因见同目录 report.json。"), "",
        ]
    lines += ["## 数据边界", "",
              ("旧日志缺少逐帧运动聚合时显示未知，不填 0。按跳请求与有效起跳分别计数，"
              "未松开的按跳时长视作截尾样本。回撤与停滞只是行为事实，不代表操作错误。"), "",
              ("重试仅在同一完整会话、同地图同协议的相邻尝试之间识别；"
              "没有跨会话时间证据时不推断重试。原日志只读，重复会话 ID 被拒绝。"), "",
              "本报告不调整难度，不调用 LLM 或 Jev；尚不能替代真实人工画像有效性验收。", ""]
    return "\n".join(lines)
