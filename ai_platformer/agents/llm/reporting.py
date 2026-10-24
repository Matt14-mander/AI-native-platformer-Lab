"""Read-only consolidation of batch and ad-hoc evidence without pooling cohorts."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path

from ai_platformer.agents.llm.evaluation import sha256_file
from ai_platformer.agents.llm.generation import digest, read_json
from ai_platformer.content.generation_request import GenerationRequest, request_hash
from ai_platformer.content.generation_suite import compare_expectations, load_suite
from ai_platformer.content.level_spec import geometry_hash, load_level_spec

TOKEN_KEYS = ("promptTokenCount", "candidatesTokenCount", "totalTokenCount", "thoughtsTokenCount")


def fraction(numerator, denominator):
    return {
        "numerator": numerator,
        "denominator": denominator,
        "rate": numerator / denominator if denominator else None,
    }


def distribution(values):
    values = sorted(
        value
        for value in values
        if isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and value >= 0
    )

    def percentile(fraction):
        position = (len(values) - 1) * fraction
        lower = math.floor(position)
        upper = math.ceil(position)
        return values[lower] + (values[upper] - values[lower]) * (position - lower)

    return {
        "samples": len(values),
        "mean": sum(values) / len(values) if values else None,
        "p50": percentile(0.5) if values else None,
        "p95": percentile(0.95) if values else None,
        "max": max(values) if values else None,
    }


def api_metrics(cases, mode):
    attempts = outputs = unknown = cached = 0
    errors = Counter()
    token_totals = Counter()
    token_samples = Counter()
    usage_attempts = 0
    latencies, valid_latencies = [], []
    for case in cases:
        translation = case.get("translation", {})
        if mode != "gemini":
            continue
        if translation.get("cache_hit"):
            cached += 1
            continue
        duration = translation.get("seconds")
        if duration is not None:
            latencies.append(duration)
            if translation.get("status") == "ok":
                valid_latencies.append(duration)
        # Old logs may contain both an output and a validation error for one attempt.
        groups = defaultdict(list)
        for record in case.get("attempt_records", []):
            if type(record.get("attempt")) is int and record["attempt"] > 0:
                groups[record["attempt"]].append(record)
        for records in groups.values():
            attempts += 1
            if any("output" in record for record in records):
                outputs += 1
            elif not any(
                record.get("http_status")
                or record.get("connection_category")
                or record.get("error") == "provider connection failed"
                or str(record.get("error", "")).startswith("provider HTTP ")
                for record in records
            ):
                unknown += 1
            for record in records:
                if record.get("http_status"):
                    errors[f"http_{record['http_status']}"] += 1
                elif record.get("connection_category"):
                    errors[record["connection_category"]] += 1
                elif record.get("error") == "provider connection failed":
                    errors["connection_unspecified"] += 1
            usages = [
                record["usage"] for record in records if isinstance(record.get("usage"), dict)
            ]
            if usages:
                usage_attempts += 1
                for key in TOKEN_KEYS:
                    value = usages[-1].get(key)
                    if type(value) is int and value >= 0:
                        token_totals[key] += value
                        token_samples[key] += 1
    return {
        "applicable": mode == "gemini",
        "cache_hits": cached,
        "recorded_uncached_attempts": attempts,
        "observed_json_outputs": outputs,
        "attempts_with_unknown_response_outcome": unknown,
        "observed_json_response_rate_lower_bound": fraction(outputs, attempts),
        "transport_error_events": dict(errors),
        "translation_latency_seconds": distribution(latencies),
        "valid_translation_latency_seconds": distribution(valid_latencies),
        "reported_tokens": {
            key: token_totals[key] if token_samples[key] else None for key in TOKEN_KEYS
        },
        "token_field_sample_counts": {key: token_samples[key] for key in TOKEN_KEYS},
        "usage_reported_attempts": usage_attempts,
        "billed_cost_usd": None,
    }


def case_failures(case):
    failures = []
    if case.get("failure"):
        failures.append(
            {
                "category": case["failure"].get("stage", "pipeline") + "_error",
                "reason": case["failure"].get("reason"),
            }
        )
    if case.get("semantics", {}).get("passed") is False:
        failures.append(
            {"category": "semantic_mismatch", "mismatches": case["semantics"].get("mismatches")}
        )
    if case.get("route", {}).get("status") in {"unknown", "error"}:
        failures.append(
            {"category": "route_" + case["route"]["status"], "reason": case["route"].get("reason")}
        )
    ppo = case.get("ppo", {})
    if ppo.get("status") == "error":
        failures.append({"category": "ppo_error", "reason": ppo.get("reason")})
    elif ppo.get("status") == "ok":
        if ppo.get("outcome") != "success":
            failures.append({"category": "ppo_non_success", "outcome": ppo.get("outcome")})
        elif ppo.get("meets_expected_target") is False:
            failures.append(
                {
                    "category": "ppo_collection_miss",
                    "collected": ppo.get("coins_collected"),
                    "total": ppo.get("coins_total"),
                    "required_ratio": case.get("expected_target"),
                    "final_player": case.get("final_player"),
                }
            )
    return failures


def cohort_summary(cases, planned, mode, *, expectations_planned):
    valid = sum(case.get("translation", {}).get("status") == "ok" for case in cases)
    asserted = sum(type(case.get("semantics", {}).get("passed")) is bool for case in cases)
    matched = sum(case.get("semantics", {}).get("passed") is True for case in cases)
    static = sum(case.get("generation", {}).get("status") == "valid" for case in cases)
    verified = sum(case.get("route", {}).get("status") == "verified" for case in cases)
    evaluated = [case["ppo"] for case in cases if case.get("ppo", {}).get("status") == "ok"]
    succeeded = sum(episode.get("outcome") == "success" for episode in evaluated)
    met = sum(episode.get("meets_expected_target") is True for episode in evaluated)
    coin_episodes = [episode for episode in evaluated if episode.get("coins_total", 0) > 0]
    collected = sum(episode["coins_collected"] for episode in coin_episodes)
    total = sum(episode["coins_total"] for episode in coin_episodes)
    categories = Counter(item["category"] for case in cases for item in case_failures(case))
    return {
        "planned": planned,
        "processed": len(cases),
        "pending": planned - len(cases),
        "valid_requests_over_planned": fraction(valid, planned),
        "llm_semantic_matches_over_planned": fraction(matched, expectations_planned)
        if mode == "gemini" and expectations_planned
        else None,
        "design_matches_over_asserted": fraction(matched, asserted),
        "static_valid_over_planned": fraction(static, planned),
        "route_verified_over_planned": fraction(verified, planned),
        "ppo_evaluated": len(evaluated),
        "ppo_success_over_planned": fraction(succeeded, planned),
        "ppo_success_over_evaluated": fraction(succeeded, len(evaluated)),
        "ppo_target_over_evaluated": fraction(met, len(evaluated)),
        "collection_ratio_nonempty_maps": distribution(
            [e["coins_collected"] / e["coins_total"] for e in coin_episodes]
        ),
        "weighted_coin_collection": fraction(collected, total),
        "failure_categories": dict(categories),
        "api": api_metrics(cases, mode),
    }


def load_attempts(folder):
    path = folder / "llm.json"
    return read_json(path) if path.exists() else {}


def load_batch(folder):
    folder = Path(folder).resolve()
    run, report = read_json(folder / "run.json"), read_json(folder / "report.json")
    identity = run["identity"]
    if digest(identity) != run["run_sha256"] or report["run_sha256"] != run["run_sha256"]:
        raise ValueError("batch run/report identity mismatch")
    suite = load_suite(folder / "suite.json")
    if suite.model_dump() != identity["suite"] or report["suite_id"] != suite.suite_id:
        raise ValueError("batch suite snapshot changed")
    if report["mode"] != identity["settings"]["provider"]:
        raise ValueError("batch provider mismatch")
    planned = [(case, seed) for case in suite.cases for seed in suite.generation_seeds]
    if identity.get("limit") is not None:
        planned = planned[: identity["limit"]]
    index = {case.expected_request(seed).level_id: (case, seed) for case, seed in planned}
    cases, seen = [], set()
    for original in report["cases"]:
        level_id = original["level_id"]
        if level_id not in index or level_id in seen:
            raise ValueError("unexpected or duplicate candidate ID")
        seen.add(level_id)
        package = folder / "cases" / level_id
        row = read_json(package / "result.json")
        if row != original or row["run_sha256"] != run["run_sha256"]:
            raise ValueError("batch candidate result changed")
        for name, expected_hash in row["artifact_sha256"].items():
            if (
                Path(name).name != name
                or not name.endswith(".json")
                or sha256_file(package / name) != expected_hash
            ):
                raise ValueError("batch candidate artifact changed")
        case, seed = index[level_id]
        if row["generation_seed"] != seed or row["case_id"] != case.case_id:
            raise ValueError("candidate prompt/seed mismatch")
        normalized = dict(row)
        normalized["expected_target"] = case.expected_request(seed).min_coin_ratio
        normalized["package"] = str(package)
        llm = load_attempts(package)
        normalized["attempt_records"] = llm.get("attempts", [])
        if (package / "request.json").exists():
            request = GenerationRequest.model_validate(read_json(package / "request.json"))
            if request.level_id != level_id or request.seed != seed:
                raise ValueError("request caller-owned fields mismatch")
            if request_hash(request) != row.get("generation", {}).get("request_sha256"):
                raise ValueError("batch request hash mismatch")
            spec = load_level_spec(package / "level.json")
            if geometry_hash(spec) != row.get("generation", {}).get("geometry_sha256"):
                raise ValueError("batch geometry hash mismatch")
            semantics = compare_expectations(request, case.expected)
            if semantics != row.get("semantics"):
                raise ValueError("batch semantic result mismatch")
        if (package / "ppo.json").exists():
            ppo = read_json(package / "ppo.json")
            for key, value in row.get("ppo", {}).items():
                if key != "seconds" and value != ppo.get(key):
                    raise ValueError("batch PPO summary mismatch")
            normalized["final_player"] = ppo.get("final_state", {}).get("player")
        cases.append(normalized)
    mode = report["mode"]
    if mode not in {"gemini", "expected-fixture"}:
        raise ValueError("unsupported batch mode")
    source = {
        "kind": "batch",
        "path": str(folder),
        "source_id": run["run_sha256"],
        "suite_id": suite.suite_id,
        "suite_sha256": digest(identity["suite"]),
        "prompt_definitions": [case.model_dump() for case in suite.cases],
        "mode": mode,
        "model": identity["settings"]["model"],
        "tls_mode": identity["settings"].get("tls_mode", "default"),
        "checkpoint_sha256": identity["checkpoint_sha256"],
        "policy_seed": suite.policy_seed,
        "input_sha256": {
            name: sha256_file(folder / name) for name in ("run.json", "report.json", "suite.json")
        },
        "cases": cases,
        "pending_level_ids": sorted(set(index) - seen),
    }
    source["summary"] = cohort_summary(cases, len(planned), mode, expectations_planned=len(planned))
    source["by_prompt"] = {
        case.case_id: cohort_summary(
            [row for row in cases if row["case_id"] == case.case_id],
            sum(c.case_id == case.case_id for c, _ in planned),
            mode,
            expectations_planned=sum(c.case_id == case.case_id for c, _ in planned),
        )
        for case in suite.cases
        if any(c.case_id == case.case_id for c, _ in planned)
    }
    source["by_generation_seed"] = {
        str(seed): cohort_summary(
            [row for row in cases if row["generation_seed"] == seed],
            sum(s == seed for _, s in planned),
            mode,
            expectations_planned=sum(s == seed for _, s in planned),
        )
        for seed in suite.generation_seeds
        if any(s == seed for _, s in planned)
    }
    return source


def load_single(folder, *, ppo=None, checkpoint_sha256=None, policy_seed=100):
    folder = Path(folder).resolve()
    report, intent = read_json(folder / "report.json"), read_json(folder / "intent.json")
    llm = load_attempts(folder)
    mode = llm.get("identity", {}).get("provider", intent.get("provider"))
    if mode not in {"gemini", "fixture"}:
        raise ValueError("unsupported single-package provider")
    mode = "gemini" if mode == "gemini" else "expected-fixture"
    case = {
        "case_id": "single_candidate",
        "level_id": intent["level_id"],
        "generation_seed": intent["seed"],
        "policy_seed": policy_seed,
        "package": str(folder),
        "semantics": {},
        "attempt_records": llm.get("attempts", []),
        "translation": {
            "status": "ok" if llm.get("request_sha256") else "error",
            "cache_hit": llm.get("cache_hit", False),
        },
    }
    if (folder / "request.json").exists():
        request = GenerationRequest.model_validate(read_json(folder / "request.json"))
        if request_hash(request) != llm.get("request_sha256"):
            raise ValueError("single request hash mismatch")
        spec = load_level_spec(folder / "level.json")
        case["generation"] = {"status": "valid", "geometry_sha256": geometry_hash(spec)}
        case["expected_target"] = request.min_coin_ratio
    if (folder / "route.json").exists():
        route = read_json(folder / "route.json")
        case["route"] = {"status": route["status"], "reason": route.get("reason")}
    if ppo:
        case["ppo"] = {
            key: value for key, value in ppo.items() if key not in {"actions", "final_state"}
        }
        case["final_player"] = ppo.get("final_state", {}).get("player")
    if report["status"] in {"invalid", "error"}:
        case["failure"] = {
            "stage": "single_pipeline",
            "reason": report.get("reason", report["status"]),
        }
    names = [
        name
        for name in (
            "report.json",
            "intent.json",
            "llm.json",
            "request.json",
            "level.json",
            "route.json",
        )
        if (folder / name).exists()
    ]
    hashes = {name: sha256_file(folder / name) for name in names}
    source = {
        "kind": "single",
        "path": str(folder),
        "source_id": digest(hashes),
        "suite_id": None,
        "mode": mode,
        "model": llm.get("identity", {}).get("model", intent.get("model")),
        "tls_mode": llm.get("identity", {}).get("tls_mode", intent.get("tls_mode", "default")),
        "checkpoint_sha256": checkpoint_sha256,
        "policy_seed": policy_seed,
        "input_sha256": hashes,
        "cases": [case],
        "by_prompt": {},
        "pending_level_ids": [],
        "semantics_note": "No frozen expectation assertions in this single package; semantic accuracy is unscored.",
    }
    source["summary"] = cohort_summary([case], 1, mode, expectations_planned=0)
    return source


def build_report(sources):
    sources = deepcopy(sources)
    ids = [(source["kind"], source["path"]) for source in sources]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate input source")
    failures, geometry = [], defaultdict(list)
    for source in sources:
        for case in source["cases"]:
            reference = {
                "source_id": source["source_id"],
                "mode": source["mode"],
                "kind": source["kind"],
                "level_id": case["level_id"],
                "package": case["package"],
            }
            for failure in case_failures(case):
                failures.append({**reference, **failure})
            value = case.get("generation", {}).get("geometry_sha256")
            if value:
                geometry[value].append(reference)
    # Raw model responses are audit inputs, not part of the consolidated output.
    for source in sources:
        for case in source["cases"]:
            case.pop("attempt_records", None)
    return {
        "schema_version": 1,
        "report_kind": "unified_generation_evaluation_v1",
        "sources": sources,
        "evidence_counts": {
            "live_batches": sum(s["kind"] == "batch" and s["mode"] == "gemini" for s in sources),
            "live_single_packages": sum(
                s["kind"] == "single" and s["mode"] == "gemini" for s in sources
            ),
            "offline_batches": sum(
                s["kind"] == "batch" and s["mode"] == "expected-fixture" for s in sources
            ),
        },
        "failure_catalog": failures,
        "geometry_audit": {
            "unique_geometry_count": len(geometry),
            "duplicate_groups": [group for group in geometry.values() if len(group) > 1],
        },
        "billed_cost_usd": None,
        "notes": [
            "离线、真实批次、单图和重复运行不合并计算总成功率。",
            "API JSON输出观测率仅为已记录尝试的下界，不代表HTTP精确成功率或服务可用性。",
            "缺失的历史耗时、用量和语义断言保留为未知；只覆盖所选来源，不等于全部调试请求或账单。",
            "失败类型可以重叠；路线unknown不代表不可通关。",
            "单图PPO目标来自生成请求；没有冻结需求断言时，不推断其LLM语义正确率。",
            "本报告限于探索性的静态生成族；未改变训练/保留集，也未重新训练模型。",
        ],
    }


def markdown(report):
    def text(value):
        return str(value).replace("|", "\\|").replace("\n", " ").replace("\r", " ")

    def rate(value):
        if not value or value["rate"] is None:
            return "未评估"
        return f"{value['numerator']}/{value['denominator']} ({value['rate']:.1%})"

    def percent(value):
        return f"{value:.2%}" if value is not None else "未评估"

    def number(value):
        return f"{value:.3f}" if value is not None else "未记录"

    lines = [
        "# 生成关卡统一批量报告",
        "",
        "离线覆盖、真实API批次和真实单图分开展示，不合并为总成功率。",
        f"真实API批次：{report['evidence_counts']['live_batches']}；真实单图：{report['evidence_counts']['live_single_packages']}；离线批次：{report['evidence_counts']['offline_batches']}。",
        "",
        "| 类型 | 模型 | 已处理/计划 | 请求合法 | LLM需求匹配 | 路线verified | PPO通关/已评估 | PPO收集达标/已评估 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for source in report["sources"]:
        summary = source["summary"]
        kind = ("真实API" if source["mode"] == "gemini" else "离线fixture") + (
            "批次" if source["kind"] == "batch" else "单图"
        )
        lines.append(
            f"| {kind} | {text(source['model'])} | {summary['processed']}/{summary['planned']} | {rate(summary['valid_requests_over_planned'])} | {rate(summary['llm_semantic_matches_over_planned'])} | {rate(summary['route_verified_over_planned'])} | {rate(summary['ppo_success_over_evaluated'])} | {rate(summary['ppo_target_over_evaluated'])} |"
        )
    for source in report["sources"]:
        summary, api = source["summary"], source["summary"]["api"]
        lines.extend(
            [
                "",
                f"## {text(source['path'])}",
                "",
                f"模式：{source['mode']}；TLS：{source['tls_mode']}；策略seed：{source['policy_seed']}。",
                "",
                f"Checkpoint SHA256：`{source['checkpoint_sha256'] or '未绑定'}`",
                "",
                f"收集率均值（有松果地图）：{percent(summary['collection_ratio_nonempty_maps']['mean'])}；样本数：{summary['collection_ratio_nonempty_maps']['samples']}；按松果加权：{rate(summary['weighted_coin_collection'])}。",
                "",
                f"已记录未缓存API尝试：{api['recorded_uncached_attempts']}；观测到JSON输出：{api['observed_json_outputs']}；响应结果不明尝试：{api['attempts_with_unknown_response_outcome']}；缓存命中：{api['cache_hits']}；JSON输出观测率下界：{rate(api['observed_json_response_rate_lower_bound'])}。",
                "",
                f"翻译耗时（秒，含校验/重试；不含路线和PPO）：均值 {number(api['translation_latency_seconds']['mean'])}，P50 {number(api['translation_latency_seconds']['p50'])}，P95 {number(api['translation_latency_seconds']['p95'])}，样本数 {api['translation_latency_seconds']['samples']}。",
                "",
                f"报告的total tokens：{api['reported_tokens']['totalTokenCount'] if api['reported_tokens']['totalTokenCount'] is not None else '未记录'}；费用：未估算，失败调用消耗可能未报告。",
                "",
            ]
        )
        if source["by_prompt"]:
            lines += [
                "| 提示 | 已处理/计划 | 需求匹配/已断言 | 路线verified | PPO通关 | PPO收集达标 |",
                "| --- | --- | --- | --- | --- | --- |",
            ]
            for name, group in source["by_prompt"].items():
                lines.append(
                    f"| {text(name)} | {group['processed']}/{group['planned']} | {rate(group['design_matches_over_asserted'])} | {rate(group['route_verified_over_planned'])} | {rate(group['ppo_success_over_evaluated'])} | {rate(group['ppo_target_over_evaluated'])} |"
                )
        if source.get("by_generation_seed"):
            lines += [
                "",
                "| 生成seed | 已处理/计划 | 路线verified | PPO通关 | PPO收集达标 | 收集率均值 |",
                "| --- | --- | --- | --- | --- | --- |",
            ]
            for seed, group in source["by_generation_seed"].items():
                lines.append(
                    f"| {seed} | {group['processed']}/{group['planned']} | {rate(group['route_verified_over_planned'])} | {rate(group['ppo_success_over_evaluated'])} | {rate(group['ppo_target_over_evaluated'])} | {percent(group['collection_ratio_nonempty_maps']['mean'])} |"
                )
        lines += [
            "",
            "| 地图 | 生成seed | 需求匹配 | 路线 | PPO结果 | 松果 | 目标达标 |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
        for case in source["cases"]:
            ppo = case.get("ppo", {})
            lines.append(
                f"| {text(case['level_id'])} | {case['generation_seed']} | {case.get('semantics', {}).get('passed', '未断言')} | {case.get('route', {}).get('status', '未执行')} | {ppo.get('outcome', '未执行')} | {ppo.get('coins_collected', '—')}/{ppo.get('coins_total', '—')} | {ppo.get('meets_expected_target', '未评估')} |"
            )
        if source["pending_level_ids"]:
            lines += ["", "未执行地图：" + ", ".join(source["pending_level_ids"])]
    lines += ["", "## 失败目录", "", "| 地图 | 类型 | 原因/目标 |", "| --- | --- | --- |"]
    for failure in report["failure_catalog"]:
        reason = (
            failure.get("reason")
            or failure.get("outcome")
            or str(
                failure.get("mismatches") or f"最低收集率 {failure.get('required_ratio', '未知')}"
            )
        )
        lines.append(f"| {text(failure['level_id'])} | {failure['category']} | {text(reason)} |")
    lines += [
        "",
        f"跨来源不同几何：{report['geometry_audit']['unique_geometry_count']}；重复组：{len(report['geometry_audit']['duplicate_groups'])}。重复运行不作为新增独立样本。",
        "",
        "## 统计边界",
        "",
    ]
    lines += [f"- {note}" for note in report["notes"]]
    return "\n".join(lines) + "\n"
