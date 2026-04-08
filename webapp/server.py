from __future__ import annotations

import os
import uuid
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory

from agent_runtime import (
    AgentRegistry,
    QUALITY_LADDER,
    PROJECT_ROOT,
    SessionMetrics,
    clamp_quality,
    discover_last_trained_model,
    resolve_model_path,
    serialize_model_path,
)


BASE_DIR = Path(__file__).resolve().parent
FRONTEND_DIST = BASE_DIR / "frontend" / "dist"
LOCAL_FALLBACK_MODEL = PROJECT_ROOT / "src" / "ppo" / "nn_model_ep_500000.pth"
MODEL_HINT = discover_last_trained_model()
MODEL_PATH = resolve_model_path(MODEL_HINT, LOCAL_FALLBACK_MODEL) if MODEL_HINT else LOCAL_FALLBACK_MODEL

MATCHES = [
    {
        "id": "match-1",
        "title": "الحلقة الاولى كاملة - Big Buck Bunny",
        "league": "Big Buck Bunny",
        "time": "يعرض الأن",
        "channels": ["Spacetoon", "4K Arena"],
        "venue": "فيلم أنميشن قصير",
        "poster": "linear-gradient(135deg, #123b78, #0a1225 55%, #d6ac56)",
        "stream_url": "https://download.blender.org/peach/bigbuckbunny_movies/BigBuckBunny_640x360.m4v",
        "quality_streams": {
            "360p": "https://download.blender.org/peach/bigbuckbunny_movies/BigBuckBunny_320x180.mp4",
            "480p": "https://download.blender.org/peach/bigbuckbunny_movies/big_buck_bunny_480p_h264.mov",
            "720p": "https://download.blender.org/peach/bigbuckbunny_movies/big_buck_bunny_720p_h264.mov",
            "1080p": "https://download.blender.org/peach/bigbuckbunny_movies/big_buck_bunny_1080p_h264.mov",
        },
        "bandwidth_probe_url": "https://storage.googleapis.com/shaka-demo-assets/angel-one-hls/v-0576p-1400k-libx264-s1.mp4",
    },
    {
        "id": "match-2",
        "title": "كلاسيكو - برشلونة vs ريال مدريد",
        "league": "La Liga Classic",
        "time": "اليوم - 10:00 PM",
        "channels": ["Futbol HD", "Stadium One"],
        "venue": "كامب نو",
        "poster": "linear-gradient(135deg, #531d86, #101826 58%, #d7b364)",
        "stream_url": "https://devstreaming-cdn.apple.com/videos/streaming/examples/img_bipbop_adv_example_fmp4/master.m3u8",
    },
    {
        "id": "match-3",
        "title": "نهائي دوري الأبطال - بايرن ميونخ vs باريس سان جيرمان",
        "league": "Champions League Final",
        "time": "غداً - 09:45 PM",
        "channels": ["Champions TV", "Ultra Sports"],
        "venue": "أليانز أرينا",
        "poster": "linear-gradient(135deg, #7d1720, #111827 58%, #d9b76d)",
        "stream_url": "https://devstreaming-cdn.apple.com/videos/streaming/examples/img_bipbop_adv_example_ts/master.m3u8",
    },
    {
        "id": "match-4",
        "title": "ديربي ميلانو - إنتر vs ميلان",
        "league": "Serie A Derby",
        "time": "غداً - 11:15 PM",
        "channels": ["Derby Plus", "Football Prime"],
        "venue": "سان سيرو",
        "poster": "linear-gradient(135deg, #144b6d, #0f172a 55%, #c8a245)",
        "stream_url": "https://devstreaming-cdn.apple.com/videos/streaming/examples/img_bipbop_adv_example_fmp4/master.m3u8",
    },
]


def compute_qoe(
    current_idx: int,
    previous_idx: int | None,
    rebuffer_seconds: float,
    rebuf_penalty: float = 5.5,
    smooth_penalty: float = 0.8,
) -> float:
    quality_score = QUALITY_LADDER[current_idx]["score"]
    switch_penalty = smooth_penalty * abs(current_idx - previous_idx) if previous_idx is not None else 0.0
    rebuffer_penalty = rebuf_penalty * rebuffer_seconds
    return round(quality_score - switch_penalty - rebuffer_penalty, 3)


def quality_index(label: str | None, default_idx: int = 2) -> int:
    return next(
        (index for index, item in enumerate(QUALITY_LADDER) if item["label"] == label),
        clamp_quality(default_idx),
    )


def quality_profile(label: str | None, default_idx: int = 2) -> dict:
    return QUALITY_LADDER[quality_index(label, default_idx)]


def describe_network_state(network_speed: float) -> str:
    if network_speed > 5:
        return f"سرعة الشبكة قوية ({network_speed:.1f} Mbps)"
    if network_speed >= 2:
        return f"سرعة الشبكة متوسطة ({network_speed:.1f} Mbps)"
    return f"سرعة الشبكة ضعيفة ({network_speed:.1f} Mbps)"


def describe_buffer_state(buffer_size: float) -> str:
    if buffer_size < 3:
        return f"المخزن المؤقت منخفض ({buffer_size:.1f}s)"
    if buffer_size > 10:
        return f"المخزن المؤقت مرتفع ({buffer_size:.1f}s)"
    return f"المخزن المؤقت مستقر ({buffer_size:.1f}s)"


def describe_quality_change(previous_quality: str | None, next_quality: str) -> str:
    if not previous_quality or previous_quality == next_quality:
        return f"تم الإبقاء على الجودة عند {next_quality}"

    previous_idx = quality_index(previous_quality)
    next_idx = quality_index(next_quality)
    if next_idx > previous_idx:
        return f"تم رفع الجودة من {previous_quality} إلى {next_quality}"
    if next_idx < previous_idx:
        return f"تم خفض الجودة من {previous_quality} إلى {next_quality}"
    return f"تم الإبقاء على الجودة عند {next_quality}"


def build_decision_reason(
    *,
    controller_type: str,
    previous_quality: str | None,
    target_quality: str,
    network_speed: float,
    buffer_size: float,
    rebuffer_seconds: float,
    manual_ceiling_applied: bool,
    manual_quality: str,
    model_reason: str,
) -> str:
    parts = [
        describe_quality_change(previous_quality, target_quality),
        f"لأن {describe_network_state(network_speed)} و{describe_buffer_state(buffer_size)}.",
    ]

    if rebuffer_seconds > 0:
        parts.append(f"تم رصد توقف مؤقت مقداره {rebuffer_seconds:.2f}s لذا أُعطيت الأولوية للاستقرار.")

    if manual_ceiling_applied:
        parts.append(f"كما تم منع الرفع فوق {manual_quality} بسبب Manual Quality.")

    if controller_type == "real":
        parts.append(f"هذا القرار صادر من النموذج الحقيقي. {model_reason}")

    return " ".join(parts)


app = Flask(__name__, static_folder=None)
registry = AgentRegistry(MODEL_PATH)
sessions: dict[str, SessionMetrics] = {}


@app.get("/")
def index():
    return serve_frontend("index.html")


@app.get("/<path:path>")
def frontend_assets(path: str):
    if path.startswith("api/"):
        return jsonify({"error": "Not found"}), 404
    return serve_frontend(path)


def serve_frontend(path: str):
    if FRONTEND_DIST.exists():
        target = FRONTEND_DIST / path
        if path != "index.html" and target.exists():
            return send_from_directory(FRONTEND_DIST, path)
        return send_from_directory(FRONTEND_DIST, "index.html")

    templates_dir = BASE_DIR / "templates"
    fallback_file = templates_dir / "index.html"
    if fallback_file.exists():
        return send_from_directory(templates_dir, "index.html")

    return jsonify({"error": "Frontend build not found"}), 503


@app.get("/api/bootstrap")
def bootstrap():
    return jsonify(
        {
            "matches": MATCHES,
            "quality_ladder": QUALITY_LADDER,
            **registry.bootstrap_payload(),
        }
    )


@app.get("/api/model-catalog")
def model_catalog():
    return jsonify(registry.model_catalog_payload())


@app.post("/api/validate-model")
def validate_model():
    payload = request.get_json(silent=True) or {}
    controller_type = payload.get("controller_type", "simulated")
    model_path = payload.get("model_path")
    resolved_model_path = resolve_model_path(model_path, MODEL_PATH)
    requested_model_path = model_path
    if model_path and not resolved_model_path.exists():
        return jsonify(
            {
                "valid": False,
                "controller_type": controller_type,
                "model_path": requested_model_path,
                "resolved_model_path": str(resolved_model_path),
                "message": f"Model file not found: {requested_model_path}",
            }
        ), 400

    resolved_relative_path = serialize_model_path(resolved_model_path) if resolved_model_path.exists() else requested_model_path
    agent = registry.get_agent(controller_type, resolved_relative_path)
    valid = bool(getattr(agent, "model_loaded", False) or controller_type == "simulated")
    message = getattr(agent, "model_summary", "Validation completed.")
    if controller_type == "simulated" and not valid:
        valid = True

    status_code = 200 if valid else 400
    return jsonify(
        {
            "valid": valid,
            "controller_type": controller_type,
            "controller_label": getattr(agent, "controller_label", controller_type),
            "model_path": serialize_model_path(agent.model_path) if getattr(agent, "model_path", None) and agent.model_path.exists() else requested_model_path,
            "resolved_model_path": resolved_relative_path,
            "message": message,
        }
    ), status_code


@app.post("/api/session")
def create_session():
    payload = request.get_json(silent=True) or {}
    match_id = payload.get("match_id")
    mode = payload.get("mode", "ai")
    controller_type = payload.get("controller_type", "simulated")
    model_path = payload.get("model_path")
    requested_model_path = model_path

    if match_id not in {match["id"] for match in MATCHES}:
        return jsonify({"error": "Unknown match id"}), 400

    resolved_model_path = resolve_model_path(model_path, MODEL_PATH)
    if model_path and not resolved_model_path.exists():
        return jsonify({"error": f"Model file not found: {requested_model_path}"}), 400

    resolved_relative_path = serialize_model_path(resolved_model_path) if resolved_model_path.exists() else requested_model_path
    agent = registry.get_agent(controller_type, resolved_relative_path)
    if controller_type == "real" and not getattr(agent, "model_loaded", False):
        return jsonify({"error": getattr(agent, "model_summary", "Failed to load real model.")}), 400

    session_id = uuid.uuid4().hex
    runtime_state = agent.initial_session_state() if hasattr(agent, "initial_session_state") else {}
    session = SessionMetrics(
        session_id=session_id,
        match_id=match_id,
        mode=mode,
        controller_type=controller_type,
        controller_label=getattr(agent, "controller_label", controller_type),
        model_path=serialize_model_path(agent.model_path) if getattr(agent, "model_path", None) else resolved_relative_path,
        current_quality="720p",
        runtime_state=runtime_state,
    )
    sessions[session_id] = session
    return jsonify(
        {
            "session_id": session_id,
            "mode": mode,
            "current_quality": session.current_quality,
            "controller_type": session.controller_type,
            "controller_label": session.controller_label,
            "model_path": session.model_path,
        }
    )


@app.post("/api/session/<session_id>/decision")
def get_decision(session_id: str):
    session = sessions.get(session_id)
    if not session:
        return jsonify({"error": "Session not found"}), 404

    payload = request.get_json(silent=True) or {}
    network_speed = float(payload.get("network_speed", 3.0))
    buffer_size = float(payload.get("buffer_size", 6.0))
    current_quality = payload.get("current_quality", session.current_quality)
    manual_quality = payload.get("manual_quality", current_quality)
    rebuffer_seconds = max(0.0, float(payload.get("rebuffer_seconds", 0.0)))
    rebuf_penalty = float(payload.get("rebuf_penalty", 5.5))
    smooth_penalty = float(payload.get("smooth_penalty", 0.8))

    agent = registry.get_agent(session.controller_type, session.model_path)
    try:
        recommendation = agent.recommend(
            network_speed=network_speed,
            buffer_size=buffer_size,
            current_quality=current_quality,
            session_state=session.runtime_state,
        )
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400

    recommended_idx = quality_index(recommendation["quality"])
    manual_ceiling_idx = quality_index(manual_quality)
    capped_idx = min(recommended_idx, manual_ceiling_idx)
    ai_quality = QUALITY_LADDER[capped_idx]["label"]
    ai_profile = QUALITY_LADDER[capped_idx]

    applied_quality = manual_quality if session.mode == "manual" else ai_quality
    previous_idx = next(
        (i for i, item in enumerate(QUALITY_LADDER) if item["label"] == session.current_quality),
        None,
    )
    applied_idx = quality_index(applied_quality)
    qoe = compute_qoe(
        applied_idx,
        previous_idx,
        rebuffer_seconds,
        rebuf_penalty=rebuf_penalty,
        smooth_penalty=smooth_penalty,
    )

    manual_ceiling_applied = capped_idx != recommended_idx
    reason = build_decision_reason(
        controller_type=session.controller_type,
        previous_quality=session.current_quality,
        target_quality=ai_quality,
        network_speed=network_speed,
        buffer_size=buffer_size,
        rebuffer_seconds=rebuffer_seconds,
        manual_ceiling_applied=manual_ceiling_applied,
        manual_quality=manual_quality,
        model_reason=recommendation["reason"],
    )

    session.log_decision(
        network_speed=network_speed,
        buffer_size=buffer_size,
        applied_quality=applied_quality,
        ai_quality=ai_quality,
        rebuffer_seconds=rebuffer_seconds,
        qoe=qoe,
    )

    return jsonify(
        {
            "session_id": session_id,
            "mode": session.mode,
            "controller_type": session.controller_type,
            "controller_label": session.controller_label,
            "model_path": session.model_path,
            "recommended_quality": ai_quality,
            "applied_quality": applied_quality,
            "recommended_bitrate_mbps": ai_profile["bitrate_mbps"],
            "resolution": ai_profile["resolution"],
            "reason": reason,
            "qoe": qoe,
            "qoe_settings": {
                "rebuf_penalty": rebuf_penalty,
                "smooth_penalty": smooth_penalty,
            },
            "action_probabilities": recommendation.get("action_probabilities", []),
            "summary": session.summary(),
        }
    )


@app.get("/api/session/<session_id>/stats")
def get_stats(session_id: str):
    session = sessions.get(session_id)
    if not session:
        return jsonify({"error": "Session not found"}), 404
    return jsonify(
        {
            "session": {
                "session_id": session.session_id,
                "match_id": session.match_id,
                "mode": session.mode,
                "controller_type": session.controller_type,
                "controller_label": session.controller_label,
                "model_path": session.model_path,
                "current_quality": session.current_quality,
                "decisions": session.decisions,
                "total_rebuffer_seconds": session.total_rebuffer_seconds,
                "rebuffer_events": session.rebuffer_events,
            },
            "summary": session.summary(),
        }
    )


@app.post("/api/session/<session_id>/end")
def end_session(session_id: str):
    session = sessions.get(session_id)
    if not session:
        return jsonify({"error": "Session not found"}), 404
    return jsonify({"summary": session.summary()})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "8000")), debug=True)
