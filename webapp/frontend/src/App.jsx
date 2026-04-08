import { useEffect, useMemo, useRef, useState } from "react";
import MatchCard from "./components/MatchCard";
import MetricCard from "./components/MetricCard";
import Player from "./components/Player";
import QoEControls from "./components/QoEControls";

const emptySummary = {
  avg_quality: "N/A",
  total_rebuffer_seconds: 0,
  final_qoe: 0,
  rebuffer_events: 0,
  controller_label: "N/A",
  model_path: "",
};

export default function App() {
  const [matches, setMatches] = useState([]);
  const [qualityLadder, setQualityLadder] = useState([]);
  const [selectedMatch, setSelectedMatch] = useState(null);
  const [agentSummary, setAgentSummary] = useState("Loading agent info...");
  const [theme, setTheme] = useState("dark");
  const [mode, setMode] = useState("ai");
  const [manualQuality, setManualQuality] = useState("720p");
  const [sessionId, setSessionId] = useState(null);
  const [appliedQuality, setAppliedQuality] = useState("720p");
  const [aiQuality, setAiQuality] = useState("720p");
  const [networkSpeed, setNetworkSpeed] = useState(4.8);
  const [networkBias, setNetworkBias] = useState(0);
  const [signalStrength, setSignalStrength] = useState(85);
  const [signalOverrideEnabled, setSignalOverrideEnabled] = useState(false);
  const [networkTelemetrySource, setNetworkTelemetrySource] = useState("browser estimate");
  const [bufferSize, setBufferSize] = useState(6);
  const [qoe, setQoe] = useState(0);
  const [rebufferEvents, setRebufferEvents] = useState(0);
  const [summary, setSummary] = useState(emptySummary);
  const [recommendationReason, setRecommendationReason] = useState("Decision reason will appear here.");
  const [streamResolution, setStreamResolution] = useState("1280x720");
  const [sessionLog, setSessionLog] = useState([]);
  const [controllers, setControllers] = useState([]);
  const [controllerType, setControllerType] = useState("real");
  const [lastTrainedModel, setLastTrainedModel] = useState("");
  const [selectedModelOption, setSelectedModelOption] = useState("__last__");
  const [customModelPath, setCustomModelPath] = useState("");
  const [resolvedModelPath, setResolvedModelPath] = useState("");
  const [activeControllerLabel, setActiveControllerLabel] = useState("Real Pensieve PPO");
  const [activeModelPath, setActiveModelPath] = useState("");
  const [actionProbabilities, setActionProbabilities] = useState([]);
  const [errorMessage, setErrorMessage] = useState("");
  const [validationMessage, setValidationMessage] = useState("");
  const [validationState, setValidationState] = useState("idle");
  const [showQoEControls, setShowQoEControls] = useState(false);
  const [qoeControls, setQoeControls] = useState({
    rebufPenalty: 5.5,
    smoothPenalty: 0.8,
  });

  const timerRef = useRef(null);
  const modeRef = useRef(mode);
  const manualQualityRef = useRef(manualQuality);
  const customModelInputRef = useRef(null);
  const qoePopoverRef = useRef(null);
  const qoeControlValuesRef = useRef(qoeControls);
  const measuredBandwidthRef = useRef(0);
  const playbackTelemetryRef = useRef({
    bufferSeconds: 6,
    totalRebufferSeconds: 0,
    stallCount: 0,
    isStalling: false,
    updatedAt: 0,
  });
  const networkBiasRef = useRef(0);
  const signalStrengthRef = useRef(85);
  const signalOverrideEnabledRef = useRef(false);

  useEffect(() => {
    document.body.dataset.theme = theme;
  }, [theme]);

  useEffect(() => {
    modeRef.current = mode;
  }, [mode]);

  useEffect(() => {
    manualQualityRef.current = manualQuality;
  }, [manualQuality]);

  useEffect(() => {
    networkBiasRef.current = networkBias;
  }, [networkBias]);

  useEffect(() => {
    signalStrengthRef.current = signalStrength;
    setNetworkSpeed(resolveEffectiveNetworkSpeed());
  }, [signalStrength]);

  useEffect(() => {
    signalOverrideEnabledRef.current = signalOverrideEnabled;
    networkBiasRef.current = 0;
    setNetworkBias(0);
    if (!signalOverrideEnabled) {
      setBufferSize(playbackTelemetryRef.current.bufferSeconds || 0);
      setRebufferEvents(playbackTelemetryRef.current.stallCount || 0);
    }
    setNetworkSpeed(resolveEffectiveNetworkSpeed());
  }, [signalOverrideEnabled]);

  useEffect(() => {
    qoeControlValuesRef.current = qoeControls;
  }, [qoeControls]);

  useEffect(() => {
    if (!showQoEControls) {
      return undefined;
    }

    const handlePointerDown = (event) => {
      if (!qoePopoverRef.current?.contains(event.target)) {
        setShowQoEControls(false);
      }
    };

    const handleEscape = (event) => {
      if (event.key === "Escape") {
        setShowQoEControls(false);
      }
    };

    document.addEventListener("mousedown", handlePointerDown);
    document.addEventListener("keydown", handleEscape);
    return () => {
      document.removeEventListener("mousedown", handlePointerDown);
      document.removeEventListener("keydown", handleEscape);
    };
  }, [showQoEControls]);

  useEffect(() => {
    const connection = navigator.connection || navigator.mozConnection || navigator.webkitConnection;
    if (!connection) return undefined;

    const updateConnectionEstimate = () => {
      if (Number.isFinite(connection.downlink) && connection.downlink > 0) {
        handleBandwidthSample({
          mbps: connection.downlink,
          source: "browser estimate",
        });
      }
    };

    updateConnectionEstimate();
    connection.addEventListener?.("change", updateConnectionEstimate);
    return () => {
      connection.removeEventListener?.("change", updateConnectionEstimate);
    };
  }, []);

  useEffect(() => {
    bootstrap();
    return () => {
      if (timerRef.current) {
        clearInterval(timerRef.current);
      }
    };
  }, []);

  useEffect(() => {
    const intervalId = window.setInterval(() => {
      refreshModelCatalog();
    }, 10000);

    const onVisibilityChange = () => {
      if (!document.hidden) {
        refreshModelCatalog();
      }
    };

    document.addEventListener("visibilitychange", onVisibilityChange);
    return () => {
      window.clearInterval(intervalId);
      document.removeEventListener("visibilitychange", onVisibilityChange);
    };
  }, [lastTrainedModel, selectedModelOption]);

  async function bootstrap() {
    const response = await fetch("/api/bootstrap");
    const data = await response.json();

    const initialLast = data.last_trained_model ?? "";
    const initialDefault = data.default_model_path ?? "";
    const initialModel = initialLast || initialDefault;
    const initialController = initialLast ? "real" : "simulated";
    const initialControllerLabel = initialLast ? "Real Pensieve PPO" : "Simulated PPO Rules";

    setMatches(data.matches ?? []);
    setSelectedMatch(data.matches?.[0] ?? null);
    setQualityLadder(data.quality_ladder ?? []);
    setControllers(data.available_controllers ?? []);
    setControllerType(initialController);
    setLastTrainedModel(initialLast);
    setSelectedModelOption(initialLast ? "__last__" : "__custom__");
    setCustomModelPath(initialModel);
    setResolvedModelPath(initialModel);
    setManualQuality("720p");
    setAppliedQuality("720p");
    setAiQuality("720p");
    setActiveControllerLabel(initialControllerLabel);
    setActiveModelPath(initialModel);
    setAgentSummary(
      data.agent?.model_loaded
        ? `Model ready: ${data.agent.model_summary}`
        : `Simulation mode: ${data.agent?.model_summary ?? "Unavailable"}`
    );
  }

  async function refreshModelCatalog() {
    const response = await fetch("/api/model-catalog");
    const data = await response.json();
    const nextLastModel = data.last_trained_model ?? "";

    if (nextLastModel && nextLastModel !== lastTrainedModel) {
      setLastTrainedModel(nextLastModel);
      if (selectedModelOption === "__last__") {
        setCustomModelPath(nextLastModel);
        setResolvedModelPath(nextLastModel);
      }
      setValidationState("success");
      setValidationMessage(`New model detected automatically: ${extractModelName(nextLastModel)}`);
    }
  }

  async function validateModel() {
    setValidationState("loading");
    setValidationMessage("");
    setErrorMessage("");

    const effectiveModelPath = resolveEffectiveModelPath();
    const response = await fetch("/api/validate-model", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        controller_type: controllerType,
        model_path: effectiveModelPath,
      }),
    });

    const data = await response.json();
    const resolved = data.resolved_model_path || data.model_path || effectiveModelPath;
    setResolvedModelPath(resolved);

    if (response.ok) {
      setValidationState("success");
      setValidationMessage(data.message || "Model validation passed.");
      return;
    }

    setValidationState("error");
    setValidationMessage(data.message || data.error || "Model validation failed.");
  }

  async function startSession() {
    if (!selectedMatch) return;

    await endSession(true);
    setErrorMessage("");
    setValidationMessage("");

    const effectiveModelPath = resolveEffectiveModelPath();
    const response = await fetch("/api/session", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        match_id: selectedMatch.id,
        mode,
        controller_type: controllerType,
        model_path: effectiveModelPath,
      }),
    });

    const data = await response.json();
    if (!response.ok) {
      setErrorMessage(data.error || "Unable to start session.");
      setResolvedModelPath(data.resolved_model_path || effectiveModelPath);
      return;
    }

    setSessionId(data.session_id);
    setAppliedQuality(data.current_quality);
    setAiQuality(data.current_quality);
    setQoe(0);
    setRebufferEvents(0);
    playbackTelemetryRef.current = {
      bufferSeconds: 6,
      totalRebufferSeconds: 0,
      stallCount: 0,
      isStalling: false,
      updatedAt: Date.now(),
    };
    setSummary({
      ...emptySummary,
      controller_label: data.controller_label,
      model_path: data.model_path,
    });
    setBufferSize(6);
    setNetworkSpeed(resolveEffectiveNetworkSpeed(4.8));
    setSessionLog([]);
    setActionProbabilities([]);
    setActiveControllerLabel(data.controller_label);
    setActiveModelPath(data.model_path);
    setResolvedModelPath(data.model_path || effectiveModelPath);
    beginDecisionLoop(data.session_id, data.current_quality);
  }

  async function endSession(silent = false) {
    if (timerRef.current) {
      clearInterval(timerRef.current);
      timerRef.current = null;
    }

    if (!sessionId) return;

    const response = await fetch(`/api/session/${sessionId}/end`, { method: "POST" });
    const data = await response.json();
    setSummary(data.summary);
    if (!silent) {
      pushLog(`Session ended. Final QoE: ${Number(data.summary.final_qoe).toFixed(3)}`);
    }
    setSessionId(null);
  }

  function beginDecisionLoop(activeSessionId, initialQuality) {
    let localQuality = initialQuality;
    let localNetwork = resolveEffectiveNetworkSpeed(networkSpeed);
    let localBuffer = 6;
    let lastObservedRebuffer = playbackTelemetryRef.current.totalRebufferSeconds || 0;

    timerRef.current = window.setInterval(async () => {
      localNetwork = resolveEffectiveNetworkSpeed(localNetwork);
      const playbackTelemetry = playbackTelemetryRef.current;
      const hasFreshPlaybackTelemetry = Date.now() - (playbackTelemetry.updatedAt || 0) < 6000;
      const useActualPlaybackTelemetry = !signalOverrideEnabledRef.current && hasFreshPlaybackTelemetry;
      let rebufferSeconds = 0;

      if (useActualPlaybackTelemetry) {
        localBuffer = clampBufferSeconds(playbackTelemetry.bufferSeconds);
        rebufferSeconds = Math.max(
          0,
          Number((playbackTelemetry.totalRebufferSeconds - lastObservedRebuffer).toFixed(2))
        );
        lastObservedRebuffer = playbackTelemetry.totalRebufferSeconds;
      } else {
        const currentBitrate = bitrateForLabel(localQuality, qualityLadder);
        const bufferDrift = ((localNetwork / Math.max(currentBitrate, 0.25)) - 1) * 2.35;
        localBuffer = clampBufferSeconds(localBuffer + bufferDrift - 0.45);
        rebufferSeconds = localBuffer <= 0.35
          ? Number(Math.min(2.4, 0.7 + (0.35 - localBuffer) * 3).toFixed(2))
          : 0;

        if (rebufferSeconds > 0) {
          localBuffer = 0.4;
        }
      }

      const response = await fetch(`/api/session/${activeSessionId}/decision`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          network_speed: localNetwork,
          buffer_size: localBuffer,
          current_quality: localQuality,
          manual_quality: manualQualityRef.current,
          rebuffer_seconds: rebufferSeconds,
          rebuf_penalty: qoeControlValuesRef.current.rebufPenalty,
          smooth_penalty: qoeControlValuesRef.current.smoothPenalty,
        }),
      });

      const data = await response.json();
      if (!response.ok) {
        setErrorMessage(data.error || "Error while taking decision.");
        clearInterval(timerRef.current);
        timerRef.current = null;
        return;
      }

      localQuality = data.applied_quality;
      setNetworkSpeed(localNetwork);
      setBufferSize(localBuffer);
      setRebufferEvents(useActualPlaybackTelemetry ? playbackTelemetry.stallCount : data.summary.rebuffer_events);
      setAppliedQuality(data.applied_quality);
      setAiQuality(data.recommended_quality);
      setQoe(data.qoe);
      setSummary(data.summary);
      setStreamResolution(data.resolution);
      setRecommendationReason(data.reason);
      setActiveControllerLabel(data.controller_label);
      setActiveModelPath(data.model_path);
      setResolvedModelPath((previous) => data.model_path || previous);
      setActionProbabilities(data.action_probabilities || []);
      pushLog(
        `${modeRef.current === "manual" ? "Manual" : "AI"} | ${data.controller_label} | Net ${localNetwork.toFixed(1)} Mbps | Buffer ${localBuffer.toFixed(1)}s | Quality ${data.applied_quality} | QoE ${Number(data.qoe).toFixed(3)}`
      );
    }, 3000);
  }

  function pushLog(message) {
    const timestamp = new Date().toLocaleTimeString("ar-IQ", {
      hour: "2-digit",
      minute: "2-digit",
      second: "2-digit",
    });
    setSessionLog((previous) => [{ timestamp, message }, ...previous].slice(0, 20));
  }

  function simulateNetwork(type) {
    if (!signalOverrideEnabledRef.current) {
      pushLog("Signal Strength is on Auto, so manual network simulation is disabled.");
      return;
    }

    const delta = type === "fast" ? 1.5 : -1.5;
    setNetworkBias((value) => clampNetworkSpeed(value + delta, -8, 8));
    setNetworkSpeed((value) => clampNetworkSpeed(value + delta));
    setBufferSize((value) => (type === "fast" ? Math.min(18, value + 2) : Math.max(0, value - 1.5)));
  }

  function handleBandwidthSample(sample) {
    if (!sample || !Number.isFinite(sample.mbps) || sample.mbps <= 0) {
      return;
    }

    const smoothedMbps = measuredBandwidthRef.current > 0
      ? (measuredBandwidthRef.current * 0.72) + (sample.mbps * 0.28)
      : sample.mbps;

    measuredBandwidthRef.current = smoothedMbps;
    setNetworkTelemetrySource(sample.source || "live player telemetry");
    setNetworkSpeed(resolveEffectiveNetworkSpeed(smoothedMbps));
  }

  function handlePlaybackTelemetry(sample) {
    if (!sample || !Number.isFinite(sample.bufferSeconds)) {
      return;
    }

    playbackTelemetryRef.current = {
      bufferSeconds: sample.bufferSeconds,
      totalRebufferSeconds: Number.isFinite(sample.totalRebufferSeconds) ? sample.totalRebufferSeconds : 0,
      stallCount: Number.isFinite(sample.stallCount) ? sample.stallCount : 0,
      isStalling: Boolean(sample.isStalling),
      updatedAt: Date.now(),
    };

    if (!signalOverrideEnabledRef.current) {
      setBufferSize(clampBufferSeconds(sample.bufferSeconds));
      setRebufferEvents(sample.stallCount ?? 0);
    }
  }

  function resolveEffectiveNetworkSpeed(fallbackMbps = 4.8) {
    const connection = navigator.connection || navigator.mozConnection || navigator.webkitConnection;
    const browserEstimate = Number.isFinite(connection?.downlink) && connection.downlink > 0
      ? connection.downlink
      : 0;
    const baseMbps = measuredBandwidthRef.current || browserEstimate || fallbackMbps;
    if (!signalOverrideEnabledRef.current) {
      return clampNetworkSpeed(baseMbps, 0.3, 20);
    }
    const signalCap = getSignalCapacityMbps(signalStrengthRef.current);
    const signalFactor = getSignalFactor(signalStrengthRef.current);
    const constrainedBase = Math.min(baseMbps * signalFactor, signalCap);
    return clampNetworkSpeed(constrainedBase + (networkBiasRef.current * 0.65), 0.3, Math.max(signalCap + 1.2, 1.2));
  }

  function resetSelection(match) {
    if (timerRef.current) {
      clearInterval(timerRef.current);
      timerRef.current = null;
    }
    setSelectedMatch(match);
    setSessionId(null);
    setAppliedQuality("720p");
    setAiQuality("720p");
    setQoe(0);
    setRebufferEvents(0);
    playbackTelemetryRef.current = {
      bufferSeconds: 6,
      totalRebufferSeconds: 0,
      stallCount: 0,
      isStalling: false,
      updatedAt: 0,
    };
    setSummary(emptySummary);
    setRecommendationReason("Decision reason will appear here.");
    setSessionLog([]);
    setActionProbabilities([]);
    setErrorMessage("");
  }

  function resolveEffectiveModelPath() {
    return resolveModelOptionPath(
      selectedModelOption,
      customModelInputRef.current?.value ?? customModelPath,
      lastTrainedModel
    );
  }

  const appliedBitrate = useMemo(
    () => bitrateForLabel(appliedQuality, qualityLadder),
    [appliedQuality, qualityLadder]
  );
  const sessionBadge = sessionId ? `Session ${sessionId.slice(0, 8)}` : "No active session";
  const showCustomInput = selectedModelOption === "__custom__";
  const lastModelDisplay = formatLastModelLabel(lastTrainedModel) || "No Model";
  const displayedResolvedPath = resolvedModelPath || resolveEffectiveModelPath() || "N/A";
  const connectionState = getConnectionState(networkSpeed);
  const signalState = getSignalState(signalStrength);
  const signalModeLabel = signalOverrideEnabled ? "Manual signal control" : "Auto device signal";
  const signalMetricValue = signalOverrideEnabled ? `${signalStrength}%` : "Auto";
  const signalMetricNote = signalOverrideEnabled
    ? `${signalState.label} signal`
    : `Using ${networkTelemetrySource}`;
  const controllerModelSelection = formatControllerModelSelection({
    selectedModelOption,
    lastTrainedModel,
    customModelPath,
    resolvedModelPath,
    activeModelPath,
  });

  return (
    <div className="page-shell">
      <header className="topbar">
        <div className="brand-lockup">
          <div className="brand-mark">RS</div>
          <div>
            <h1>ReinforceStream</h1>
            <p>Adaptive sports streaming dashboard linked to your project models</p>
          </div>
        </div>

        <nav className="topnav">
          <a href="#matches">Matches</a>
          <a href="#watch">Watch</a>
          <a href="#insights">Insights</a>
        </nav>

        <button className="ghost-btn" onClick={() => setTheme(theme === "dark" ? "light" : "dark")}>
          {theme === "dark" ? "Light Mode" : "Dark Mode"}
        </button>
      </header>

      <section className="hero">
        <div className="hero-copy">
          <span className="eyebrow">Adaptive Bitrate Streaming + Real/Simulated PPO</span>
          <h2>AI-Powered Adaptive Video Streaming Control</h2>
          <p>
            You can choose the last trained model automatically or enter a custom path,
            validate it, and clearly see the resolved file path that the backend will use.
          </p>
          <div className="hero-actions">
            <a href="#watch" className="primary-btn link-btn">Start Watching</a>
            <button className="ghost-btn" onClick={() => simulateNetwork("fast")} disabled={!signalOverrideEnabled}>
              Simulate Fast Network
            </button>
            <button className="ghost-btn" onClick={() => simulateNetwork("slow")} disabled={!signalOverrideEnabled}>
              Simulate Slow Network
            </button>
          </div>
        </div>

        <div className="hero-panel">
          <div className="status-chip">
            <span className="dot"></span>
            <span>{agentSummary}</span>
          </div>
          <div className="mini-grid">
            <article><strong>{appliedBitrate.toFixed(2)} Mbps</strong><span>Current Bitrate</span></article>
            <article><strong>{bufferSize.toFixed(1)} s</strong><span>Buffer Size</span></article>
            <article><strong>{Number(qoe).toFixed(3)}</strong><span>QoE Score</span></article>
            <article><strong>{activeControllerLabel}</strong><span>Controller</span></article>
          </div>
        </div>
      </section>

      <main className="main-grid">
        <section className="matches-panel" id="matches">
          <div className="section-head">
            <div>
              <span className="eyebrow">Current & Upcoming</span>
              <h3>Matches</h3>
            </div>
          </div>
          <div className="matches-grid">
            {matches.map((match) => (
              <MatchCard
                key={match.id}
                match={match}
                active={selectedMatch?.id === match.id}
                onSelect={() => resetSelection(match)}
              />
            ))}
          </div>
        </section>

        <section className="watch-panel" id="watch">
          <div className="section-head">
            <div>
              <span className="eyebrow">Live Player</span>
              <h3>{selectedMatch ? selectedMatch.title : "Choose a match"}</h3>
            </div>
            <div className="section-tools" ref={qoePopoverRef}>
              <button
                className="icon-control-btn"
                type="button"
                aria-haspopup="dialog"
                aria-expanded={showQoEControls}
                onClick={() => setShowQoEControls((value) => !value)}
              >
                <svg viewBox="0 0 24 24" aria-hidden="true">
                  <path d="M4 7h8M16 7h4M10 7a2 2 0 1 0 4 0a2 2 0 0 0-4 0ZM4 17h4M12 17h8M8 17a2 2 0 1 0 4 0a2 2 0 0 0-4 0ZM4 12h14M20 12h0" />
                </svg>
                <span>QoE Controls</span>
              </button>

              {showQoEControls ? (
                <div className="qoe-popover" role="dialog" aria-label="QoE Controls">
                  <QoEControls onChange={setQoeControls} />
                </div>
              ) : null}
            </div>
          </div>

          <div className="controller-grid">
            <label className="select-wrap">
              <span>Playback Mode</span>
              <select value={mode} onChange={(event) => setMode(event.target.value)}>
                <option value="ai">AI Agent</option>
                <option value="manual">Manual</option>
              </select>
            </label>

            <label className="select-wrap">
              <span>Controller</span>
              <select value={controllerType} onChange={(event) => setControllerType(event.target.value)}>
                {controllers.map((controller) => (
                  <option key={controller.value} value={controller.value}>{controller.label}</option>
                ))}
              </select>
            </label>

            <label className="select-wrap">
              <span>Model Selection</span>
              <select
                value={selectedModelOption}
                onChange={(event) => {
                  const nextOption = event.target.value;
                  setSelectedModelOption(nextOption);
                  if (nextOption === "__last__") {
                    setCustomModelPath(lastTrainedModel || "");
                    setResolvedModelPath(lastTrainedModel || "");
                  }
                }}
              >
                {lastTrainedModel ? (
                  <option value="__last__">{`Use Last Model (${lastModelDisplay}) - \u064a\u0646\u0635\u062d \u0628\u0647`}</option>
                ) : null}
                <option value="__custom__">Custom Model</option>
              </select>
            </label>

            {showCustomInput ? (
              <label className="select-wrap model-path-wrap">
                <span>Custom Model Path</span>
                <input
                  className="text-input"
                  value={customModelPath}
                  onChange={(event) => setCustomModelPath(event.target.value)}
                  ref={customModelInputRef}
                  placeholder="Example: src/ppo/nn_model_ep_500000.pth"
                />
                <div className="inline-validation-row">
                  <button
                    className="ghost-btn"
                    type="button"
                    onClick={validateModel}
                    disabled={validationState === "loading"}
                  >
                    {validationState === "loading" ? "Validating..." : "Validate Model"}
                  </button>
                  {validationMessage ? (
                    <div className={`validation-banner ${validationState === "success" ? "success" : "error"}`}>
                      {validationMessage}
                    </div>
                  ) : null}
                </div>
                <small className="resolved-path">{`Resolved to: ${displayedResolvedPath}`}</small>
              </label>
            ) : null}

            <label className="select-wrap">
              <span>Manual Quality</span>
              <select value={manualQuality} onChange={(event) => setManualQuality(event.target.value)}>
                {qualityLadder.map((quality) => (
                  <option key={quality.label} value={quality.label}>{quality.label}</option>
                ))}
              </select>
            </label>

            <label className="select-wrap signal-wrap">
              <div className="signal-header-row">
                <span>{signalOverrideEnabled ? `Signal Strength (${signalStrength}%)` : "Signal Strength (Auto)"}</span>
                <button
                  className={`signal-toggle-btn ${signalOverrideEnabled ? "active" : ""}`}
                  type="button"
                  onClick={() => setSignalOverrideEnabled((value) => !value)}
                  aria-pressed={signalOverrideEnabled}
                  title={signalOverrideEnabled ? "Disable manual signal control" : "Enable manual signal control"}
                >
                  <svg viewBox="0 0 24 24" aria-hidden="true">
                    <path d="M12 3v8" />
                    <path d="M7.4 5.9A8 8 0 1 0 16.6 5.9" />
                  </svg>
                  <span>{signalOverrideEnabled ? "Manual" : "Auto"}</span>
                </button>
              </div>
              <input
                className="range-input"
                type="range"
                min="10"
                max="100"
                step="5"
                value={signalStrength}
                onChange={(event) => setSignalStrength(Number(event.target.value))}
                disabled={!signalOverrideEnabled}
              />
              <small className="range-caption">
                {signalOverrideEnabled
                  ? `${signalState.label} signal | speed cap ${getSignalCapacityMbps(signalStrength).toFixed(1)} Mbps`
                  : `Auto mode | depending on live device throughput via ${networkTelemetrySource}`}
              </small>
            </label>
          </div>

          {!showCustomInput ? (
            <div className="validation-row">
              <small className="resolved-path">{`Resolved to: ${displayedResolvedPath}`}</small>
            </div>
          ) : null}

          {errorMessage ? <div className="error-banner">{errorMessage}</div> : null}

          <Player
            streamUrl={selectedMatch?.stream_url}
            streamSources={selectedMatch?.quality_streams}
            bandwidthProbeUrl={selectedMatch?.bandwidth_probe_url}
            league={selectedMatch?.league}
            resolution={streamResolution}
            qualityLabel={appliedQuality}
            targetBitrateMbps={appliedBitrate}
            onBandwidthSample={handleBandwidthSample}
            onPlaybackTelemetry={handlePlaybackTelemetry}
            connectionStatus={connectionState.label}
            connectionTone={connectionState.tone}
            connectionSource={`${networkTelemetrySource} | ${signalModeLabel}`}
          />

          <div className="watch-actions">
            <button className="primary-btn" onClick={startSession}>Start Session</button>
            <button className="ghost-btn" onClick={() => endSession(false)}>End Session</button>
            <button className="ghost-btn" onClick={() => setBufferSize((value) => Math.min(18, value + 3.5))}>Increase Buffer</button>
            <button className="ghost-btn" onClick={() => setBufferSize((value) => Math.max(0, value - 3.5))}>Decrease Buffer</button>
          </div>
        </section>

        <section className="insights-panel" id="insights">
          <div className="section-head">
            <div>
              <span className="eyebrow">Streaming Telemetry</span>
              <h3>Technical Insights</h3>
            </div>
          </div>

          <div className="metrics-grid">
            <MetricCard title="Bitrate" value={`${appliedBitrate.toFixed(2)} Mbps`} note="Applied bitrate" />
            <MetricCard title="Network Speed" value={`${networkSpeed.toFixed(1)} Mbps`} note={`${connectionState.label} via ${networkTelemetrySource}`} />
            <MetricCard title="Signal Strength" value={signalMetricValue} note={signalMetricNote} />
            <MetricCard title="Buffer Size" value={`${bufferSize.toFixed(1)} s`} note="Buffered playback time" />
            <MetricCard title="Rebuffer Events" value={`${rebufferEvents}`} note="Playback stalls" />
            <MetricCard title="QoE Score" value={Number(qoe).toFixed(3)} note="Current QoE" />
            <MetricCard title="AI Recommendation" value={aiQuality} note={recommendationReason} emphasis />
          </div>

          <div className="comparison-layout">
            <article className="comparison-card">
              <h4>Controller Binding</h4>
              <div className="compare-row"><span>Controller</span><strong>{activeControllerLabel}</strong></div>
              <div className="compare-row"><span>Model Selected</span><strong className="path-value">{controllerModelSelection}</strong></div>
              <div className="compare-row"><span>Mode</span><strong>{mode === "manual" ? "Manual" : "AI Agent"}</strong></div>
            </article>

            <article className="comparison-card">
              <h4>Session Statistics</h4>
              <div className="compare-row"><span>Average Quality</span><strong>{summary.avg_quality}</strong></div>
              <div className="compare-row"><span>Total Rebuffer</span><strong>{Number(summary.total_rebuffer_seconds).toFixed(2)} s</strong></div>
              <div className="compare-row"><span>Final QoE</span><strong>{Number(summary.final_qoe).toFixed(3)}</strong></div>
            </article>
          </div>

          {actionProbabilities.length ? (
            <article className="comparison-card probability-card">
              <h4>PPO Action Probabilities</h4>
              <div className="probability-list">
                {qualityLadder.map((quality, index) => (
                  <div key={quality.label} className="probability-row">
                    <span>{quality.label}</span>
                    <strong>{Number(actionProbabilities[index] ?? 0).toFixed(4)}</strong>
                  </div>
                ))}
              </div>
            </article>
          ) : null}

          <article className="timeline-card">
            <div className="timeline-head">
              <h4>Live Session Log</h4>
              <span>{sessionBadge}</span>
            </div>
            <div className="session-log">
              {sessionLog.map((item, index) => (
                <div className="timeline-row" key={`${item.timestamp}-${index}`}>
                  <span>{item.timestamp}</span>
                  <strong>{item.message}</strong>
                </div>
              ))}
            </div>
          </article>
        </section>
      </main>
    </div>
  );
}

function bitrateForLabel(label, qualityLadder) {
  return qualityLadder.find((item) => item.label === label)?.bitrate_mbps ?? 1.2;
}

function clampNetworkSpeed(value, min = 0.3, max = 20) {
  return Math.max(min, Math.min(max, value));
}

function clampBufferSeconds(value, min = 0, max = 18) {
  return Math.max(min, Math.min(max, Number.isFinite(value) ? value : min));
}

function getSignalFactor(signalStrength) {
  const normalized = Math.max(0.1, Math.min(1, signalStrength / 100));
  return 0.12 + (normalized * 0.88);
}

function getSignalCapacityMbps(signalStrength) {
  if (signalStrength >= 90) return 12;
  if (signalStrength >= 80) return 9;
  if (signalStrength >= 70) return 7;
  if (signalStrength >= 60) return 5.2;
  if (signalStrength >= 50) return 3.8;
  if (signalStrength >= 40) return 2.7;
  if (signalStrength >= 30) return 1.8;
  if (signalStrength >= 20) return 1.1;
  return 0.55;
}

function getSignalState(signalStrength) {
  if (signalStrength >= 80) {
    return { label: "Excellent", tone: "live" };
  }
  if (signalStrength >= 60) {
    return { label: "Strong", tone: "live" };
  }
  if (signalStrength >= 40) {
    return { label: "Fair", tone: "degraded" };
  }
  if (signalStrength >= 20) {
    return { label: "Weak", tone: "degraded" };
  }
  return { label: "Very Weak", tone: "critical" };
}

function getConnectionState(networkSpeed) {
  if (networkSpeed >= 5) {
    return { label: "Live", tone: "live" };
  }
  if (networkSpeed >= 2) {
    return { label: "Degraded", tone: "degraded" };
  }
  return { label: "Critical", tone: "critical" };
}

function extractModelName(modelPath) {
  if (!modelPath) return "";
  const parts = modelPath.split(/[\\/]/);
  return parts[parts.length - 1]?.replace(/\.(pth|pt)$/i, "") || modelPath;
}

function formatLastModelLabel(modelPath) {
  const modelName = extractModelName(modelPath);
  const epochMatch = modelName.match(/(ep_\d+)/i);
  return epochMatch?.[1] || modelName;
}

function formatControllerModelSelection({
  selectedModelOption,
  lastTrainedModel,
  customModelPath,
  resolvedModelPath,
  activeModelPath,
}) {
  const activeLabel = formatLastModelLabel(activeModelPath);
  const customLabel = formatLastModelLabel(resolvedModelPath || customModelPath);
  const lastLabel = formatLastModelLabel(lastTrainedModel);

  if (selectedModelOption === "__last__") {
    return lastLabel ? `Use Last Model (${lastLabel})` : "Use Last Model";
  }

  if (selectedModelOption === "__custom__") {
    return customLabel ? `Custom Model (${customLabel})` : "Custom Model";
  }

  return activeLabel || customLabel || lastLabel || "Not set";
}

function resolveModelOptionPath(selectedModelOption, customModelPath, lastTrainedModel) {
  if (selectedModelOption === "__custom__") {
    return customModelPath.trim();
  }
  if (selectedModelOption === "__last__") {
    return (lastTrainedModel || "").trim();
  }
  return selectedModelOption.trim();
}
