const { useEffect, useRef, useState } = React;

const QUALITY_BITRATES = {
  "480p": 1.2,
  "720p": 3.0,
  "1080p": 6.0,
  "4K": 12.0,
};

function App() {
  const [matches, setMatches] = useState([]);
  const [selectedMatch, setSelectedMatch] = useState(null);
  const [agentSummary, setAgentSummary] = useState("جارٍ تحميل بيانات الوكيل...");
  const [theme, setTheme] = useState("dark");
  const [mode, setMode] = useState("ai");
  const [manualQuality, setManualQuality] = useState("720p");
  const [sessionId, setSessionId] = useState(null);
  const [appliedQuality, setAppliedQuality] = useState("720p");
  const [aiQuality, setAiQuality] = useState("720p");
  const [networkSpeed, setNetworkSpeed] = useState(4.8);
  const [bufferSize, setBufferSize] = useState(6.0);
  const [qoe, setQoe] = useState(0);
  const [rebufferEvents, setRebufferEvents] = useState(0);
  const [summary, setSummary] = useState({
    avg_quality: "N/A",
    total_rebuffer_seconds: 0,
    final_qoe: 0,
    rebuffer_events: 0,
  });
  const [recommendationReason, setRecommendationReason] = useState("سيظهر تفسير القرار هنا");
  const [streamResolution, setStreamResolution] = useState("1280x720");
  const [sessionLog, setSessionLog] = useState([]);
  const timerRef = useRef(null);
  const modeRef = useRef(mode);
  const manualQualityRef = useRef(manualQuality);

  useEffect(() => {
    modeRef.current = mode;
  }, [mode]);

  useEffect(() => {
    manualQualityRef.current = manualQuality;
  }, [manualQuality]);

  useEffect(() => {
    document.body.dataset.theme = theme;
  }, [theme]);

  useEffect(() => {
    bootstrap();
    return () => {
      if (timerRef.current) {
        clearInterval(timerRef.current);
      }
    };
  }, []);

  async function bootstrap() {
    const response = await fetch("/api/bootstrap");
    const data = await response.json();
    setMatches(data.matches);
    setSelectedMatch(data.matches[0]);
    setAgentSummary(
      data.agent.model_loaded
        ? `Model ready: ${data.agent.model_summary}`
        : `Simulation mode: ${data.agent.model_summary}`
    );
  }

  async function startSession() {
    if (!selectedMatch) return;
    await endSession(true);

    const response = await fetch("/api/session", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        match_id: selectedMatch.id,
        mode,
      }),
    });
    const data = await response.json();
    setSessionId(data.session_id);
    setAppliedQuality(data.current_quality);
    setAiQuality(data.current_quality);
    setQoe(0);
    setRebufferEvents(0);
    setSummary({
      avg_quality: "N/A",
      total_rebuffer_seconds: 0,
      final_qoe: 0,
      rebuffer_events: 0,
    });
    setBufferSize(6);
    setNetworkSpeed(4.8);
    setSessionLog([]);
    beginDecisionLoop(data.session_id, data.current_quality);
  }

  async function endSession(silent = false) {
    if (timerRef.current) {
      clearInterval(timerRef.current);
      timerRef.current = null;
    }

    if (!sessionId) return;

    const response = await fetch(`/api/session/${sessionId}/end`, {
      method: "POST",
    });
    const data = await response.json();
    setSummary(data.summary);
    if (!silent) {
      pushLog(`انتهت الجلسة. QoE النهائي: ${Number(data.summary.final_qoe).toFixed(3)}`);
    }
    setSessionId(null);
  }

  function beginDecisionLoop(activeSessionId, initialQuality) {
    let localQuality = initialQuality;
    let localNetwork = 4.8;
    let localBuffer = 6.0;

    timerRef.current = setInterval(async () => {
      const jitter = (Math.random() - 0.5) * 1.4;
      localNetwork = Math.max(0.6, Math.min(14, localNetwork + jitter));
      const currentBitrate = QUALITY_BITRATES[localQuality] ?? 3.0;
      localBuffer = Math.max(0, Math.min(18, localBuffer + (localNetwork - currentBitrate) * 0.55));
      const rebufferSeconds = localBuffer <= 0.35 ? Number((Math.random() * 1.4 + 0.3).toFixed(2)) : 0;

      if (rebufferSeconds > 0) {
        localBuffer = 0.4;
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
        }),
      });
      const data = await response.json();

      localQuality = data.applied_quality;
      setNetworkSpeed(localNetwork);
      setBufferSize(localBuffer);
      setAppliedQuality(data.applied_quality);
      setAiQuality(data.recommended_quality);
      setQoe(data.qoe);
      setSummary(data.summary);
      setStreamResolution(data.resolution);
      setRecommendationReason(data.reason);
      setRebufferEvents(data.summary.rebuffer_events);
      pushLog(
        `${modeRef.current === "manual" ? "Manual" : "AI"} | شبكة ${localNetwork.toFixed(1)} Mbps | Buffer ${localBuffer.toFixed(1)}s | جودة ${data.applied_quality} | QoE ${Number(data.qoe).toFixed(3)}`
      );
    }, 3000);
  }

  function pushLog(message) {
    const timestamp = new Date().toLocaleTimeString("ar-IQ", {
      hour: "2-digit",
      minute: "2-digit",
      second: "2-digit",
    });
    setSessionLog((prev) => [{ timestamp, message }, ...prev].slice(0, 20));
  }

  function simulateNetwork(type) {
    setNetworkSpeed((value) => {
      if (type === "fast") return Math.min(14, value + 4.5);
      return Math.max(0.6, value - 3.5);
    });
    setBufferSize((value) => {
      if (type === "fast") return Math.min(18, value + 2);
      return Math.max(0, value - 1.5);
    });
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
    setSummary({
      avg_quality: "N/A",
      total_rebuffer_seconds: 0,
      final_qoe: 0,
      rebuffer_events: 0,
    });
    setRecommendationReason("سيظهر تفسير القرار هنا");
    setSessionLog([]);
  }

  const appliedBitrate = QUALITY_BITRATES[appliedQuality] ?? 3.0;
  const sessionBadge = sessionId ? `Session ${sessionId.slice(0, 8)}` : "لا توجد جلسة نشطة";

  return (
    <div className="page-shell">
      <header className="topbar">
        <div className="brand-lockup">
          <div className="brand-mark">SC</div>
          <div>
            <h1>SportCast AI</h1>
            <p>منصة بث رياضي مباشر بتكيف ذكي للجودة</p>
          </div>
        </div>

        <nav className="topnav">
          <a href="#matches">المباريات</a>
          <a href="#watch">المشاهدة</a>
          <a href="#insights">التحليلات</a>
        </nav>

        <button className="ghost-btn" onClick={() => setTheme(theme === "dark" ? "light" : "dark")}>
          {theme === "dark" ? "Light Mode" : "Dark Mode"}
        </button>
      </header>

      <section className="hero">
        <div className="hero-copy">
          <span className="eyebrow">Adaptive Bitrate Streaming + Simulated PPO</span>
          <h2>راقب كيف يختار الوكيل الجودة المناسبة لحظة بلحظة أثناء بث المباريات.</h2>
          <p>
            الواجهة الآن مبنية بـ React، وتعرض المباريات الحالية والقادمة، ومشغل فيديو مباشر،
            ولوحة فنية كاملة توضح الشبكة والمخزن المؤقت والـ QoE ووضع المقارنة.
          </p>
          <div className="hero-actions">
            <a href="#watch" className="primary-btn link-btn">ابدأ المشاهدة</a>
            <button className="ghost-btn" onClick={() => simulateNetwork("fast")}>Simulate Fast Network</button>
            <button className="ghost-btn" onClick={() => simulateNetwork("slow")}>Simulate Slow Network</button>
          </div>
        </div>

        <div className="hero-panel">
          <div className="status-chip">
            <span className="dot"></span>
            <span>{agentSummary}</span>
          </div>
          <div className="mini-grid">
            <article>
              <strong>{appliedBitrate.toFixed(1)} Mbps</strong>
              <span>Current Bitrate</span>
            </article>
            <article>
              <strong>{bufferSize.toFixed(1)} s</strong>
              <span>Buffer Size</span>
            </article>
            <article>
              <strong>{Number(qoe).toFixed(3)}</strong>
              <span>QoE Score</span>
            </article>
            <article>
              <strong>{mode.toUpperCase()}</strong>
              <span>Mode</span>
            </article>
          </div>
        </div>
      </section>

      <main className="main-grid">
        <section className="matches-panel" id="matches">
          <div className="section-head">
            <div>
              <span className="eyebrow">Current & Upcoming</span>
              <h3>قائمة المباريات</h3>
            </div>
          </div>
          <div className="matches-grid">
            {matches.map((match) => (
              <article
                key={match.id}
                className={`match-card ${selectedMatch?.id === match.id ? "active" : ""}`}
                onClick={() => resetSelection(match)}
              >
                <div className="match-thumb" style={{ background: match.poster }}>
                  <strong>{match.league}</strong>
                </div>
                <div className="match-body">
                  <h4>{match.title}</h4>
                  <div className="meta-row"><span>{match.time}</span><span>{match.venue}</span></div>
                  <div className="meta-row"><span>القنوات</span><span>{match.channels.join(" • ")}</span></div>
                </div>
              </article>
            ))}
          </div>
        </section>

        <section className="watch-panel" id="watch">
          <div className="section-head">
            <div>
              <span className="eyebrow">Live Player</span>
              <h3>{selectedMatch ? selectedMatch.title : "اختر مباراة للبدء"}</h3>
            </div>
            <div className="control-row">
              <label className="select-wrap">
                <span>Mode</span>
                <select value={mode} onChange={(e) => setMode(e.target.value)}>
                  <option value="ai">AI Agent</option>
                  <option value="manual">Manual</option>
                </select>
              </label>
              <label className="select-wrap">
                <span>Manual Quality</span>
                <select value={manualQuality} onChange={(e) => setManualQuality(e.target.value)}>
                  <option value="480p">480p</option>
                  <option value="720p">720p</option>
                  <option value="1080p">1080p</option>
                  <option value="4K">4K</option>
                </select>
              </label>
            </div>
          </div>

          <Player streamUrl={selectedMatch?.stream_url} league={selectedMatch?.league} resolution={streamResolution} />

          <div className="watch-actions">
            <button className="primary-btn" onClick={startSession}>بدء جلسة مشاهدة</button>
            <button className="ghost-btn" onClick={() => endSession(false)}>إنهاء الجلسة</button>
            <button className="ghost-btn" onClick={() => setBufferSize((v) => Math.min(18, v + 3.5))}>رفع Buffer</button>
            <button className="ghost-btn" onClick={() => setBufferSize((v) => Math.max(0, v - 3.5))}>خفض Buffer</button>
          </div>
        </section>

        <section className="insights-panel" id="insights">
          <div className="section-head">
            <div>
              <span className="eyebrow">Streaming Telemetry</span>
              <h3>لوحة المعلومات الفنية</h3>
            </div>
          </div>

          <div className="metrics-grid">
            <MetricCard title="Bitrate" value={`${appliedBitrate.toFixed(1)} Mbps`} note="المعدل المطبق فعلياً على البث" />
            <MetricCard title="Network Speed" value={`${networkSpeed.toFixed(1)} Mbps`} note="سرعة الشبكة الحالية من جهة العميل" />
            <MetricCard title="Buffer Size" value={`${bufferSize.toFixed(1)} s`} note="الزمن المخزن قبل تشغيل التقطعات" />
            <MetricCard title="Rebuffer Events" value={`${rebufferEvents}`} note="عدد مرات توقف البث لإعادة التعبئة" />
            <MetricCard title="QoE Score" value={Number(qoe).toFixed(3)} note="مؤشر جودة التجربة اللحظي" />
            <MetricCard title="AI Recommendation" value={aiQuality} note={recommendationReason} emphasis />
          </div>

          <div className="comparison-layout">
            <article className="comparison-card">
              <h4>Manual vs AI</h4>
              <div className="compare-row"><span>الجودة اليدوية</span><strong>{manualQuality}</strong></div>
              <div className="compare-row"><span>اقتراح الذكاء الاصطناعي</span><strong>{aiQuality}</strong></div>
              <div className="compare-row"><span>الوضع الحالي</span><strong>{mode === "manual" ? "Manual" : "AI Agent"}</strong></div>
            </article>

            <article className="comparison-card">
              <h4>Session Statistics</h4>
              <div className="compare-row"><span>متوسط الجودة</span><strong>{summary.avg_quality}</strong></div>
              <div className="compare-row"><span>إجمالي وقت التوقف</span><strong>{Number(summary.total_rebuffer_seconds).toFixed(2)} s</strong></div>
              <div className="compare-row"><span>QoE النهائي</span><strong>{Number(summary.final_qoe).toFixed(3)}</strong></div>
            </article>
          </div>

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

function Player({ streamUrl, league, resolution }) {
  const videoRef = useRef(null);

  useEffect(() => {
    if (!streamUrl || !videoRef.current) return;

    let hlsInstance = null;
    if (window.Hls && window.Hls.isSupported()) {
      hlsInstance = new window.Hls();
      hlsInstance.loadSource(streamUrl);
      hlsInstance.attachMedia(videoRef.current);
    } else {
      videoRef.current.src = streamUrl;
    }

    return () => {
      if (hlsInstance) {
        hlsInstance.destroy();
      }
    };
  }, [streamUrl]);

  return (
    <div className="player-shell">
      <video ref={videoRef} controls playsInline />
      <div className="player-overlay">
        <span>{league || "لا توجد مباراة نشطة"}</span>
        <span>{resolution}</span>
      </div>
    </div>
  );
}

function MetricCard({ title, value, note, emphasis = false }) {
  return (
    <article className={`metric-card ${emphasis ? "emphasis" : ""}`}>
      <span>{title}</span>
      <strong>{value}</strong>
      <small>{note}</small>
    </article>
  );
}

ReactDOM.createRoot(document.getElementById("root")).render(<App />);
