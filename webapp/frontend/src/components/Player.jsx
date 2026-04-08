import { useEffect, useRef, useState } from "react";
import Hls from "hls.js";

const QUALITY_HEIGHTS = {
  "360p": 360,
  "480p": 480,
  "720p": 720,
  "900p": 900,
  "1080p": 1080,
  "4K": 2160,
};

const QUALITY_ORDER = ["360p", "480p", "720p", "900p", "1080p", "4K"];
const DIRECT_SOURCE_SWITCH_COOLDOWN_MS = 2200;

export default function Player({
  streamUrl,
  streamSources,
  bandwidthProbeUrl,
  league,
  resolution,
  qualityLabel,
  targetBitrateMbps,
  onBandwidthSample,
  onPlaybackTelemetry,
  connectionStatus,
  connectionTone,
  connectionSource,
}) {
  const videoRef = useRef(null);
  const hlsRef = useRef(null);
  const directSourceRef = useRef("");
  const lastDirectSwitchAtRef = useRef(0);
  const pendingDirectSwitchTimerRef = useRef(0);
  const transitionCleanupTimerRef = useRef(0);
  const bandwidthCallbackRef = useRef(onBandwidthSample);
  const playbackCallbackRef = useRef(onPlaybackTelemetry);
  const stallStateRef = useRef(createEmptyStallState());
  const [activeLevelInfo, setActiveLevelInfo] = useState("Stream level will appear here");
  const [transitionFrame, setTransitionFrame] = useState("");
  const [isTransitioningQuality, setIsTransitioningQuality] = useState(false);

  useEffect(() => {
    bandwidthCallbackRef.current = onBandwidthSample;
  }, [onBandwidthSample]);

  useEffect(() => {
    playbackCallbackRef.current = onPlaybackTelemetry;
  }, [onPlaybackTelemetry]);

  useEffect(() => {
    const video = videoRef.current;
    if ((!streamUrl && !hasDirectSources(streamSources)) || !video) {
      setActiveLevelInfo("No active stream");
      return undefined;
    }

    video.muted = true;

    if (hasDirectSources(streamSources)) {
      if (hlsRef.current) {
        hlsRef.current.destroy();
        hlsRef.current = null;
      }
      const source = pickDirectSource(streamSources, qualityLabel);
      syncDirectSource(video, source, directSourceRef, setActiveLevelInfo, {
        preserveTime: false,
        lastSwitchAtRef: lastDirectSwitchAtRef,
        pendingTimerRef: pendingDirectSwitchTimerRef,
        beginTransition: () => beginDirectTransition(video, setTransitionFrame, setIsTransitioningQuality),
        endTransition: () => endDirectTransition(setIsTransitioningQuality, transitionCleanupTimerRef),
      });
      return undefined;
    }

    let hls = null;
    directSourceRef.current = "";

    if (Hls.isSupported()) {
      hls = new Hls({
        enableWorker: true,
        autoStartLoad: true,
      });
      hlsRef.current = hls;
      hls.loadSource(streamUrl);
      hls.attachMedia(video);

      hls.on(Hls.Events.MANIFEST_PARSED, () => {
        applyRequestedLevel(hls, qualityLabel, targetBitrateMbps, setActiveLevelInfo);
        reportBandwidthSample(bandwidthCallbackRef, hls.bandwidthEstimate / 1_000_000, "HLS player telemetry");
        video.play().catch(() => {});
      });

      hls.on(Hls.Events.FRAG_LOADED, () => {
        reportBandwidthSample(bandwidthCallbackRef, hls.bandwidthEstimate / 1_000_000, "HLS fragment telemetry");
      });

      hls.on(Hls.Events.LEVEL_SWITCHED, (_, data) => {
        const activeLevel = hls.levels[data.level];
        setActiveLevelInfo(formatLevelLabel(activeLevel));
      });

      hls.on(Hls.Events.ERROR, (_, data) => {
        if (data?.fatal) {
          setActiveLevelInfo("Stream error while switching quality");
        }
      });
    } else if (video.canPlayType("application/vnd.apple.mpegurl")) {
      video.src = streamUrl;
      setActiveLevelInfo("Native HLS playback");
      video.play().catch(() => {});
    } else {
      setActiveLevelInfo("HLS playback is not supported in this browser");
    }

    return () => {
      window.clearTimeout(pendingDirectSwitchTimerRef.current);
      window.clearTimeout(transitionCleanupTimerRef.current);
      if (hlsRef.current) {
        hlsRef.current.destroy();
        hlsRef.current = null;
      }
    };
  }, [streamUrl, streamSources]);

  useEffect(() => {
    const video = videoRef.current;
    if (!video) return;

    if (hasDirectSources(streamSources)) {
      const source = pickDirectSource(streamSources, qualityLabel);
      syncDirectSource(video, source, directSourceRef, setActiveLevelInfo, {
        preserveTime: true,
        lastSwitchAtRef: lastDirectSwitchAtRef,
        pendingTimerRef: pendingDirectSwitchTimerRef,
        beginTransition: () => beginDirectTransition(video, setTransitionFrame, setIsTransitioningQuality),
        endTransition: () => endDirectTransition(setIsTransitioningQuality, transitionCleanupTimerRef),
      });
      return;
    }

    if (!hlsRef.current) return;
    applyRequestedLevel(hlsRef.current, qualityLabel, targetBitrateMbps, setActiveLevelInfo);
  }, [qualityLabel, targetBitrateMbps, streamUrl, streamSources]);

  useEffect(() => {
    if (!hasDirectSources(streamSources) || !bandwidthProbeUrl) {
      return undefined;
    }

    let cancelled = false;
    let timerId = 0;

    const runProbe = async () => {
      const measuredMbps = await measureProbeMbps(bandwidthProbeUrl);
      if (!cancelled) {
        reportBandwidthSample(bandwidthCallbackRef, measuredMbps, "live bandwidth probe");
        timerId = window.setTimeout(runProbe, 7000);
      }
    };

    runProbe();
    return () => {
      cancelled = true;
      window.clearTimeout(timerId);
    };
  }, [streamSources, bandwidthProbeUrl]);

  useEffect(() => {
    const video = videoRef.current;
    if (!video) {
      return undefined;
    }

    stallStateRef.current = createEmptyStallState();

    const emitPlaybackTelemetry = () => {
      reportPlaybackTelemetry(video, playbackCallbackRef, stallStateRef);
    };

    const beginStall = () => {
      if (
        video.paused ||
        video.seeking ||
        video.ended ||
        stallStateRef.current.isStalling ||
        !stallStateRef.current.hasStartedPlayback
      ) {
        emitPlaybackTelemetry();
        return;
      }

      stallStateRef.current = {
        ...stallStateRef.current,
        isStalling: true,
        startedAt: performance.now(),
        stallCount: stallStateRef.current.stallCount + 1,
      };
      emitPlaybackTelemetry();
    };

    const endStall = () => {
      if (!stallStateRef.current.isStalling) {
        emitPlaybackTelemetry();
        return;
      }

      const startedAt = stallStateRef.current.startedAt || performance.now();
      const elapsedSeconds = Math.max((performance.now() - startedAt) / 1000, 0);
      stallStateRef.current = {
        ...stallStateRef.current,
        isStalling: false,
        startedAt: 0,
        totalSeconds: stallStateRef.current.totalSeconds + elapsedSeconds,
      };
      emitPlaybackTelemetry();
    };

    const markPlaybackStarted = () => {
      if (stallStateRef.current.hasStartedPlayback) {
        return;
      }

      stallStateRef.current = {
        ...stallStateRef.current,
        hasStartedPlayback: true,
      };
      emitPlaybackTelemetry();
    };

    const handlePlaying = () => {
      markPlaybackStarted();
      endStall();
    };

    const handleTimeUpdate = () => {
      if (video.currentTime > 0.05) {
        markPlaybackStarted();
      }
      emitPlaybackTelemetry();
    };

    const pollingId = window.setInterval(emitPlaybackTelemetry, 1000);

    video.addEventListener("loadedmetadata", emitPlaybackTelemetry);
    video.addEventListener("loadeddata", emitPlaybackTelemetry);
    video.addEventListener("waiting", beginStall);
    video.addEventListener("stalled", beginStall);
    video.addEventListener("playing", handlePlaying);
    video.addEventListener("canplay", endStall);
    video.addEventListener("canplaythrough", endStall);
    video.addEventListener("seeked", endStall);
    video.addEventListener("pause", endStall);
    video.addEventListener("timeupdate", handleTimeUpdate);
    video.addEventListener("progress", emitPlaybackTelemetry);

    emitPlaybackTelemetry();

    return () => {
      window.clearInterval(pollingId);
      video.removeEventListener("loadedmetadata", emitPlaybackTelemetry);
      video.removeEventListener("loadeddata", emitPlaybackTelemetry);
      video.removeEventListener("waiting", beginStall);
      video.removeEventListener("stalled", beginStall);
      video.removeEventListener("playing", handlePlaying);
      video.removeEventListener("canplay", endStall);
      video.removeEventListener("canplaythrough", endStall);
      video.removeEventListener("seeked", endStall);
      video.removeEventListener("pause", endStall);
      video.removeEventListener("timeupdate", handleTimeUpdate);
      video.removeEventListener("progress", emitPlaybackTelemetry);
    };
  }, [streamUrl, streamSources]);

  return (
    <div className="player-shell">
      <video ref={videoRef} controls playsInline muted autoPlay preload="auto" />
      {isTransitioningQuality ? (
        <div className={`player-transition ${transitionFrame ? "has-frame" : "fallback"}`}>
          {transitionFrame ? <img src={transitionFrame} alt="" /> : <span>Switching quality...</span>}
        </div>
      ) : null}
      <div className="player-overlay">
        <span>{league || "No active match"}</span>
        <span>{`${qualityLabel || "Auto"} | ${resolution}`}</span>
      </div>
      <div className="player-status-row">
        <span className={`player-status-badge status-${connectionTone || "degraded"}`}>
          {`Connection: ${connectionStatus || "Degraded"}`}
        </span>
        <span className="player-status-badge">{`Target quality: ${qualityLabel || "Auto"}`}</span>
        <span className="player-status-badge accent">{activeLevelInfo}</span>
        <span className="player-status-badge">{`Telemetry: ${connectionSource || "player"}`}</span>
      </div>
    </div>
  );
}

function createEmptyStallState() {
  return {
    hasStartedPlayback: false,
    isStalling: false,
    startedAt: 0,
    totalSeconds: 0,
    stallCount: 0,
  };
}

function hasDirectSources(streamSources) {
  return Boolean(streamSources && Object.keys(streamSources).length);
}

function syncDirectSource(video, source, directSourceRef, setActiveLevelInfo, options) {
  const {
    preserveTime,
    lastSwitchAtRef,
    pendingTimerRef,
    beginTransition,
    endTransition,
  } = options;

  if (!source?.url) {
    setActiveLevelInfo("No direct source available");
    return;
  }

  setActiveLevelInfo(`Active source: ${source.label} | Official Blender file`);

  if (directSourceRef.current === source.url) {
    video.play().catch(() => {});
    return;
  }

  const now = Date.now();
  const timeSinceLastSwitch = now - (lastSwitchAtRef.current || 0);
  if (preserveTime && directSourceRef.current && timeSinceLastSwitch < DIRECT_SOURCE_SWITCH_COOLDOWN_MS) {
    window.clearTimeout(pendingTimerRef.current);
    pendingTimerRef.current = window.setTimeout(() => {
      syncDirectSource(video, source, directSourceRef, setActiveLevelInfo, options);
    }, DIRECT_SOURCE_SWITCH_COOLDOWN_MS - timeSinceLastSwitch);
    setActiveLevelInfo(`Preparing switch to ${source.label}...`);
    return;
  }

  const previousTime = preserveTime ? video.currentTime || 0 : 0;
  const shouldResume = !video.paused || video.autoplay;
  if (preserveTime && directSourceRef.current) {
    beginTransition();
  }

  lastSwitchAtRef.current = now;
  directSourceRef.current = source.url;

  const restorePlayback = () => {
    if (previousTime > 0) {
      try {
        const safeTime = video.duration ? Math.min(previousTime, Math.max(video.duration - 0.5, 0)) : previousTime;
        video.currentTime = safeTime;
      } catch {
        // Ignore seek restore issues on source switches.
      }
    }
    if (shouldResume) {
      video.play().catch(() => {});
    }
    endTransition();
  };

  const failTransition = () => {
    endTransition();
  };

  video.addEventListener("error", failTransition, { once: true });
  video.addEventListener("loadedmetadata", restorePlayback, { once: true });
  video.src = source.url;
  video.load();
}

function applyRequestedLevel(hls, qualityLabel, targetBitrateMbps, setActiveLevelInfo) {
  if (!hls?.levels?.length) {
    setActiveLevelInfo("Single stream detected");
    return;
  }

  const selectedIndex = pickClosestLevel(hls.levels, qualityLabel, targetBitrateMbps);
  if (selectedIndex < 0) {
    setActiveLevelInfo("No matching stream level");
    return;
  }

  hls.loadLevel = selectedIndex;
  hls.nextLevel = selectedIndex;
  hls.currentLevel = selectedIndex;
  setActiveLevelInfo(formatLevelLabel(hls.levels[selectedIndex]));
}

function pickClosestLevel(levels, qualityLabel, targetBitrateMbps) {
  const targetHeight = QUALITY_HEIGHTS[qualityLabel] ?? null;
  const targetBitrate = typeof targetBitrateMbps === "number" ? targetBitrateMbps * 1_000_000 : null;

  return levels.reduce(
    (best, level, index) => {
      const levelBitrate = level.bitrate || 1;
      const levelHeight = level.height || parseResolutionHeight(level) || 0;
      const bitrateScore = targetBitrate ? Math.abs(Math.log(levelBitrate / targetBitrate)) : 0;
      const heightScore = targetHeight && levelHeight ? Math.abs(levelHeight - targetHeight) / 240 : 0;
      const totalScore = bitrateScore + heightScore;

      if (totalScore < best.score) {
        return { index, score: totalScore };
      }

      return best;
    },
    { index: -1, score: Number.POSITIVE_INFINITY }
  ).index;
}

function pickDirectSource(streamSources, qualityLabel) {
  const sources = Object.entries(streamSources || {})
    .filter(([, url]) => Boolean(url))
    .map(([label, url]) => ({
      label,
      url,
      order: QUALITY_ORDER.indexOf(label),
    }))
    .filter((item) => item.order >= 0)
    .sort((left, right) => left.order - right.order);

  if (!sources.length) {
    return null;
  }

  const targetOrder = QUALITY_ORDER.indexOf(qualityLabel);
  if (targetOrder < 0) {
    return sources[sources.length - 1];
  }

  return sources.reduce((best, current) => {
    const currentDistance = Math.abs(current.order - targetOrder);
    const bestDistance = Math.abs(best.order - targetOrder);
    return currentDistance < bestDistance ? current : best;
  });
}

function parseResolutionHeight(level) {
  const resolution = level?.attrs?.RESOLUTION;
  if (!resolution || !String(resolution).includes("x")) return 0;
  return Number(String(resolution).split("x")[1]) || 0;
}

function formatLevelLabel(level) {
  if (!level) return "Unknown stream level";
  const height = level.height || parseResolutionHeight(level);
  const bitrateMbps = level.bitrate ? (level.bitrate / 1_000_000).toFixed(2) : "N/A";
  return `Active stream: ${height || "?"}p @ ${bitrateMbps} Mbps`;
}

function reportBandwidthSample(callbackRef, measuredMbps, source) {
  if (!callbackRef.current || !Number.isFinite(measuredMbps) || measuredMbps <= 0) {
    return;
  }

  callbackRef.current({
    mbps: measuredMbps,
    source,
  });
}

function beginDirectTransition(video, setTransitionFrame, setIsTransitioningQuality) {
  let snapshot = "";

  try {
    if (video.videoWidth > 0 && video.videoHeight > 0) {
      const canvas = document.createElement("canvas");
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      const context = canvas.getContext("2d");
      context?.drawImage(video, 0, 0, canvas.width, canvas.height);
      snapshot = canvas.toDataURL("image/jpeg", 0.82);
    }
  } catch {
    snapshot = "";
  }

  setTransitionFrame(snapshot);
  setIsTransitioningQuality(true);
}

function endDirectTransition(setIsTransitioningQuality, transitionCleanupTimerRef) {
  window.clearTimeout(transitionCleanupTimerRef.current);
  transitionCleanupTimerRef.current = window.setTimeout(() => {
    setIsTransitioningQuality(false);
  }, 180);
}

function reportPlaybackTelemetry(video, callbackRef, stallStateRef) {
  if (!callbackRef.current || !video) {
    return;
  }

  const bufferSeconds = getBufferedAhead(video);
  const liveStallSeconds = stallStateRef.current.isStalling
    ? Math.max((performance.now() - stallStateRef.current.startedAt) / 1000, 0)
    : 0;

  callbackRef.current({
    bufferSeconds: Number(bufferSeconds.toFixed(2)),
    totalRebufferSeconds: Number((stallStateRef.current.totalSeconds + liveStallSeconds).toFixed(2)),
    stallCount: stallStateRef.current.stallCount,
    isStalling: stallStateRef.current.isStalling,
  });
}

function getBufferedAhead(video) {
  if (!video.buffered?.length) {
    return 0;
  }

  const currentTime = video.currentTime || 0;
  for (let index = 0; index < video.buffered.length; index += 1) {
    const start = video.buffered.start(index);
    const end = video.buffered.end(index);
    if (currentTime >= start && currentTime <= end) {
      return Math.max(end - currentTime, 0);
    }
  }

  return 0;
}

async function measureProbeMbps(url) {
  const cacheBust = url.includes("?") ? `${url}&probe=${Date.now()}` : `${url}?probe=${Date.now()}`;
  const startedAt = performance.now();

  try {
    const response = await fetch(cacheBust, {
      cache: "no-store",
      mode: "cors",
    });

    if (!response.ok) {
      return 0;
    }

    const payload = await response.blob();
    const durationSeconds = Math.max((performance.now() - startedAt) / 1000, 0.1);
    return (payload.size * 8) / durationSeconds / 1_000_000;
  } catch {
    return 0;
  }
}
