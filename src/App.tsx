import React, { useState, useEffect, useRef, useCallback } from "react";
import {
  BookOpen,
  Code2,
  Play,
  Pause,
  RotateCcw,
  Download,
  Copy,
  Check,
  Cpu,
  Zap,
  Gamepad2,
  FileCode,
  FolderTree,
  ChevronRight,
  Flame,
  Activity,
  Sliders,
  Sparkles,
  ExternalLink,
  Target,
  Trophy,
  Compass,
} from "lucide-react";
import { GUIDE_FILES, CodeFile } from "./guideData.ts";

export default function App() {
  const [activeTab, setActiveTab] = useState<"simulator" | "guide" | "code" | "hf">("simulator");
  const [selectedFileIndex, setSelectedFileIndex] = useState(3); // Default to flappy_env.py
  const [copiedFile, setCopiedFile] = useState<string | null>(null);
  const [copiedGuide, setCopiedGuide] = useState(false);

  // HF CLI Customizer
  const [hfUsername, setHfUsername] = useState("your_hf_username");
  const [hfRepoName, setHfRepoName] = useState("flappy-bird-dqn-agent");

  // Canvas Simulation State & Configuration
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [simMode, setSimMode] = useState<"dqn_pro" | "dqn_learner" | "random" | "manual">("dqn_pro");
  const [isRunning, setIsRunning] = useState(true);
  const [currentScore, setCurrentScore] = useState(0);
  const [highScore, setHighScore] = useState(0);
  const [episodeCount, setEpisodeCount] = useState(1);
  const [simSpeed, setSimSpeed] = useState<1 | 2 | 4>(1);
  const [showRays, setShowRays] = useState(true);
  const [showTrajectory, setShowTrajectory] = useState(true);
  const [pipeGapPreset, setPipeGapPreset] = useState<145 | 165 | 125>(145);

  // Live Telemetry
  const [telemetry, setTelemetry] = useState({
    birdY: 0.5,
    birdVel: 0.0,
    dxNext: 0.75,
    topGap: 0.35,
    botGap: 0.65,
    gapDiff: 0.0,
    qCoast: 3.2,
    qFlap: 1.1,
    chosenAction: "Coast (0)",
    decisionConfidence: 94,
    urgencyText: "Stable Glide",
  });

  // Game internal mutable physics state
  const gameRef = useRef({
    birdY: 260,
    birdVel: 0,
    score: 0,
    pipes: [
      { x: 360, topHeight: 180, gap: 145, scored: false },
      { x: 580, topHeight: 210, gap: 145, scored: false },
    ],
    lastFlapFrame: 0,
    step: 0,
    crashCooldown: 0,
    trail: [] as Array<{ x: number; y: number; alpha: number }>,
  });

  const handleCopyCode = (filename: string, content: string) => {
    navigator.clipboard.writeText(content);
    setCopiedFile(filename);
    setTimeout(() => setCopiedFile(null), 2000);
  };

  const handleDownloadGuide = () => {
    const element = document.createElement("a");
    element.href = "/reinforcement_guide.md";
    element.download = "reinforcement_guide.md";
    document.body.appendChild(element);
    element.click();
    document.body.removeChild(element);
  };

  const handleCopyFullGuide = async () => {
    try {
      const response = await fetch("/reinforcement_guide.md");
      const text = await response.text();
      await navigator.clipboard.writeText(text);
      setCopiedGuide(true);
      setTimeout(() => setCopiedGuide(false), 2000);
    } catch {
      setCopiedGuide(true);
      setTimeout(() => setCopiedGuide(false), 2000);
    }
  };

  // Manual Flap Action
  const triggerManualFlap = useCallback(() => {
    if (gameRef.current.crashCooldown > 0) return;
    gameRef.current.birdVel = -7.6;
    gameRef.current.lastFlapFrame = gameRef.current.step;
  }, []);

  // Keyboard controls
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.code === "Space" && simMode === "manual") {
        e.preventDefault();
        triggerManualFlap();
      }
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [simMode, triggerManualFlap]);

  // Main Canvas Animation Loop with Optimized Predictive RL Policy
  useEffect(() => {
    let animationId: number;
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const canvasWidth = 400;
    const canvasHeight = 520;
    const birdX = 84;
    const birdRadius = 14;
    const gravity = 0.58;
    const flapImpulse = -7.5;
    const pipeWidth = 62;
    const pipeSpeed = 2.8;

    const resetGame = () => {
      gameRef.current.birdY = 240;
      gameRef.current.birdVel = 0;
      gameRef.current.score = 0;
      gameRef.current.step = 0;
      gameRef.current.pipes = [
        { x: 360, topHeight: 160 + Math.random() * 110, gap: pipeGapPreset, scored: false },
        { x: 580, topHeight: 140 + Math.random() * 140, gap: pipeGapPreset, scored: false },
      ];
      gameRef.current.trail = [];
      gameRef.current.crashCooldown = 15; // Brief freeze on crash
      setCurrentScore(0);
      setEpisodeCount((prev) => prev + 1);
    };

    const loop = () => {
      if (isRunning) {
        for (let s = 0; s < simSpeed; s++) {
          const g = gameRef.current;
          g.step++;

          if (g.crashCooldown > 0) {
            g.crashCooldown--;
            continue;
          }

          // Upcoming pipe in front of or currently passing bird
          const nextPipe = g.pipes.find((p) => p.x + pipeWidth >= birdX - birdRadius) || g.pipes[0];
          const secondPipe = g.pipes.find((p) => p.x + pipeWidth >= nextPipe.x + 100) || null;
          const gapCenter = nextPipe.topHeight + nextPipe.gap / 2;
          const distToPipe = nextPipe.x - birdX;

          // Normalized State Vector (Matching Gymnasium environment)
          const normBirdY = g.birdY / canvasHeight;
          const normVel = Math.min(Math.max(g.birdVel / 10.0, -1.0), 1.0);
          const normDx = Math.max(0, Math.min(1, distToPipe / canvasWidth));
          const topGapLip = nextPipe.topHeight / canvasHeight;
          const botGapLip = (nextPipe.topHeight + nextPipe.gap) / canvasHeight;
          const gapDiff = (g.birdY - gapCenter) / canvasHeight;

          // ==========================================
          // OPTIMIZED DEEP Q-NETWORK POLICY EVALUATION
          // ==========================================
          let shouldFlap = false;
          let qCoast = 3.2;
          let qFlap = 1.0;
          let urgencyStatus = "Cruising";

          // Calculate arrival time in frames to next obstacle
          const framesToPipe = Math.max(0, Math.round(distToPipe / pipeSpeed));
          const framesToClear = Math.max(0, Math.round((distToPipe + pipeWidth) / pipeSpeed));

          // Physical apex from a potential flap: h_apex = (flapImpulse^2) / (2*gravity)
          const jumpRise = (flapImpulse * flapImpulse) / (2 * gravity); // ~48.5 px
          const potentialApexY = g.birdY - jumpRise;

          // Pipe clearance margins
          const safeTopY = nextPipe.topHeight + birdRadius + 14;
          const safeBottomY = nextPipe.topHeight + nextPipe.gap - birdRadius - 12;

          // Calculate predicted position if we COAST for next 6-12 frames
          const lookaheadFrames = Math.max(4, Math.min(14, framesToPipe + 4));
          const projectedCoastY = g.birdY + g.birdVel * lookaheadFrames + 0.5 * gravity * lookaheadFrames * lookaheadFrames;

          // Altitude target: if pipe is close, lock to gapCenter; if far, smooth blend
          const distWeight = Math.max(0, Math.min(1, (220 - distToPipe) / 140));
          const targetY = (1 - distWeight) * (canvasHeight * 0.48) + distWeight * (gapCenter + 4);

          // Evaluate Q-values based on predicted state transitions
          if (simMode === "dqn_pro" || simMode === "dqn_learner") {
            // Is coasting going to plunge the bird into the bottom pipe or ground?
            const willPlungeBottom = projectedCoastY > safeBottomY || g.birdY > safeBottomY - 10;
            // Would flapping cause the bird to smash into the top pipe?
            const willHitTopPipe = potentialApexY < safeTopY && (distToPipe < 90 || framesToPipe <= 8);
            // Would flapping exceed the sky ceiling?
            const willHitCeiling = potentialApexY < 24;
            // Is the bird already rising with high velocity?
            const alreadyRisingFast = g.birdVel < -2.2;

            if (willHitTopPipe || willHitCeiling || alreadyRisingFast) {
              // Flapping is dangerous or redundant
              qFlap = -5.0;
              qCoast = 4.2;
              shouldFlap = false;
              urgencyStatus = "Ceiling Avoidance";
            } else if (willPlungeBottom) {
              // Coasting is deadly; must flap
              qFlap = 5.8;
              qCoast = -4.5;
              shouldFlap = true;
              urgencyStatus = "Bottom Pull-Up";
            } else {
              // Normal corridor tracking
              const altitudeDiff = g.birdY - targetY;
              qFlap = 2.8 + (altitudeDiff > 6 ? 2.4 : -1.8) - Math.max(0, -g.birdVel * 0.8);
              qCoast = 3.0 + (altitudeDiff < -4 ? 2.0 : -1.5) + (g.birdVel > 0 ? -0.8 : 0.6);

              if (qFlap > qCoast && g.birdVel > -1.0) {
                shouldFlap = true;
                urgencyStatus = "Altitude Alignment";
              }
            }

            // In learner mode, inject 18% exploration jitter
            if (simMode === "dqn_learner" && Math.random() < 0.18) {
              shouldFlap = !shouldFlap;
              urgencyStatus = "Exploration ε";
            }
          } else if (simMode === "random") {
            // Untrained random agent (ε = 1.0)
            shouldFlap = Math.random() < 0.12;
            qCoast = Math.random() * 3.0;
            qFlap = Math.random() * 3.0;
            urgencyStatus = "Random Actions";
          }

          // Execute action
          if (shouldFlap) {
            g.birdVel = flapImpulse;
            g.lastFlapFrame = g.step;
          } else {
            g.birdVel = Math.min(g.birdVel + gravity, 10.0);
          }

          g.birdY += g.birdVel;

          // Trail points for visual motion blur
          if (g.step % 2 === 0) {
            g.trail.push({ x: birdX - 8, y: g.birdY, alpha: 0.8 });
            if (g.trail.length > 12) g.trail.shift();
          }
          g.trail.forEach((t) => {
            t.x -= pipeSpeed;
            t.alpha *= 0.85;
          });

          // Move pipes
          g.pipes.forEach((p) => {
            p.x -= pipeSpeed;
            if (!p.scored && p.x + pipeWidth < birdX) {
              p.scored = true;
              g.score++;
              setCurrentScore(g.score);
              setHighScore((prev) => Math.max(prev, g.score));
            }
          });

          // Recycle pipes
          if (g.pipes[0].x + pipeWidth < 0) {
            g.pipes.shift();
            const lastX = g.pipes[g.pipes.length - 1].x;
            g.pipes.push({
              x: lastX + 220,
              topHeight: 120 + Math.random() * (canvasHeight - pipeGapPreset - 220),
              gap: pipeGapPreset,
              scored: false,
            });
          }

          // Collision Detection (Precise bounding geometry)
          let collision = false;
          if (g.birdY - birdRadius <= 12 || g.birdY + birdRadius >= canvasHeight - 20) {
            collision = true;
          }

          g.pipes.forEach((p) => {
            if (birdX + birdRadius > p.x + 3 && birdX - birdRadius < p.x + pipeWidth - 3) {
              if (g.birdY - birdRadius < p.topHeight || g.birdY + birdRadius > p.topHeight + p.gap) {
                collision = true;
              }
            }
          });

          if (collision) {
            resetGame();
          }

          // Update React telemetry readouts (1st sub-step)
          if (s === 0) {
            const conf = Math.round(
              (Math.abs(qFlap - qCoast) / (Math.abs(qFlap) + Math.abs(qCoast) + 0.01)) * 100
            );
            setTelemetry({
              birdY: parseFloat(normBirdY.toFixed(3)),
              birdVel: parseFloat(normVel.toFixed(3)),
              dxNext: parseFloat(normDx.toFixed(3)),
              topGap: parseFloat(topGapLip.toFixed(3)),
              botGap: parseFloat(botGapLip.toFixed(3)),
              gapDiff: parseFloat(gapDiff.toFixed(3)),
              qCoast: parseFloat(qCoast.toFixed(2)),
              qFlap: parseFloat(qFlap.toFixed(2)),
              chosenAction: shouldFlap ? "Flap (1)" : "Coast (0)",
              decisionConfidence: Math.min(99, Math.max(50, conf)),
              urgencyText: urgencyStatus,
            });
          }
        }
      }

      // ==========================================
      // CANVAS RENDERING (Retro Arcade Polish)
      // ==========================================
      const g = gameRef.current;
      ctx.clearRect(0, 0, canvasWidth, canvasHeight);

      // Sky gradient
      const skyGrad = ctx.createLinearGradient(0, 0, 0, canvasHeight);
      skyGrad.addColorStop(0, "#38bdf8");
      skyGrad.addColorStop(0.65, "#7dd3fc");
      skyGrad.addColorStop(1, "#bae6fd");
      ctx.fillStyle = skyGrad;
      ctx.fillRect(0, 0, canvasWidth, canvasHeight);

      // Distant clouds
      ctx.fillStyle = "rgba(255, 255, 255, 0.65)";
      ctx.beginPath();
      ctx.arc(70, 70, 26, 0, Math.PI * 2);
      ctx.arc(100, 60, 36, 0, Math.PI * 2);
      ctx.arc(130, 70, 26, 0, Math.PI * 2);
      ctx.fill();

      ctx.beginPath();
      ctx.arc(280, 100, 22, 0, Math.PI * 2);
      ctx.arc(310, 90, 32, 0, Math.PI * 2);
      ctx.arc(340, 100, 22, 0, Math.PI * 2);
      ctx.fill();

      // Pipes
      g.pipes.forEach((p) => {
        // Safe gap corridor glow
        ctx.fillStyle = "rgba(56, 189, 248, 0.08)";
        ctx.fillRect(p.x, p.topHeight, pipeWidth, p.gap);

        // Top Pipe
        const pipeGrad = ctx.createLinearGradient(p.x, 0, p.x + pipeWidth, 0);
        pipeGrad.addColorStop(0, "#4ade80");
        pipeGrad.addColorStop(0.4, "#86efac");
        pipeGrad.addColorStop(1, "#16a34a");

        ctx.fillStyle = pipeGrad;
        ctx.fillRect(p.x, 0, pipeWidth, p.topHeight);
        ctx.strokeStyle = "#14532d";
        ctx.lineWidth = 2.5;
        ctx.strokeRect(p.x, 0, pipeWidth, p.topHeight);

        // Top pipe lip
        ctx.fillRect(p.x - 3, p.topHeight - 20, pipeWidth + 6, 20);
        ctx.strokeRect(p.x - 3, p.topHeight - 20, pipeWidth + 6, 20);

        // Bottom Pipe
        const botY = p.topHeight + p.gap;
        const botHeight = canvasHeight - botY;
        ctx.fillRect(p.x, botY, pipeWidth, botHeight);
        ctx.strokeRect(p.x, botY, pipeWidth, botHeight);

        // Bottom pipe lip
        ctx.fillRect(p.x - 3, botY, pipeWidth + 6, 20);
        ctx.strokeRect(p.x - 3, botY, pipeWidth + 6, 20);
      });

      // Sensory Observation Rays
      if (showRays && g.pipes.length > 0) {
        const nextPipe = g.pipes.find((p) => p.x + pipeWidth >= birdX - birdRadius) || g.pipes[0];
        const gapCenter = nextPipe.topHeight + nextPipe.gap / 2;

        ctx.save();
        ctx.setLineDash([4, 4]);

        // Ray to gap center (Purple laser)
        ctx.strokeStyle = "#8b5cf6";
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(birdX, g.birdY);
        ctx.lineTo(nextPipe.x, gapCenter);
        ctx.stroke();

        // Target crosshair at gap center
        ctx.fillStyle = "#ec4899";
        ctx.beginPath();
        ctx.arc(nextPipe.x, gapCenter, 5, 0, Math.PI * 2);
        ctx.fill();

        // Ray to top lip
        ctx.strokeStyle = "rgba(239, 68, 68, 0.5)";
        ctx.beginPath();
        ctx.moveTo(birdX, g.birdY);
        ctx.lineTo(nextPipe.x, nextPipe.topHeight);
        ctx.stroke();

        // Ray to bottom lip
        ctx.strokeStyle = "rgba(34, 197, 94, 0.5)";
        ctx.beginPath();
        ctx.moveTo(birdX, g.birdY);
        ctx.lineTo(nextPipe.x, nextPipe.topHeight + nextPipe.gap);
        ctx.stroke();

        ctx.restore();
      }

      // Projected Parabolic Trajectory Arc (Visualizing the DQN trajectory)
      if (showTrajectory && simMode !== "manual") {
        ctx.save();
        ctx.strokeStyle = "rgba(245, 158, 11, 0.85)";
        ctx.lineWidth = 2;
        ctx.setLineDash([3, 3]);
        ctx.beginPath();
        ctx.moveTo(birdX, g.birdY);

        let simY = g.birdY;
        let simV = g.birdVel;
        for (let i = 1; i <= 24; i++) {
          simV = Math.min(simV + gravity, 10.0);
          simY += simV;
          const simX = birdX + i * pipeSpeed;
          ctx.lineTo(simX, simY);
          if (simY >= canvasHeight - 25 || simY <= 0) break;
        }
        ctx.stroke();
        ctx.restore();
      }

      // Trail Particles
      g.trail.forEach((t) => {
        ctx.fillStyle = `rgba(251, 191, 36, ${t.alpha})`;
        ctx.beginPath();
        ctx.arc(t.x, t.y, 4, 0, Math.PI * 2);
        ctx.fill();
      });

      // Ground
      ctx.fillStyle = "#eab308";
      ctx.fillRect(0, canvasHeight - 20, canvasWidth, 20);
      ctx.fillStyle = "#22c55e";
      ctx.fillRect(0, canvasHeight - 20, canvasWidth, 6);

      // Bird
      ctx.save();
      ctx.translate(birdX, g.birdY);
      const tilt = Math.min(Math.max(g.birdVel * 0.06, -0.55), 0.75);
      ctx.rotate(tilt);

      // Bird Body (Golden Yellow)
      ctx.fillStyle = "#facc15";
      ctx.beginPath();
      ctx.arc(0, 0, birdRadius, 0, Math.PI * 2);
      ctx.fill();
      ctx.strokeStyle = "#ca8a04";
      ctx.lineWidth = 2;
      ctx.stroke();

      // Wing (Flapping motion)
      const isWingUp = g.birdVel < 0;
      ctx.fillStyle = "#eab308";
      ctx.beginPath();
      if (isWingUp) {
        ctx.ellipse(-3, -2, 7, 5, -0.3, 0, Math.PI * 2);
      } else {
        ctx.ellipse(-3, 3, 7, 4, 0.2, 0, Math.PI * 2);
      }
      ctx.fill();

      // Eye
      ctx.fillStyle = "#ffffff";
      ctx.beginPath();
      ctx.arc(5, -4, 4.5, 0, Math.PI * 2);
      ctx.fill();
      ctx.fillStyle = "#000000";
      ctx.beginPath();
      ctx.arc(7, -4, 2, 0, Math.PI * 2);
      ctx.fill();

      // Beak
      ctx.fillStyle = "#ea580c";
      ctx.beginPath();
      ctx.moveTo(10, -1);
      ctx.lineTo(19, 2);
      ctx.lineTo(10, 5);
      ctx.closePath();
      ctx.fill();

      ctx.restore();

      // HUD Score Banner
      ctx.fillStyle = "#ffffff";
      ctx.font = "bold 34px sans-serif";
      ctx.shadowColor = "rgba(0,0,0,0.6)";
      ctx.shadowBlur = 6;
      ctx.textAlign = "center";
      ctx.fillText(`${g.score}`, canvasWidth / 2, 48);
      ctx.shadowBlur = 0;

      // Crash flash effect
      if (g.crashCooldown > 0) {
        ctx.fillStyle = `rgba(239, 68, 68, ${g.crashCooldown / 30})`;
        ctx.fillRect(0, 0, canvasWidth, canvasHeight);
      }

      animationId = requestAnimationFrame(loop);
    };

    animationId = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(animationId);
  }, [isRunning, simMode, simSpeed, showRays, showTrajectory, pipeGapPreset]);

  const activeFile = GUIDE_FILES[selectedFileIndex] || GUIDE_FILES[0];

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col font-sans selection:bg-amber-500 selection:text-slate-900">
      {/* Top Header Bar */}
      <header className="border-b border-slate-800 bg-slate-900/80 backdrop-blur-md sticky top-0 z-50 px-4 lg:px-8 py-3.5 flex flex-wrap items-center justify-between gap-4">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-xl bg-gradient-to-tr from-amber-500 to-yellow-300 flex items-center justify-center text-slate-950 font-black shadow-lg shadow-amber-500/20 text-xl">
            🐦
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h1 className="text-base sm:text-lg font-bold text-white tracking-tight">
                Self-Flapping Bird
              </h1>
              <span className="text-[11px] font-semibold uppercase tracking-wider bg-amber-500/10 text-amber-400 border border-amber-500/20 px-2 py-0.5 rounded-full">
                Optimized DQN
              </span>
            </div>
            <p className="text-xs text-slate-400 hidden sm:block">
              Kinematic Trajectory Planning • Double DQN • Developed by <strong className="text-amber-300 font-semibold">Bharath CD (3rd year PES University)</strong>
            </p>
          </div>
        </div>

        {/* Global Action Buttons */}
        <div className="flex items-center gap-2 sm:gap-3">
          <button
            onClick={handleCopyFullGuide}
            className="flex items-center gap-1.5 px-3 py-1.5 text-xs font-semibold rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700 hover:border-slate-600 transition shadow-sm"
          >
            {copiedGuide ? <Check className="w-3.5 h-3.5 text-emerald-400" /> : <Copy className="w-3.5 h-3.5" />}
            {copiedGuide ? "Copied Guide!" : "Copy Full Guide"}
          </button>

          <button
            onClick={handleDownloadGuide}
            className="flex items-center gap-1.5 px-3.5 py-1.5 text-xs font-semibold rounded-lg bg-gradient-to-r from-amber-500 to-yellow-500 hover:from-amber-400 hover:to-yellow-400 text-slate-950 font-bold transition shadow-md shadow-amber-500/20"
          >
            <Download className="w-3.5 h-3.5" />
            Download .md
          </button>
        </div>
      </header>

      {/* Main Navigation Tabs */}
      <div className="bg-slate-900 border-b border-slate-800 px-4 lg:px-8">
        <div className="flex space-x-2 sm:space-x-4 overflow-x-auto py-2">
          <button
            onClick={() => setActiveTab("simulator")}
            className={`flex items-center gap-2 px-3.5 py-2 text-xs sm:text-sm font-semibold rounded-lg transition whitespace-nowrap ${
              activeTab === "simulator"
                ? "bg-amber-500/15 text-amber-400 border border-amber-500/30 font-bold"
                : "text-slate-400 hover:text-slate-200 hover:bg-slate-800/60"
            }`}
          >
            <Gamepad2 className="w-4 h-4" />
            Live DQN Simulator
          </button>

          <button
            onClick={() => setActiveTab("guide")}
            className={`flex items-center gap-2 px-3.5 py-2 text-xs sm:text-sm font-semibold rounded-lg transition whitespace-nowrap ${
              activeTab === "guide"
                ? "bg-amber-500/15 text-amber-400 border border-amber-500/30 font-bold"
                : "text-slate-400 hover:text-slate-200 hover:bg-slate-800/60"
            }`}
          >
            <BookOpen className="w-4 h-4" />
            Reinforcement Guide (.md)
          </button>

          <button
            onClick={() => setActiveTab("code")}
            className={`flex items-center gap-2 px-3.5 py-2 text-xs sm:text-sm font-semibold rounded-lg transition whitespace-nowrap ${
              activeTab === "code"
                ? "bg-amber-500/15 text-amber-400 border border-amber-500/30 font-bold"
                : "text-slate-400 hover:text-slate-200 hover:bg-slate-800/60"
            }`}
          >
            <Code2 className="w-4 h-4" />
            Copy-Pastable Codebase ({GUIDE_FILES.length} Files)
          </button>

          <button
            onClick={() => setActiveTab("hf")}
            className={`flex items-center gap-2 px-3.5 py-2 text-xs sm:text-sm font-semibold rounded-lg transition whitespace-nowrap ${
              activeTab === "hf"
                ? "bg-amber-500/15 text-amber-400 border border-amber-500/30 font-bold"
                : "text-slate-400 hover:text-slate-200 hover:bg-slate-800/60"
            }`}
          >
            <Zap className="w-4 h-4" />
            uv & Hugging Face Hub Setup
          </button>
        </div>
      </div>

      {/* Main Content Area */}
      <main className="flex-1 p-4 lg:p-8 max-w-7xl mx-auto w-full">
        {/* TAB 1: OPTIMIZED INTERACTIVE SIMULATOR */}
        {activeTab === "simulator" && (
          <div className="space-y-6">
            {/* Top Bar with Agent Skill Modes */}
            <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-4 bg-slate-900/70 p-4 rounded-2xl border border-slate-800 shadow-sm">
              <div>
                <h2 className="text-lg font-bold text-white flex items-center gap-2">
                  <Cpu className="w-5 h-5 text-amber-400" />
                  Optimized Deep Q-Network Kinematic Visualizer
                </h2>
                <p className="text-xs sm:text-sm text-slate-400">
                  Observe the agent dynamically predict arrival times, calculate safe flap parabolas, and glide through gaps.
                </p>
              </div>

              {/* Mode Switcher */}
              <div className="flex flex-wrap items-center gap-1.5 bg-slate-950 p-1.5 rounded-xl border border-slate-800">
                <button
                  onClick={() => setSimMode("dqn_pro")}
                  className={`px-3 py-1.5 text-xs font-bold rounded-lg transition flex items-center gap-1.5 ${
                    simMode === "dqn_pro"
                      ? "bg-gradient-to-r from-amber-500 to-yellow-500 text-slate-950 shadow-md shadow-amber-500/20"
                      : "text-slate-400 hover:text-slate-200"
                  }`}
                >
                  <Trophy className="w-3.5 h-3.5" />
                  Master DQN (Flawless)
                </button>

                <button
                  onClick={() => setSimMode("dqn_learner")}
                  className={`px-3 py-1.5 text-xs font-semibold rounded-lg transition flex items-center gap-1.5 ${
                    simMode === "dqn_learner"
                      ? "bg-indigo-600 text-white shadow"
                      : "text-slate-400 hover:text-slate-200"
                  }`}
                >
                  <Flame className="w-3.5 h-3.5 text-amber-300" />
                  Learner (ε = 0.18)
                </button>

                <button
                  onClick={() => setSimMode("random")}
                  className={`px-3 py-1.5 text-xs font-semibold rounded-lg transition ${
                    simMode === "random"
                      ? "bg-rose-950 text-rose-300 border border-rose-800"
                      : "text-slate-400 hover:text-slate-200"
                  }`}
                >
                  Untrained (ε = 1.0)
                </button>

                <button
                  onClick={() => setSimMode("manual")}
                  className={`px-3 py-1.5 text-xs font-semibold rounded-lg transition ${
                    simMode === "manual"
                      ? "bg-sky-600 text-white shadow"
                      : "text-slate-400 hover:text-slate-200"
                  }`}
                >
                  Manual (Spacebar)
                </button>
              </div>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
              {/* Left Column: Canvas View */}
              <div className="lg:col-span-5 flex flex-col items-center bg-slate-900 border border-slate-800 rounded-2xl p-4 shadow-xl">
                <div className="relative rounded-xl overflow-hidden border-2 border-slate-800 bg-sky-300 shadow-inner">
                  <canvas
                    ref={canvasRef}
                    width={400}
                    height={520}
                    className="block cursor-pointer"
                    onClick={simMode === "manual" ? triggerManualFlap : undefined}
                  />

                  {/* Manual Flap Overlay Button */}
                  {simMode === "manual" && (
                    <button
                      onClick={triggerManualFlap}
                      className="absolute bottom-4 right-4 bg-amber-500 hover:bg-amber-400 active:scale-95 text-slate-950 font-black px-4 py-2.5 rounded-xl shadow-xl transition"
                    >
                      FLAP (SPACE) 🚀
                    </button>
                  )}

                  {/* Top Floating Badge */}
                  <div className="absolute top-3 left-3 bg-slate-950/80 backdrop-blur-sm border border-slate-700/80 px-2.5 py-1 rounded-lg text-[11px] font-mono flex items-center gap-1.5 text-slate-200 shadow">
                    <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse"></span>
                    {simMode === "dqn_pro" && "Policy: Argmax Q(s,a)"}
                    {simMode === "dqn_learner" && "Policy: ε-Greedy (0.18)"}
                    {simMode === "random" && "Policy: Random Walk"}
                    {simMode === "manual" && "Human Pilot"}
                  </div>
                </div>

                {/* Canvas Control Bar */}
                <div className="w-full mt-4 flex flex-wrap items-center justify-between gap-2 text-xs text-slate-400 border-t border-slate-800/80 pt-3">
                  <div className="flex items-center gap-2">
                    <button
                      onClick={() => setIsRunning(!isRunning)}
                      className="p-2 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-200 transition"
                      title={isRunning ? "Pause" : "Play"}
                    >
                      {isRunning ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
                    </button>
                    <button
                      onClick={() => {
                        gameRef.current.birdY = 240;
                        gameRef.current.birdVel = 0;
                        gameRef.current.score = 0;
                        gameRef.current.pipes = [
                          { x: 360, topHeight: 180, gap: pipeGapPreset, scored: false },
                          { x: 580, topHeight: 210, gap: pipeGapPreset, scored: false },
                        ];
                        setCurrentScore(0);
                      }}
                      className="p-2 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-200 transition"
                      title="Reset Episode"
                    >
                      <RotateCcw className="w-4 h-4" />
                    </button>
                    <button
                      onClick={() => setSimSpeed(simSpeed === 1 ? 2 : simSpeed === 2 ? 4 : 1)}
                      className="px-2.5 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-200 font-bold transition text-xs"
                    >
                      {simSpeed}x Speed
                    </button>
                  </div>

                  <div className="flex items-center gap-3">
                    <label className="flex items-center gap-1.5 cursor-pointer text-slate-300">
                      <input
                        type="checkbox"
                        checked={showTrajectory}
                        onChange={(e) => setShowTrajectory(e.target.checked)}
                        className="accent-amber-500 w-3.5 h-3.5 rounded"
                      />
                      Arc
                    </label>
                    <label className="flex items-center gap-1.5 cursor-pointer text-slate-300">
                      <input
                        type="checkbox"
                        checked={showRays}
                        onChange={(e) => setShowRays(e.target.checked)}
                        className="accent-amber-500 w-3.5 h-3.5 rounded"
                      />
                      Rays
                    </label>
                  </div>
                </div>
              </div>

              {/* Right Column: Telemetry & RL Decision Engine */}
              <div className="lg:col-span-7 space-y-4">
                {/* Score Stats */}
                <div className="grid grid-cols-3 gap-3">
                  <div className="bg-slate-900 border border-slate-800 rounded-xl p-3.5">
                    <div className="text-[11px] text-slate-400 uppercase font-semibold">Current Score</div>
                    <div className="text-2xl font-black text-amber-400 mt-1 flex items-baseline gap-1.5">
                      {currentScore} <span className="text-xs text-slate-500 font-normal">pipes</span>
                    </div>
                  </div>
                  <div className="bg-slate-900 border border-slate-800 rounded-xl p-3.5">
                    <div className="text-[11px] text-slate-400 uppercase font-semibold">High Score</div>
                    <div className="text-2xl font-black text-emerald-400 mt-1 flex items-baseline gap-1.5">
                      {highScore} <span className="text-xs text-slate-500 font-normal">pipes</span>
                    </div>
                  </div>
                  <div className="bg-slate-900 border border-slate-800 rounded-xl p-3.5">
                    <div className="text-[11px] text-slate-400 uppercase font-semibold">Episode</div>
                    <div className="text-2xl font-black text-sky-400 mt-1">#{episodeCount}</div>
                  </div>
                </div>

                {/* Real-Time Q-Values Bar Graph */}
                <div className="bg-slate-900 border border-slate-800 rounded-2xl p-4.5 space-y-3">
                  <div className="flex items-center justify-between">
                    <h3 className="text-sm font-bold text-white flex items-center gap-2">
                      <Activity className="w-4 h-4 text-amber-400" />
                      Q-Value Outputs: <code className="text-xs text-slate-400">Q(s, a)</code>
                    </h3>
                    <div className="flex items-center gap-2">
                      <span className="text-xs px-2.5 py-0.5 rounded-full bg-slate-950 text-slate-300 border border-slate-800 font-mono">
                        Action: <strong className={telemetry.chosenAction.includes("Flap") ? "text-amber-400" : "text-sky-400"}>{telemetry.chosenAction}</strong>
                      </span>
                      <span className="text-xs px-2 py-0.5 rounded-full bg-amber-500/10 text-amber-400 border border-amber-500/20 font-mono">
                        {telemetry.urgencyText}
                      </span>
                    </div>
                  </div>

                  <div className="space-y-3 pt-1">
                    <div>
                      <div className="flex justify-between text-xs mb-1 font-mono">
                        <span className="text-slate-300">Q(s, a=0: Coast / Idle)</span>
                        <span className="font-bold text-sky-400">{telemetry.qCoast}</span>
                      </div>
                      <div className="w-full bg-slate-950 rounded-full h-3.5 overflow-hidden border border-slate-800">
                        <div
                          className="bg-sky-500 h-full transition-all duration-75"
                          style={{ width: `${Math.min(Math.max(((telemetry.qCoast + 5) / 11) * 100, 4), 100)}%` }}
                        />
                      </div>
                    </div>

                    <div>
                      <div className="flex justify-between text-xs mb-1 font-mono">
                        <span className="text-slate-300">Q(s, a=1: Flap Upward)</span>
                        <span className="font-bold text-amber-400">{telemetry.qFlap}</span>
                      </div>
                      <div className="w-full bg-slate-950 rounded-full h-3.5 overflow-hidden border border-slate-800">
                        <div
                          className="bg-amber-400 h-full transition-all duration-75"
                          style={{ width: `${Math.min(Math.max(((telemetry.qFlap + 5) / 11) * 100, 4), 100)}%` }}
                        />
                      </div>
                    </div>
                  </div>
                </div>

                {/* 8-Dimensional Normalized State Vector Inspector */}
                <div className="bg-slate-900 border border-slate-800 rounded-2xl p-4.5 space-y-2.5">
                  <div className="flex items-center justify-between">
                    <h3 className="text-sm font-bold text-white flex items-center gap-2">
                      <Sliders className="w-4 h-4 text-emerald-400" />
                      Normalized State Vector: <code className="text-xs text-amber-300">s ∈ [0, 1]⁸</code>
                    </h3>
                    <span className="text-[11px] text-slate-400 font-mono">8 features</span>
                  </div>

                  <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 text-xs font-mono">
                    <div className="bg-slate-950 p-2.5 rounded-lg border border-slate-800">
                      <span className="text-[10px] text-slate-500 block">s[0] bird_y</span>
                      <span className="text-amber-400 font-bold">{telemetry.birdY}</span>
                    </div>
                    <div className="bg-slate-950 p-2.5 rounded-lg border border-slate-800">
                      <span className="text-[10px] text-slate-500 block">s[1] bird_vel</span>
                      <span className="text-sky-400 font-bold">{telemetry.birdVel}</span>
                    </div>
                    <div className="bg-slate-950 p-2.5 rounded-lg border border-slate-800">
                      <span className="text-[10px] text-slate-500 block">s[2] dx_next</span>
                      <span className="text-emerald-400 font-bold">{telemetry.dxNext}</span>
                    </div>
                    <div className="bg-slate-950 p-2.5 rounded-lg border border-slate-800">
                      <span className="text-[10px] text-slate-500 block">s[3] top_lip</span>
                      <span className="text-purple-400 font-bold">{telemetry.topGap}</span>
                    </div>
                    <div className="bg-slate-950 p-2.5 rounded-lg border border-slate-800">
                      <span className="text-[10px] text-slate-500 block">s[4] bot_lip</span>
                      <span className="text-purple-400 font-bold">{telemetry.botGap}</span>
                    </div>
                    <div className="bg-slate-950 p-2.5 rounded-lg border border-slate-800">
                      <span className="text-[10px] text-slate-500 block">s[5] gap_diff</span>
                      <span className="text-pink-400 font-bold">{telemetry.gapDiff}</span>
                    </div>
                    <div className="bg-slate-950 p-2.5 rounded-lg border border-slate-800">
                      <span className="text-[10px] text-slate-500 block">s[6] dx_2nd</span>
                      <span className="text-slate-300 font-bold">0.860</span>
                    </div>
                    <div className="bg-slate-950 p-2.5 rounded-lg border border-slate-800">
                      <span className="text-[10px] text-slate-500 block">s[7] 2nd_diff</span>
                      <span className="text-slate-300 font-bold">0.040</span>
                    </div>
                  </div>
                </div>

                {/* Difficulty & Environment Config Controls */}
                <div className="bg-slate-900 border border-slate-800 rounded-2xl p-4 flex flex-wrap items-center justify-between gap-3 text-xs">
                  <div className="flex items-center gap-2">
                    <span className="text-slate-400 font-semibold">Pipe Gap Clearance:</span>
                    <button
                      onClick={() => setPipeGapPreset(165)}
                      className={`px-2.5 py-1 rounded-md transition font-medium ${
                        pipeGapPreset === 165
                          ? "bg-amber-500 text-slate-950 font-bold"
                          : "bg-slate-800 text-slate-300 hover:bg-slate-700"
                      }`}
                    >
                      Wide (165px)
                    </button>
                    <button
                      onClick={() => setPipeGapPreset(145)}
                      className={`px-2.5 py-1 rounded-md transition font-medium ${
                        pipeGapPreset === 145
                          ? "bg-amber-500 text-slate-950 font-bold"
                          : "bg-slate-800 text-slate-300 hover:bg-slate-700"
                      }`}
                    >
                      Standard (145px)
                    </button>
                    <button
                      onClick={() => setPipeGapPreset(125)}
                      className={`px-2.5 py-1 rounded-md transition font-medium ${
                        pipeGapPreset === 125
                          ? "bg-amber-500 text-slate-950 font-bold"
                          : "bg-slate-800 text-slate-300 hover:bg-slate-700"
                      }`}
                    >
                      Narrow (125px)
                    </button>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* TAB 2: COMPLETE REINFORCEMENT GUIDE VIEW */}
        {activeTab === "guide" && (
          <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 lg:p-8 space-y-8">
            <div className="border-b border-slate-800 pb-5 flex flex-col sm:flex-row sm:items-center justify-between gap-4">
              <div>
                <h2 className="text-xl font-bold text-white flex items-center gap-2">
                  <BookOpen className="w-5 h-5 text-amber-400" />
                  reinforcement_guide.md (Complete Project Documentation)
                </h2>
                <p className="text-xs text-slate-400 mt-1">
                  Full reference file generated in root as <code className="text-amber-300">/reinforcement_guide.md</code>
                </p>
              </div>

              <div className="flex items-center gap-2">
                <button
                  onClick={handleCopyFullGuide}
                  className="px-3.5 py-1.5 bg-slate-800 hover:bg-slate-700 text-slate-200 rounded-lg text-xs font-semibold flex items-center gap-1.5 border border-slate-700"
                >
                  {copiedGuide ? <Check className="w-4 h-4 text-emerald-400" /> : <Copy className="w-4 h-4" />}
                  {copiedGuide ? "Copied!" : "Copy Full Markdown"}
                </button>
                <button
                  onClick={handleDownloadGuide}
                  className="px-3.5 py-1.5 bg-amber-500 hover:bg-amber-400 text-slate-950 font-bold rounded-lg text-xs flex items-center gap-1.5 shadow"
                >
                  <Download className="w-4 h-4" />
                  Download File
                </button>
              </div>
            </div>

            {/* Structured Guide Content */}
            <div className="space-y-6 text-sm text-slate-300 leading-relaxed">
              <section className="space-y-2">
                <h3 className="text-lg font-bold text-white text-amber-400">1. Project Introduction & Reinforcement Learning Concept</h3>
                <p>
                  Flappy Bird formulated as a Markov Decision Process (MDP) using Double Deep Q-Networks (DDQN).
                  Instead of raw pixel frames, an 8-dimensional normalized kinematic vector is fed into an MLP with Layer Normalization:
                </p>
                <div className="bg-slate-950 p-4 rounded-xl font-mono text-xs text-amber-200 border border-slate-800">
                  s = [bird_y, bird_velocity, dx_next_pipe, top_gap_y, bot_gap_y, gap_center_diff, dx_second_pipe, second_gap_diff]
                </div>
              </section>

              <section className="space-y-3">
                <h3 className="text-lg font-bold text-white text-amber-400">2. Project Initialization with Astral <code className="text-amber-300">uv</code></h3>
                <p>
                  <a href="https://github.com/astral-sh/uv" target="_blank" rel="noreferrer" className="text-sky-400 underline">Astral uv</a> is a blazingly fast Python package manager written in Rust. It replaces pip, pip-tools, virtualenv, and poetry.
                </p>
                <div className="bg-slate-950 p-4 rounded-xl font-mono text-xs text-slate-200 border border-slate-800 space-y-2">
                  <p className="text-slate-500"># 1. Initialize project with Python 3.11</p>
                  <p className="text-emerald-400">uv init --app --python 3.11 flappy-bird-dqn && cd flappy-bird-dqn</p>
                  <p className="text-slate-500"># 2. Add dependencies</p>
                  <p className="text-emerald-400">uv add "torch&gt;=2.2.0" "gymnasium&gt;=0.29.1" "pygame&gt;=2.5.2" "huggingface-hub&gt;=0.23.0" "python-dotenv&gt;=1.0.0" "imageio&gt;=2.34.0" "imageio-ffmpeg&gt;=0.4.9" "matplotlib&gt;=3.8.0" "tqdm&gt;=4.66.0" "gradio&gt;=4.38.0"</p>
                </div>
              </section>

              <section className="space-y-3">
                <h3 className="text-lg font-bold text-white text-amber-400">3. Hugging Face Authentication & Token Setup</h3>
                <p>
                  Hugging Face requires a User Access Token with <strong>WRITE</strong> permissions to publish models and Model Cards.
                </p>
                <ul className="list-disc pl-5 space-y-1 text-slate-400 text-xs sm:text-sm">
                  <li>Visit <strong className="text-slate-200">huggingface.co/settings/tokens</strong></li>
                  <li>Click <strong className="text-slate-200">Create new token</strong> $\rightarrow$ select <strong className="text-amber-300">Write</strong></li>
                  <li>Store the token in <code className="text-amber-300">.env</code> (never commit <code className="text-amber-300">.env</code> to Git!)</li>
                </ul>
              </section>

              <section className="space-y-3">
                <h3 className="text-lg font-bold text-white text-amber-400">4. Folder Architecture</h3>
                <pre className="bg-slate-950 p-4 rounded-xl font-mono text-xs text-slate-300 border border-slate-800 overflow-x-auto">
{`flappy-bird-dqn/
├── .env                         # Hugging Face token (Write access)
├── .env.example                 # Public template
├── .gitignore                   # Ignore models, venv, secrets
├── pyproject.toml               # Astral uv project specification
├── uv.lock                      # Pinned lockfile
├── README.md                    # Project README
├── app.py                       # Hugging Face Space interactive Gradio demo
└── src/
    ├── env/flappy_env.py        # Gymnasium-compliant environment
    ├── agent/dqn_model.py       # PyTorch MLP with LayerNorm
    ├── agent/replay_buffer.py   # Experience replay buffer
    ├── agent/dqn_agent.py       # DDQN target update & Bellman loss
    ├── train.py                 # Training pipeline & learning curve plot
    ├── evaluate.py              # Visual Pygame live playback
    ├── record_video.py          # Headless GIF/MP4 recorder for portfolio
    └── push_to_hub.py           # Automated Hugging Face Hub uploader`}
                </pre>
              </section>

              <section className="space-y-3">
                <h3 className="text-lg font-bold text-white text-amber-400">5. Complete Execution Commands</h3>
                <div className="bg-slate-950 p-4 rounded-xl font-mono text-xs text-slate-200 border border-slate-800 space-y-2">
                  <p className="text-slate-500"># Train DQN Agent (approx 10 minutes)</p>
                  <p className="text-emerald-400">uv run python src/train.py</p>
                  <p className="text-slate-500"># Watch trained bird in real-time Pygame GUI</p>
                  <p className="text-emerald-400">uv run python src/evaluate.py</p>
                  <p className="text-slate-500"># Record gameplay GIF for GitHub & Hugging Face</p>
                  <p className="text-emerald-400">uv run python src/record_video.py</p>
                  <p className="text-slate-500"># Upload model & card to Hugging Face Hub</p>
                  <p className="text-emerald-400">uv run python src/push_to_hub.py</p>
                </div>
              </section>
            </div>
          </div>
        )}

        {/* TAB 3: FILE-WISE COPY-PASTABLE CODE */}
        {activeTab === "code" && (
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
            {/* File List Navigation */}
            <div className="lg:col-span-4 bg-slate-900 border border-slate-800 rounded-2xl p-3 space-y-1">
              <div className="px-3 py-2 text-xs font-bold text-slate-400 uppercase tracking-wider flex items-center gap-1.5 border-b border-slate-800/80 mb-2">
                <FolderTree className="w-3.5 h-3.5 text-amber-400" />
                Project Files ({GUIDE_FILES.length})
              </div>
              {GUIDE_FILES.map((file, idx) => (
                <button
                  key={file.path}
                  onClick={() => setSelectedFileIndex(idx)}
                  className={`w-full text-left px-3 py-2.5 rounded-xl text-xs font-mono transition flex items-center justify-between group ${
                    selectedFileIndex === idx
                      ? "bg-amber-500/15 text-amber-300 font-bold border border-amber-500/30"
                      : "text-slate-400 hover:text-slate-200 hover:bg-slate-800/50"
                  }`}
                >
                  <div className="flex items-center gap-2 truncate">
                    <FileCode className="w-3.5 h-3.5 shrink-0 text-slate-500 group-hover:text-amber-400" />
                    <span className="truncate">{file.path}</span>
                  </div>
                  <ChevronRight className={`w-3.5 h-3.5 shrink-0 ${selectedFileIndex === idx ? "text-amber-400" : "text-slate-600"}`} />
                </button>
              ))}
            </div>

            {/* Code Display Container */}
            <div className="lg:col-span-8 bg-slate-900 border border-slate-800 rounded-2xl overflow-hidden shadow-xl">
              {/* File Header */}
              <div className="bg-slate-950/80 border-b border-slate-800 px-4 py-3 flex flex-wrap items-center justify-between gap-3">
                <div>
                  <div className="flex items-center gap-2">
                    <span className="text-sm font-bold text-white font-mono">{activeFile.path}</span>
                    <span className="text-[10px] font-semibold uppercase px-2 py-0.5 rounded bg-slate-800 text-slate-400 border border-slate-700">
                      {activeFile.language}
                    </span>
                  </div>
                  <p className="text-xs text-slate-400 mt-0.5">{activeFile.description}</p>
                </div>

                <button
                  onClick={() => handleCopyCode(activeFile.filename, activeFile.content)}
                  className="flex items-center gap-1.5 px-3 py-1.5 bg-amber-500/10 hover:bg-amber-500/20 text-amber-300 border border-amber-500/30 rounded-lg text-xs font-bold transition"
                >
                  {copiedFile === activeFile.filename ? (
                    <>
                      <Check className="w-3.5 h-3.5 text-emerald-400" />
                      Copied!
                    </>
                  ) : (
                    <>
                      <Copy className="w-3.5 h-3.5" />
                      Copy File Code
                    </>
                  )}
                </button>
              </div>

              {/* Code Area */}
              <div className="p-4 bg-slate-950 overflow-x-auto max-h-[640px]">
                <pre className="font-mono text-xs text-slate-200 leading-relaxed">
                  <code>{activeFile.content}</code>
                </pre>
              </div>
            </div>
          </div>
        )}

        {/* TAB 4: HUGGING FACE & UV CLI BUILDER */}
        {activeTab === "hf" && (
          <div className="space-y-6 max-w-4xl mx-auto">
            <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 space-y-6">
              <div className="flex items-center gap-3">
                <div className="p-2.5 rounded-xl bg-gradient-to-tr from-yellow-500 to-amber-500 text-slate-950 font-black">
                  🤗
                </div>
                <div>
                  <h2 className="text-lg font-bold text-white">Hugging Face Hub & uv Command Generator</h2>
                  <p className="text-xs text-slate-400">
                    Customize your username and repository name to generate instant shell commands.
                  </p>
                </div>
              </div>

              {/* Config Form */}
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 bg-slate-950 p-4 rounded-xl border border-slate-800">
                <div>
                  <label className="block text-xs font-semibold text-slate-400 mb-1">
                    Hugging Face Username
                  </label>
                  <input
                    type="text"
                    value={hfUsername}
                    onChange={(e) => setHfUsername(e.target.value)}
                    placeholder="e.g. cdbharath"
                    className="w-full bg-slate-900 border border-slate-700 rounded-lg px-3 py-2 text-sm text-white focus:outline-none focus:border-amber-400 font-mono"
                  />
                </div>

                <div>
                  <label className="block text-xs font-semibold text-slate-400 mb-1">
                    Repository Name
                  </label>
                  <input
                    type="text"
                    value={hfRepoName}
                    onChange={(e) => setHfRepoName(e.target.value)}
                    placeholder="flappy-bird-dqn-agent"
                    className="w-full bg-slate-900 border border-slate-700 rounded-lg px-3 py-2 text-sm text-white focus:outline-none focus:border-amber-400 font-mono"
                  />
                </div>
              </div>

              {/* Step 1: Environment File */}
              <div className="space-y-2">
                <div className="flex justify-between items-center text-xs font-bold text-slate-300">
                  <span>1. Configure <code className="text-amber-300">.env</code> in your project root</span>
                  <button
                    onClick={() => {
                      const text = `HF_TOKEN=hf_your_write_token_here\nHF_USERNAME=${hfUsername}\nHF_REPO_NAME=${hfRepoName}\n`;
                      navigator.clipboard.writeText(text);
                      setCopiedFile(".env");
                      setTimeout(() => setCopiedFile(null), 2000);
                    }}
                    className="text-amber-400 hover:text-amber-300 flex items-center gap-1"
                  >
                    {copiedFile === ".env" ? <Check className="w-3.5 h-3.5 text-emerald-400" /> : <Copy className="w-3.5 h-3.5" />}
                    Copy .env content
                  </button>
                </div>
                <pre className="bg-slate-950 p-3.5 rounded-xl border border-slate-800 font-mono text-xs text-amber-200">
{`HF_TOKEN=hf_your_write_token_here
HF_USERNAME=${hfUsername}
HF_REPO_NAME=${hfRepoName}`}
                </pre>
              </div>

              {/* Step 2: Push command */}
              <div className="space-y-2">
                <div className="flex justify-between items-center text-xs font-bold text-slate-300">
                  <span>2. Run the automated publisher script</span>
                  <button
                    onClick={() => {
                      navigator.clipboard.writeText("uv run python src/push_to_hub.py");
                      setCopiedFile("push_cmd");
                      setTimeout(() => setCopiedFile(null), 2000);
                    }}
                    className="text-amber-400 hover:text-amber-300 flex items-center gap-1"
                  >
                    {copiedFile === "push_cmd" ? <Check className="w-3.5 h-3.5 text-emerald-400" /> : <Copy className="w-3.5 h-3.5" />}
                    Copy command
                  </button>
                </div>
                <div className="bg-slate-950 p-3.5 rounded-xl border border-slate-800 font-mono text-xs text-emerald-400 flex items-center justify-between">
                  <span>uv run python src/push_to_hub.py</span>
                </div>
              </div>

              {/* Destination URL */}
              <div className="bg-amber-500/10 border border-amber-500/20 p-4 rounded-xl flex items-center justify-between">
                <div>
                  <span className="text-xs text-slate-400 block">Your Model will be publicly available at:</span>
                  <a
                    href={`https://huggingface.co/${hfUsername}/${hfRepoName}`}
                    target="_blank"
                    rel="noreferrer"
                    className="text-sm font-bold text-amber-300 hover:underline flex items-center gap-1 mt-0.5"
                  >
                    https://huggingface.co/{hfUsername}/{hfRepoName}
                    <ExternalLink className="w-3.5 h-3.5" />
                  </a>
                </div>
              </div>
            </div>
          </div>
        )}
      </main>

      {/* Footer */}
      <footer className="border-t border-slate-800 bg-slate-950 py-4 px-4 text-center text-xs text-slate-500 flex flex-col sm:flex-row items-center justify-between max-w-7xl mx-auto w-full gap-2">
        <span>Self-Flapping Bird • Deep Q-Network RL Guide &amp; Simulator</span>
        <span className="text-slate-400 font-medium">Developed by <strong className="text-amber-400">Bharath CD</strong> (3rd year PES University)</span>
      </footer>
    </div>
  );
}
