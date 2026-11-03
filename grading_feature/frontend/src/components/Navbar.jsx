import { useEffect, useState, useRef } from "react";
import axios from "axios";
import { NavLink } from "react-router-dom";

export default function Navbar() {
    const [inFlight, setInFlight] = useState(false);
    const [progress, setProgress] = useState(0);
    const [stage, setStage] = useState("idle");
    const [statusMessage, setStatusMessage] = useState("");
    const [banner, setBanner] = useState(null);
    const pollIntervalRef = useRef(null);

    const stopPolling = () => {
        if (pollIntervalRef.current) {
            clearInterval(pollIntervalRef.current);
            pollIntervalRef.current = null;
        }
    };

    const pollStatus = async () => {
        try {
            const res = await axios.get("/refresh/status");
            const data = res.data || {};
            setProgress(data.progress ?? 0);
            setStage(data.stage || "idle");
            if (data.message) {
                setStatusMessage(data.message);
            }

            if (data.status === "completed") {
                stopPolling();
                setInFlight(false);
                setProgress(100);
                setBanner({ type: "success", text: "Refresh completed successfully!" });
                window.dispatchEvent(new CustomEvent("refreshCompleted", { detail: data }));
                setTimeout(() => {
                    setBanner(null);
                }, 4000);
            } else if (data.status === "failed") {
                stopPolling();
                setInFlight(false);
                setBanner({
                    type: "error",
                    text: data.error || data.message || "Refresh failed.",
                });
                setTimeout(() => {
                    setBanner(null);
                }, 5000);
            }
        } catch (err) {
            console.error("Error polling refresh status:", err);
        }
    };

    const startPolling = () => {
        stopPolling();
        pollIntervalRef.current = setInterval(pollStatus, 800);
    };

    // Check status on mount in case a job is already running
    useEffect(() => {
        axios.get("/refresh/status")
            .then((res) => {
                const data = res.data || {};
                if (data.status === "in_progress") {
                    setInFlight(true);
                    setProgress(data.progress ?? 0);
                    setStage(data.stage || "extracting");
                    setStatusMessage(data.message || "Refresh in progress...");
                    startPolling();
                }
            })
            .catch(() => {});

        return () => stopPolling();
    }, []);

    const handleRefresh = async () => {
        if (inFlight) return;

        setInFlight(true);
        setProgress(5);
        setStage("starting");
        setStatusMessage("Starting refresh...");
        setBanner(null);

        try {
            // Trigger asynchronous refresh
            await axios.post("/refresh");
            startPolling();
        } catch (err) {
            if (err?.response?.status === 409) {
                // Already in flight, attach to existing job
                const existing = err.response.data?.detail?.job;
                if (existing) {
                    setProgress(existing.progress || 10);
                    setStatusMessage(existing.message || "Job in progress...");
                }
                startPolling();
            } else {
                setInFlight(false);
                setBanner({
                    type: "error",
                    text: err?.response?.data?.detail?.message || "Failed to trigger refresh.",
                });
            }
        }
    };

    return (
        <div className="w-full bg-[#0d1117] border-b border-white/10 fixed top-0 left-0 z-50">
            <div className="h-14 flex items-center justify-between px-6">
                <div className="flex items-center gap-6">
                    <h1 className="text-2xl font-semibold text-white">
                        Finance Chatbot Dashboard
                    </h1>

                    <div className="flex items-center gap-2 text-sm">
                        <NavLink
                            to="/"
                            className={({ isActive }) =>
                                `px-3 py-1 rounded-full border border-white/10 transition ${
                                    isActive ? "bg-blue-700 text-white" : "bg-[#21262d] text-gray-200 hover:bg-[#30363d]"
                                }`
                            }
                        >
                            Grading
                        </NavLink>
                        <NavLink
                            to="/scoring"
                            className={({ isActive }) =>
                                `px-3 py-1 rounded-full border border-white/10 transition ${
                                    isActive ? "bg-blue-700 text-white" : "bg-[#21262d] text-gray-200 hover:bg-[#30363d]"
                                }`
                            }
                        >
                            Scoring
                        </NavLink>
                        <NavLink
                            to="/feedback"
                            className={({ isActive }) =>
                                `px-3 py-1 rounded-full border border-white/10 transition ${
                                    isActive ? "bg-blue-700 text-white" : "bg-[#21262d] text-gray-200 hover:bg-[#30363d]"
                                }`
                            }
                        >
                            Feedback
                        </NavLink>
                    </div>
                </div>

                <div className="flex items-center gap-3">
                    {inFlight && (
                        <div className="flex items-center gap-2 bg-[#161b22] px-3 py-1 rounded-lg border border-blue-500/30 text-xs">
                            <div className="w-24 bg-[#21262d] h-2 rounded-full overflow-hidden border border-white/10">
                                <div
                                    className="bg-blue-500 h-full transition-all duration-300 ease-out"
                                    style={{ width: `${Math.max(5, progress)}%` }}
                                />
                            </div>
                            <span className="text-blue-300 font-medium whitespace-nowrap">
                                {progress}% ({stage})
                            </span>
                        </div>
                    )}

                    <button
                        onClick={handleRefresh}
                        disabled={inFlight}
                        title={inFlight ? (statusMessage || "Refresh in flight...") : "Pull latest chats and score questions"}
                        className={`px-4 py-1.5 rounded-full border border-white/10 text-sm font-medium transition shadow-sm flex items-center gap-2 ${
                            inFlight
                                ? "bg-[#21262d]/60 text-gray-400 cursor-not-allowed opacity-75"
                                : "bg-[#21262d] hover:bg-[#30363d] text-white"
                        }`}
                    >
                        {inFlight ? (
                            <>
                                <svg
                                    className="animate-spin h-4 w-4 text-blue-400"
                                    xmlns="http://www.w3.org/2000/svg"
                                    fill="none"
                                    viewBox="0 0 24 24"
                                >
                                    <circle
                                        className="opacity-25"
                                        cx="12"
                                        cy="12"
                                        r="10"
                                        stroke="currentColor"
                                        strokeWidth="4"
                                    />
                                    <path
                                        className="opacity-75"
                                        fill="currentColor"
                                        d="M4 12a8 8 0 018-8v8H4z"
                                    />
                                </svg>
                                <span>Refreshing...</span>
                            </>
                        ) : (
                            <>
                                <svg
                                    className="h-4 w-4 text-gray-300"
                                    fill="none"
                                    stroke="currentColor"
                                    viewBox="0 0 24 24"
                                >
                                    <path
                                        strokeLinecap="round"
                                        strokeLinejoin="round"
                                        strokeWidth={2}
                                        d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15"
                                    />
                                </svg>
                                <span>Refresh Data</span>
                            </>
                        )}
                    </button>
                </div>
            </div>

            {/* Notification Banner */}
            {banner && (
                <div
                    className={`px-6 py-1.5 text-xs flex items-center justify-between border-t ${
                        banner.type === "success"
                            ? "bg-green-950/80 text-green-300 border-green-800/40"
                            : "bg-red-950/80 text-red-300 border-red-800/40"
                    }`}
                >
                    <span>{banner.text}</span>
                    <button
                        onClick={() => setBanner(null)}
                        className="text-gray-400 hover:text-white ml-4 font-bold"
                    >
                        ✕
                    </button>
                </div>
            )}
        </div>
    );
}

