import { useEffect, useMemo, useState } from "react";
import { useParams, useSearchParams } from "react-router-dom";
import axios from "axios";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  CartesianGrid,
} from "recharts";
import Pagination from "../components/Pagination";

function performanceBandFromOverall(score) {
  const numeric = Number(score || 0);
  if (numeric >= 80) return "Strong";
  if (numeric >= 60) return "Good";
  if (numeric >= 40) return "Developing";
  return "Needs Attention";
}

function dimensionLabel(score) {
  const numeric = Number(score || 0);
  if (numeric >= 1.5) return "Strong";
  if (numeric >= 1.0) return "Okay";
  return "Improve";
}

function questionFeedback(row) {
  const weakAreas = [];
  if (Number(row.relevance) < 1) weakAreas.push("relevance to client scenario");
  if (Number(row.on_topic) < 1) weakAreas.push("staying on topic");
  if (Number(row.politeness) < 1) weakAreas.push("polite phrasing");
  if (Number(row.neutrality) < 1) weakAreas.push("neutral wording");
  if (Number(row.non_imperative) < 1) weakAreas.push("asking instead of directing");
  if (Number(row.clarity_optional) < 1) weakAreas.push("clear wording");
  if (Number(row.privacy_minimization_optional) < 1) weakAreas.push("privacy-safe wording");

  if (weakAreas.length === 0) {
    return "Well-formed question with balanced tone and focus.";
  }
  return `Improve: ${weakAreas.slice(0, 2).join(", ")}.`;
}

export default function StudentFeedback() {
  const params = useParams();
  const [searchParams] = useSearchParams();

  const routeUserId = params.userId;
  const queryUserId =
    searchParams.get("user_id") || searchParams.get("openwebui_user_id");
  const initialUserId = routeUserId || queryUserId || "all";

  const [users, setUsers] = useState([]);
  const [selectedUserId, setSelectedUserId] = useState(initialUserId);
  const [useAbi, setUseAbi] = useState(true);
  const [loading, setLoading] = useState(false);
  const [loadingUsers, setLoadingUsers] = useState(false);
  const [status, setStatus] = useState("");
  const [payload, setPayload] = useState(null);
  const [currentPage, setCurrentPage] = useState(1);

  useEffect(() => {
    setSelectedUserId(initialUserId);
  }, [initialUserId]);

  useEffect(() => {
    setCurrentPage(1);
  }, [payload]);

  useEffect(() => {
    async function fetchUsers() {
      try {
        setLoadingUsers(true);
        const response = await axios.get("/users");
        setUsers(response.data || []);
      } catch (error) {
        setStatus(`Unable to load users: ${error.message}`);
      } finally {
        setLoadingUsers(false);
      }
    }

    fetchUsers();
  }, []);

  const studentOptions = useMemo(() => {
    return [
      { value: "all", label: "All users" },
      ...users.map((u) => ({
        value: u.user_id,
        label: `${u.name || "Unknown"} (${u.email || u.user_id})`,
      })),
    ];
  }, [users]);

  async function loadFeedback(scope = selectedUserId) {
    try {
      setLoading(true);
      setStatus(scope === "all" ? "Loading dashboard for all users..." : "Loading feedback...");
      const response = await axios.get(
        `/api/student-feedback?user_id=${encodeURIComponent(scope)}&useAbi=${String(useAbi)}`
      );
      setPayload(response.data);
      setStatus("Feedback loaded.");
    } catch (error) {
      const detail = error?.response?.data?.detail;
      const message = typeof detail === "string" ? detail : JSON.stringify(detail || {});
      setStatus(`Unable to load feedback: ${message || error.message}`);
      setPayload(null);
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    loadFeedback();
  }, [selectedUserId, useAbi]);

  const scopeLabel =
    selectedUserId === "all"
      ? "All users"
      : users.find((user) => user.user_id === selectedUserId)?.name || selectedUserId;

  const rows = payload?.results || [];
  const rowsPerPage = 12;
  const indexOfLast = currentPage * rowsPerPage;
  const indexOfFirst = indexOfLast - rowsPerPage;
  const currentRows = rows.slice(indexOfFirst, indexOfLast);

  const distributionData = useMemo(() => {
    const distribution = payload?.aggregate?.distribution;
    if (!distribution) return [];

    return distribution.labels.map((label, idx) => ({
      label,
      count: distribution.counts[idx],
    }));
  }, [payload]);

  return (
    <div className="space-y-6 mt-6 max-w-6xl mx-auto">
      <div className="bg-[#161b22] border border-white/10 rounded-xl p-5 space-y-4">
        <h2 className="text-xl font-semibold text-white">Feedback Dashboard</h2>
        <p className="text-sm text-gray-300">
          This page shows question quality analysis and improvement suggestions across all users, with optional per-user filtering.
        </p>

        {payload?.user && (
          <div className="text-sm text-gray-300">
            Viewing <span className="font-semibold text-white">{payload.user.name || payload.user.email || payload.user.user_id}</span>
          </div>
        )}

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
          <div className="space-y-2">
            <label className="text-sm text-gray-300">User scope</label>
            <select
              value={selectedUserId}
              onChange={(e) => setSelectedUserId(e.target.value)}
              className="w-full bg-[#0d1117] border border-white/10 rounded px-3 py-2 text-sm"
              disabled={loadingUsers}
            >
              {studentOptions.map((opt) => (
                <option key={opt.value} value={opt.value}>
                  {opt.label}
                </option>
              ))}
            </select>
          </div>

          <div className="flex items-end">
            <label className="inline-flex items-center gap-2 text-sm text-gray-300">
              <input
                type="checkbox"
                checked={useAbi}
                onChange={(e) => setUseAbi(e.target.checked)}
              />
              Include ABI pipeline
            </label>
          </div>

          <div className="flex items-end gap-2">
            <button
              onClick={() => loadFeedback(selectedUserId)}
              disabled={loading || loadingUsers}
              className="px-3 py-1.5 rounded bg-[#21262d] border border-white/10 hover:bg-[#30363d] disabled:opacity-50"
            >
              Refresh Dashboard
            </button>
          </div>
        </div>

        <p className="text-xs text-gray-400">Current scope: {scopeLabel}</p>

        {status && <p className="text-sm text-blue-200">{status}</p>}
      </div>

      {payload?.aggregate && (
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="bg-[#161b22] border border-white/10 rounded-xl p-4">
            <p className="text-xs text-gray-400">Questions Scored</p>
            <p className="text-2xl font-semibold">{payload.aggregate.count}</p>
          </div>
          <div className="bg-[#161b22] border border-white/10 rounded-xl p-4">
            <p className="text-xs text-gray-400">Average Score (0-14)</p>
            <p className="text-2xl font-semibold">{payload.aggregate.avg_total_0_14}</p>
          </div>
          <div className="bg-[#161b22] border border-white/10 rounded-xl p-4">
            <p className="text-xs text-gray-400">Overall Progress (0-100)</p>
            <p className="text-2xl font-semibold">{payload.aggregate.overall_0_100}</p>
            <p className="text-sm text-emerald-300 mt-1">
              {performanceBandFromOverall(payload.aggregate.overall_0_100)}
            </p>
          </div>
        </div>
      )}

      <div className="bg-[#161b22] border border-white/10 rounded-xl p-5">
        <h3 className="text-lg font-semibold mb-3">How To Read Your Scores</h3>
        <ul className="list-disc pl-5 text-sm text-gray-200 space-y-1">
          <li><strong>Total score (0-14):</strong> Overall quality of each question.</li>
          <li><strong>Overall progress (0-100):</strong> Your average performance across all scored questions.</li>
          <li><strong>Dimension scores (0-2):</strong> 0 = needs work, 1 = okay, 2 = strong.</li>
          <li><strong>Focus first on weak dimensions:</strong> improve those to raise both total and overall progress.</li>
        </ul>
      </div>

      {distributionData.length > 0 && (
        <div className="bg-[#161b22] border border-white/10 rounded-xl p-5">
          <h3 className="text-lg font-semibold mb-3">My Score Distribution</h3>
          <div className="w-full h-64">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={distributionData}>
                <CartesianGrid stroke="#2f3542" strokeDasharray="3 3" />
                <XAxis dataKey="label" tick={{ fill: "#c8d1db" }} />
                <YAxis allowDecimals={false} tick={{ fill: "#c8d1db" }} />
                <Tooltip />
                <Bar dataKey="count" fill="#34d399" radius={[6, 6, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}

      {payload?.aggregate?.habits?.length > 0 && (
        <div className="bg-[#161b22] border border-white/10 rounded-xl p-5">
          <h3 className="text-lg font-semibold mb-3">Improvement Suggestions</h3>
          <ul className="list-disc pl-5 text-sm text-gray-200 space-y-1">
            {payload.aggregate.habits.map((habit) => (
              <li key={habit}>{habit}</li>
            ))}
          </ul>
        </div>
      )}

      {rows.length > 0 && (
        <div className="bg-[#161b22] border border-white/10 rounded-xl p-5 overflow-x-auto">
          <h3 className="text-lg font-semibold mb-3">Per Question Breakdown</h3>
          <table className="min-w-full text-xs">
            <thead>
              <tr className="border-b border-white/10 text-gray-300">
                <th className="text-left p-2">#</th>
                <th className="text-left p-2">Question</th>
                <th className="text-left p-2">Total (0-14)</th>
                <th className="text-left p-2">Result</th>
                <th className="text-left p-2">Relevance</th>
                <th className="text-left p-2">Politeness</th>
                <th className="text-left p-2">On-topic</th>
                <th className="text-left p-2">Neutrality</th>
                <th className="text-left p-2">Question Style</th>
                <th className="text-left p-2">Clarity</th>
                <th className="text-left p-2">Privacy Safety</th>
                <th className="text-left p-2">What this means</th>
              </tr>
            </thead>
            <tbody>
              {currentRows.map((row, idx) => (
                <tr key={row.id} className="border-b border-white/5">
                  <td className="p-2">{indexOfFirst + idx + 1}</td>
                  <td className="p-2 max-w-[520px]">{row.question}</td>
                  <td className="p-2">{row.score_total}</td>
                  <td className="p-2">{row.verdict === "good" ? "Strong" : row.verdict === "ok" ? "Okay" : "Needs work"}</td>
                  <td className="p-2">{row.relevance} ({dimensionLabel(row.relevance)})</td>
                  <td className="p-2">{row.politeness} ({dimensionLabel(row.politeness)})</td>
                  <td className="p-2">{row.on_topic} ({dimensionLabel(row.on_topic)})</td>
                  <td className="p-2">{row.neutrality} ({dimensionLabel(row.neutrality)})</td>
                  <td className="p-2">{row.non_imperative} ({dimensionLabel(row.non_imperative)})</td>
                  <td className="p-2">{row.clarity_optional} ({dimensionLabel(row.clarity_optional)})</td>
                  <td className="p-2">{row.privacy_minimization_optional} ({dimensionLabel(row.privacy_minimization_optional)})</td>
                  <td className="p-2 max-w-[240px]">{questionFeedback(row)}</td>
                </tr>
              ))}
            </tbody>
          </table>

          <Pagination
            currentPage={currentPage}
            setCurrentPage={setCurrentPage}
            totalItems={rows.length}
            rowsPerPage={rowsPerPage}
          />
        </div>
      )}
    </div>
  );
}
