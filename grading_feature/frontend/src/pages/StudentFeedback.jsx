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

export default function StudentFeedback() {
  const params = useParams();
  const [searchParams] = useSearchParams();

  const routeUserId = params.userId;
  const queryUserId =
    searchParams.get("user_id") || searchParams.get("openwebui_user_id");
  const userId = routeUserId || queryUserId || "";

  const [useAbi, setUseAbi] = useState(true);
  const [loading, setLoading] = useState(false);
  const [status, setStatus] = useState("");
  const [payload, setPayload] = useState(null);
  const [currentPage, setCurrentPage] = useState(1);

  useEffect(() => {
    setCurrentPage(1);
  }, [payload]);

  async function loadFeedback() {
    if (!userId) {
      setStatus("Missing user identity. Open this page from your OpenWebUI account link.");
      return;
    }

    try {
      setLoading(true);
      setStatus("Loading your feedback...");
      const response = await axios.get(
        `/api/student-feedback/${encodeURIComponent(userId)}?useAbi=${String(useAbi)}`
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
  }, [userId, useAbi]);

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
        <h2 className="text-xl font-semibold text-white">My Chatbot Feedback</h2>
        <p className="text-sm text-gray-300">
          This page shows your question quality analysis and improvement suggestions based on your OpenWebUI chats.
        </p>

        {payload?.user && (
          <div className="text-sm text-gray-300">
            Signed in as <span className="font-semibold text-white">{payload.user.name || payload.user.email || payload.user.user_id}</span>
          </div>
        )}

        <div className="flex flex-wrap items-center gap-3">
          <label className="inline-flex items-center gap-2 text-sm text-gray-300">
            <input
              type="checkbox"
              checked={useAbi}
              onChange={(e) => setUseAbi(e.target.checked)}
            />
            Include ABI pipeline
          </label>

          <button
            onClick={loadFeedback}
            disabled={loading || !userId}
            className="px-3 py-1.5 rounded bg-[#21262d] border border-white/10 hover:bg-[#30363d] disabled:opacity-50"
          >
            Refresh My Feedback
          </button>
        </div>

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
            <p className="text-xs text-gray-400">Overall (0-100)</p>
            <p className="text-2xl font-semibold">{payload.aggregate.overall_0_100}</p>
          </div>
        </div>
      )}

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
                <th className="text-left p-2">Total</th>
                <th className="text-left p-2">Verdict</th>
                <th className="text-left p-2">Rel</th>
                <th className="text-left p-2">Pol</th>
                <th className="text-left p-2">On-topic</th>
                <th className="text-left p-2">Neutral</th>
                <th className="text-left p-2">Non-imp</th>
                <th className="text-left p-2">Clarity</th>
                <th className="text-left p-2">Privacy</th>
              </tr>
            </thead>
            <tbody>
              {currentRows.map((row, idx) => (
                <tr key={row.id} className="border-b border-white/5">
                  <td className="p-2">{indexOfFirst + idx + 1}</td>
                  <td className="p-2 max-w-[520px]">{row.question}</td>
                  <td className="p-2">{row.score_total}</td>
                  <td className="p-2">{row.verdict}</td>
                  <td className="p-2">{row.relevance}</td>
                  <td className="p-2">{row.politeness}</td>
                  <td className="p-2">{row.on_topic}</td>
                  <td className="p-2">{row.neutrality}</td>
                  <td className="p-2">{row.non_imperative}</td>
                  <td className="p-2">{row.clarity_optional}</td>
                  <td className="p-2">{row.privacy_minimization_optional}</td>
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
