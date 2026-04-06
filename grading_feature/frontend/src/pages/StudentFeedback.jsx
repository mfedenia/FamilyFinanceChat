import { useState, useEffect } from "react";
import { useParams } from "react-router-dom";
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

function isQuestionLike(text) {
  if (!text || typeof text !== "string") return false;
  const trimmed = text.trim();
  if (!trimmed) return false;

  const lower = trimmed.toLowerCase();
  const starters = [
    "what",
    "why",
    "how",
    "when",
    "where",
    "who",
    "which",
    "do ",
    "does ",
    "did ",
    "can ",
    "could ",
    "would ",
    "will ",
    "should ",
  ];

  return trimmed.includes("?") || starters.some((prefix) => lower.startsWith(prefix));
}

function extractQuestionsFromUser(user) {
  const output = [];

  if (!user || !user.chats) return output;

  (user.chats || []).forEach((chat, chatIdx) => {
    (chat.message_pairs || []).forEach((pair, pairIdx) => {
      const questionText = (pair.question || "").trim();
      if (!isQuestionLike(questionText)) {
        return;
      }

      output.push({
        id: `${user.user_id}-${chatIdx}-${pairIdx}`,
        text: questionText,
        studentId: user.user_id,
        studentName: user.name || user.email || "Unknown",
      });
    });
  });

  const seen = new Set();
  return output.filter((item) => {
    const key = `${item.studentId}:::${item.text}`;
    if (seen.has(key)) return false;
    seen.add(key);
    return true;
  });
}

export default function StudentFeedback() {
  const { userId } = useParams();
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(true);
  const [scoring, setScoring] = useState(false);
  const [result, setResult] = useState(null);
  const [currentPage, setCurrentPage] = useState(1);
  const [error, setError] = useState("");

  useEffect(() => {
    async function loadAndScore() {
      try {
        setLoading(true);
        setError("");

        // Fetch user data
        const userRes = await axios.get(`/user/${userId}`);
        setUser(userRes.data);

        // Extract questions
        const questions = extractQuestionsFromUser(userRes.data);

        if (!questions.length) {
          setError("No questions found in your chat history.");
          setLoading(false);
          return;
        }

        // Auto-score
        setScoring(true);
        const scoreRes = await axios.post("/api/score", {
          questions,
          useAbi: true,
        });

        setResult(scoreRes.data);
      } catch (err) {
        const msg = err?.response?.data?.detail || err.message;
        setError(`Failed to load feedback: ${msg}`);
      } finally {
        setLoading(false);
        setScoring(false);
      }
    }

    if (userId) {
      loadAndScore();
    }
  }, [userId]);

  if (loading) {
    return (
      <div className="mt-8 text-center">
        <p className="text-gray-300">Loading your feedback...</p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="mt-8">
        <div className="bg-red-900/30 border border-red-700 rounded-lg p-4 text-red-200">
          {error}
        </div>
      </div>
    );
  }

  if (!user) {
    return (
      <div className="mt-8 text-center">
        <p className="text-gray-300">Student not found.</p>
      </div>
    );
  }

  const tableRows = result?.results || [];
  const rowsPerPage = 10;
  const indexOfLast = currentPage * rowsPerPage;
  const indexOfFirst = indexOfLast - rowsPerPage;
  const currentRows = tableRows.slice(indexOfFirst, indexOfLast);

  return (
    <div className="space-y-6 mt-6">
      {/* Header */}
      <div className="bg-[#161b22] border border-white/10 rounded-xl p-6">
        <h1 className="text-3xl font-bold text-white mb-2">Your Question Quality Feedback</h1>
        <p className="text-gray-300">
          {user.name || user.email}
        </p>
        {scoring && <p className="text-sm text-blue-200 mt-2">Scoring your responses...</p>}
      </div>

      {/* Summary Cards */}
      {result?.aggregate && (
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <div className="bg-[#161b22] border border-white/10 rounded-xl p-4">
            <p className="text-xs text-gray-400">Total Questions</p>
            <p className="text-2xl font-semibold text-white">{result.aggregate.count}</p>
          </div>
          <div className="bg-[#161b22] border border-white/10 rounded-xl p-4">
            <p className="text-xs text-gray-400">Average Score (0-14)</p>
            <p className="text-2xl font-semibold text-white">{result.aggregate.avg_total_0_14}</p>
          </div>
          <div className="bg-[#161b22] border border-white/10 rounded-xl p-4">
            <p className="text-xs text-gray-400">Overall (0-100)</p>
            <p className="text-2xl font-semibold text-white">{result.aggregate.overall_0_100}</p>
          </div>
          <div className="bg-[#161b22] border border-white/10 rounded-xl p-4">
            <p className="text-xs text-gray-400">Trust Score (ABI)</p>
            <p className="text-2xl font-semibold text-white">{result.aggregate.abi_global?.abi_total ?? "-"}</p>
          </div>
        </div>
      )}

      {/* Score Distribution */}
      {result?.aggregate?.distribution && (
        <div className="bg-[#161b22] border border-white/10 rounded-xl p-5">
          <h3 className="text-lg font-semibold text-white mb-3">Your Score Distribution</h3>
          <div className="w-full h-64">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart
                data={result.aggregate.distribution.labels.map((label, idx) => ({
                  label,
                  count: result.aggregate.distribution.counts[idx],
                }))}
              >
                <CartesianGrid stroke="#2f3542" strokeDasharray="3 3" />
                <XAxis dataKey="label" tick={{ fill: "#c8d1db" }} />
                <YAxis allowDecimals={false} tick={{ fill: "#c8d1db" }} />
                <Tooltip />
                <Bar dataKey="count" fill="#60a5fa" radius={[6, 6, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}

      {/* Habit Feedback */}
      {result?.aggregate?.habits?.length > 0 && (
        <div className="bg-[#161b22] border border-white/10 rounded-xl p-5">
          <h3 className="text-lg font-semibold text-white mb-3">📋 Improvement Areas</h3>
          <ul className="list-disc pl-5 text-sm text-gray-200 space-y-2">
            {result.aggregate.habits.map((habit) => (
              <li key={habit}>{habit}</li>
            ))}
          </ul>
        </div>
      )}

      {/* Rubric Breakdown */}
      {result?.aggregate?.dims && (
        <div className="bg-[#161b22] border border-white/10 rounded-xl p-5">
          <h3 className="text-lg font-semibold text-white mb-3">📊 Rubric Scores (0-2)</h3>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
            {Object.entries(result.aggregate.dims).map(([key, value]) => (
              <div key={key} className="bg-[#0d1117] border border-white/5 rounded p-3">
                <p className="text-xs text-gray-400 capitalize">{key.replace(/_/g, " ")}</p>
                <p className="text-xl font-semibold text-white">{value}</p>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* ABI Trust Profile */}
      {result?.aggregate?.abi_global && (
        <div className="bg-[#161b22] border border-white/10 rounded-xl p-5">
          <h3 className="text-lg font-semibold text-white mb-3">🤝 Your Trust Profile (ABI)</h3>
          <p className="text-sm text-gray-300 mb-4">
            Ability, Benevolence, and Integrity measure different aspects of your questioning style.
          </p>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-4">
            <div className="bg-[#0d1117] border border-white/5 rounded p-4">
              <p className="text-sm text-gray-400">Ability</p>
              <p className="text-2xl font-semibold text-white">{result.aggregate.abi_global.ability}</p>
              <p className="text-xs text-gray-500 mt-2">Competence & clarity in your questions</p>
            </div>
            <div className="bg-[#0d1117] border border-white/5 rounded p-4">
              <p className="text-sm text-gray-400">Benevolence</p>
              <p className="text-2xl font-semibold text-white">{result.aggregate.abi_global.benevolence}</p>
              <p className="text-xs text-gray-500 mt-2">Care & respect for client interests</p>
            </div>
            <div className="bg-[#0d1117] border border-white/5 rounded p-4">
              <p className="text-sm text-gray-400">Integrity</p>
              <p className="text-2xl font-semibold text-white">{result.aggregate.abi_global.integrity}</p>
              <p className="text-xs text-gray-500 mt-2">Ethical & non-manipulative approach</p>
            </div>
          </div>
        </div>
      )}

      {/* Per-Question Table */}
      {tableRows.length > 0 && (
        <div className="bg-[#161b22] border border-white/10 rounded-xl p-5 overflow-x-auto">
          <h3 className="text-lg font-semibold text-white mb-3">Your Questions & Scores</h3>
          <table className="min-w-full text-xs">
            <thead>
              <tr className="border-b border-white/10 text-gray-300">
                <th className="text-left p-2">#</th>
                <th className="text-left p-2">Question</th>
                <th className="text-right p-2">Score</th>
                <th className="text-left p-2">Verdict</th>
              </tr>
            </thead>
            <tbody>
              {currentRows.map((row, idx) => (
                <tr key={row.id} className="border-b border-white/5">
                  <td className="p-2 text-gray-400">{indexOfFirst + idx + 1}</td>
                  <td className="p-2 max-w-sm text-gray-200">{row.question}</td>
                  <td className="p-2 text-right font-semibold text-white">{row.score_total}/14</td>
                  <td className="p-2">
                    <span
                      className={`text-xs px-2 py-1 rounded ${
                        row.verdict === "good"
                          ? "bg-green-900/30 text-green-200"
                          : row.verdict === "needs_work"
                          ? "bg-red-900/30 text-red-200"
                          : "bg-yellow-900/30 text-yellow-200"
                      }`}
                    >
                      {row.verdict}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>

          <Pagination
            currentPage={currentPage}
            setCurrentPage={setCurrentPage}
            totalItems={tableRows.length}
            rowsPerPage={rowsPerPage}
          />
        </div>
      )}

      {/* Footer */}
      <div className="text-center text-xs text-gray-500 pb-6">
        <p>This feedback is based on your conversations with the chatbot.</p>
        <p>Questions are scored on relevance, clarity, politeness, and respect for privacy.</p>
      </div>
    </div>
  );
}
