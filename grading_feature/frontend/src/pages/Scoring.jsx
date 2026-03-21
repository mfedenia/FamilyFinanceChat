import { useMemo, useState, useEffect } from "react";
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

function extractQuestionsFromUsers(users, selectedUserId) {
  const output = [];

  users.forEach((user) => {
    if (selectedUserId !== "all" && user.user_id !== selectedUserId) {
      return;
    }

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
  });

  const seen = new Set();
  return output.filter((item) => {
    const key = `${item.studentId}:::${item.text}`;
    if (seen.has(key)) return false;
    seen.add(key);
    return true;
  });
}

export default function Scoring() {
  const [users, setUsers] = useState([]);
  const [selectedUserId, setSelectedUserId] = useState("all");
  const [questions, setQuestions] = useState([]);
  const [useAbi, setUseAbi] = useState(false);
  const [loadingUsers, setLoadingUsers] = useState(true);
  const [isScoring, setIsScoring] = useState(false);
  const [status, setStatus] = useState("");
  const [result, setResult] = useState(null);
  const [currentPage, setCurrentPage] = useState(1);

  useEffect(() => {
    async function fetchUsers() {
      try {
        setLoadingUsers(true);
        const response = await axios.get("/users");
        setUsers(response.data || []);
      } catch (error) {
        setStatus("Failed to load users. Refresh data and try again.");
      } finally {
        setLoadingUsers(false);
      }
    }

    fetchUsers();
  }, []);

  useEffect(() => {
    setCurrentPage(1);
  }, [result]);

  const studentOptions = useMemo(() => {
    const options = [
      { value: "all", label: "All students" },
      ...users.map((u) => ({
        value: u.user_id,
        label: `${u.name || "Unknown"} (${u.email || u.user_id})`,
      })),
    ];
    return options;
  }, [users]);

  const tableRows = result?.results || [];
  const rowsPerPage = 15;
  const indexOfLast = currentPage * rowsPerPage;
  const indexOfFirst = indexOfLast - rowsPerPage;
  const currentRows = tableRows.slice(indexOfFirst, indexOfLast);

  async function handleExtract() {
    const extracted = extractQuestionsFromUsers(users, selectedUserId);
    setQuestions(extracted);
    setResult(null);
    setStatus(
      `Extracted ${extracted.length} question(s) from ${
        new Set(extracted.map((x) => x.studentId)).size
      } student(s).`
    );
  }

  async function handleScore() {
    if (!questions.length) {
      setStatus("No extracted questions. Click Extract Questions first.");
      return;
    }

    try {
      setIsScoring(true);
      setStatus("Scoring questions...");
      const response = await axios.post("/api/score", {
        questions,
        useAbi,
      });
      setResult(response.data);
      setStatus("Scoring completed.");
    } catch (error) {
      const detail = error?.response?.data?.detail;
      const message = typeof detail === "string" ? detail : JSON.stringify(detail || {});
      setStatus(`Scoring failed: ${message || error.message}`);
    } finally {
      setIsScoring(false);
    }
  }

  return (
    <div className="space-y-6 mt-6">
      <div className="bg-[#161b22] border border-white/10 rounded-xl p-5 space-y-4">
        <h2 className="text-xl font-semibold text-white">Question Quality Scoring</h2>
        <p className="text-sm text-gray-300">
          Use extracted chat data from the grading dashboard and run unified rubric scoring with optional ABI metrics.
        </p>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
          <div className="space-y-2">
            <label className="text-sm text-gray-300">Student scope</label>
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
              Apply ABI pipeline
            </label>
          </div>

          <div className="flex items-end gap-2">
            <button
              onClick={handleExtract}
              className="px-4 py-2 rounded bg-[#21262d] border border-white/10 hover:bg-[#30363d]"
              disabled={loadingUsers}
            >
              Extract Questions
            </button>
            <button
              onClick={handleScore}
              className="px-4 py-2 rounded bg-blue-700 hover:bg-blue-600 disabled:bg-blue-900"
              disabled={isScoring || !questions.length}
            >
              Score
            </button>
          </div>
        </div>

        {status && <p className="text-sm text-blue-200">{status}</p>}
      </div>

      {result?.aggregate && (
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="bg-[#161b22] border border-white/10 rounded-xl p-4">
            <p className="text-xs text-gray-400">Total Questions</p>
            <p className="text-2xl font-semibold">{result.aggregate.count}</p>
          </div>
          <div className="bg-[#161b22] border border-white/10 rounded-xl p-4">
            <p className="text-xs text-gray-400">Average Score (0-14)</p>
            <p className="text-2xl font-semibold">{result.aggregate.avg_total_0_14}</p>
          </div>
          <div className="bg-[#161b22] border border-white/10 rounded-xl p-4">
            <p className="text-xs text-gray-400">Overall (0-100)</p>
            <p className="text-2xl font-semibold">{result.aggregate.overall_0_100}</p>
          </div>
        </div>
      )}

      {result?.aggregate?.distribution && (
        <div className="bg-[#161b22] border border-white/10 rounded-xl p-5">
          <h3 className="text-lg font-semibold mb-3">Score Distribution</h3>
          <div className="w-full h-72">
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

      {result?.aggregate?.habits?.length > 0 && (
        <div className="bg-[#161b22] border border-white/10 rounded-xl p-5">
          <h3 className="text-lg font-semibold mb-3">Habit Feedback</h3>
          <ul className="list-disc pl-5 text-sm text-gray-200 space-y-1">
            {result.aggregate.habits.map((habit) => (
              <li key={habit}>{habit}</li>
            ))}
          </ul>
        </div>
      )}

      {result?.aggregate?.perStudent && (
        <div className="bg-[#161b22] border border-white/10 rounded-xl p-5 overflow-x-auto">
          <h3 className="text-lg font-semibold mb-3">Per Student Summary</h3>
          <table className="min-w-full text-sm">
            <thead>
              <tr className="border-b border-white/10 text-gray-300">
                <th className="text-left p-2">Student</th>
                <th className="text-left p-2">Questions</th>
                <th className="text-left p-2">Avg (0-14)</th>
                <th className="text-left p-2">Overall (0-100)</th>
                <th className="text-left p-2">ABI Total</th>
              </tr>
            </thead>
            <tbody>
              {Object.values(result.aggregate.perStudent).map((student) => (
                <tr key={student.studentId} className="border-b border-white/5">
                  <td className="p-2">{student.studentName}</td>
                  <td className="p-2">{student.count}</td>
                  <td className="p-2">{student.avg_total_0_14}</td>
                  <td className="p-2">{student.overall_0_100}</td>
                  <td className="p-2">{student.abi_avg?.abi_total ?? "-"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {result?.results?.length > 0 && (
        <div className="bg-[#161b22] border border-white/10 rounded-xl p-5 overflow-x-auto">
          <h3 className="text-lg font-semibold mb-3">Per Question Scores</h3>
          <table className="min-w-full text-xs">
            <thead>
              <tr className="border-b border-white/10 text-gray-300">
                <th className="text-left p-2">#</th>
                <th className="text-left p-2">Student</th>
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
                  <td className="p-2">{row.studentName}</td>
                  <td className="p-2 max-w-[500px]">{row.question}</td>
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
            totalItems={tableRows.length}
            rowsPerPage={rowsPerPage}
          />
        </div>
      )}
    </div>
  );
}
