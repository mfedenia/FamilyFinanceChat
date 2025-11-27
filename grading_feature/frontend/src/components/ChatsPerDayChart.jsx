import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid } from "recharts";
import { useEffect, useState } from "react";

export default function ChatsPerDayChart({ users }) {
  // Line Chart range
  const [range, setRange] = useState("month");

  // Daily Chats Visualization
  function getChatsPerDay(students) {
    const counts = {};

    students.forEach(student => {
      student.chats.forEach(chat => {
        if (chat.message_pairs.length === 0) return;
        const date = chat.message_pairs.at(-1).timestamp.split(" ")[0];
        counts[date] = (counts[date] || 0) + 1;
      });
    });

    return Object.keys(counts)
      .sort((a, b) => new Date(a) - new Date(b))
      .map(date => ({
        date,
        chats: counts[date]
      }));
  }
  function makeContinuousSeries(data) {
    if (!data.length) return [];

    const first = new Date(data[0].date);
    const last = new Date(data[data.length - 1].date);
    const out = [];

    for (let d = new Date(first); d <= last; d.setDate(d.getDate() + 1)) {
      const dateStr = d.toLocaleDateString("en-US");
      const found = data.find(item => item.date === dateStr);
      out.push({
        date: dateStr,
        chats: found ? found.chats : 0
      });
    }

    return out;
  }

  function filterByRange(data, range) {
    if (range === "all") return data;
    const days = range === "week" ? 7 : 30;
    const cutoff = new Date();
    cutoff.setDate(cutoff.getDate() - (days - 1));

    return data.filter(d => new Date(d.date) >= cutoff);
  }
  const raw = getChatsPerDay(users);
  const continuous = makeContinuousSeries(raw);
  const data = filterByRange(continuous, range);
  
if (!data || data.length === 0) {
  return (
    <div className="bg-[#1a1a1a] border border-gray-800 rounded-xl p-6 shadow-sm">
      
      {/* Title + buttons (empty state keeps them) */}
      <div className="flex justify-between items-center mb-4">
        <h2 className="text-gray-200 text-lg font-semibold">Chats Per Day</h2>
        <div className="flex gap-2 opacity-0">
          <button className="px-3 py-1 rounded bg-gray-800">Week</button>
          <button className="px-3 py-1 rounded bg-gray-800">Month</button>
          <button className="px-3 py-1 rounded bg-gray-800">All</button>
        </div>
      </div>

      {/* Fake chart height placeholder */}
      <div className="w-full h-96 flex items-center justify-center">
        <p className="text-gray-400 text-sm">No data for this range</p>
      </div>

    </div>
  );
}


  return (
    <div className="bg-[#1a1a1a] border border-gray-800 rounded-xl p-6 shadow-sm">

      <div className="flex justify-between items-center mb-4">
        <h2 className="text-gray-200 text-lg font-semibold">
          Chats Per Day
        </h2>

        <div className="flex gap-2">
          <button
            onClick={() => setRange("week")}
            className={`px-3 py-1 rounded text-sm ${range === "week"
              ? "bg-indigo-600 text-white"
              : "bg-gray-800 text-gray-300"
              }`}
          >
            Week
          </button>

          <button
            onClick={() => setRange("month")}
            className={`px-3 py-1 rounded text-sm ${range === "month"
              ? "bg-indigo-600 text-white"
              : "bg-gray-800 text-gray-300"
              }`}
          >
            Month
          </button>

          <button
            onClick={() => setRange("all")}
            className={`px-3 py-1 rounded text-sm ${range === "all"
              ? "bg-indigo-600 text-white"
              : "bg-gray-800 text-gray-300"
              }`}
          >
            All
          </button>
        </div>
      </div>

      {/* Chart */}
      <div className="w-full h-96">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={data}>
            <CartesianGrid stroke="#333" strokeDasharray="3 3" />
            <XAxis
              dataKey="date"
              tick={{ fill: "#aaa", fontSize: 8 }}
            />
            <YAxis
              tick={{ fill: "#aaa" }}
              allowDecimals={false}
              domain={[0, 'dataMax']}
            />
            <Tooltip
              contentStyle={{ backgroundColor: "#1f1f1f", border: "none" }}
              labelStyle={{ color: "#fff" }}
            />
            <Line
              type="monotone"
              dataKey="chats"
              stroke="#A78BFA"
              strokeWidth={3}
              dot={{ r: 2, fill: "#A78BFA" }}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
