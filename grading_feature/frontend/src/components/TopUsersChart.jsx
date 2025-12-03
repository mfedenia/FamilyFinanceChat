import { useState } from "react";
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer } from 'recharts';

export default function TopUsersChart({ users }) {
  const [limit, setLimit] = useState(5);

  function getTopUsers(users, limit) {
    const result = users.map(student => {
      const totalMessages = student.chats?.reduce((sum, chat) => {
        return sum + (chat.message_pairs?.length || 0);
      }, 0);

      return {
        name: student.name,
        Messages: totalMessages
      };
    });

    return result
      .sort((a, b) => b.Messages - a.Messages)
      .slice(0, limit);
  }

  const data = getTopUsers(users, limit);

  if (!data || data.length === 0) {
    return (
      <div className="bg-[#1a1a1a] border border-gray-800 rounded-xl p-6 shadow-sm">
        <div className="flex justify-between items-center mb-2">
          <h2 className="text-gray-200 text-lg font-semibold">Top Active Students</h2>
          <select
            className="bg-gray-800 text-gray-200 px-2 py-1 rounded"
            value={limit}
            onChange={(e) => setLimit(Number(e.target.value))}
          >
            <option value={3}>Top 3</option>
            <option value={5}>Top 5</option>
            <option value={10}>Top 10</option>
            <option value={20}>Top 20</option>
          </select>
        </div>

        <div className="w-full h-96 flex items-center justify-center">
          <p className="text-gray-400">No student activity to display</p>
        </div>
      </div>
    );
  }

  return (
    <div className="bg-[#1a1a1a] border border-gray-800 rounded-xl p-6 shadow-sm">

      {/* Header */}
      <div className="flex justify-between items-center mb-2">
        <h2 className="text-gray-200 text-lg font-semibold">
          Top Active Students (by # Messages)
        </h2>

        <select
          className="bg-gray-800 text-gray-200 px-2 py-1 rounded"
          value={limit}
          onChange={(e) => setLimit(Number(e.target.value))}
        >
          <option value={3}>Top 3</option>
          <option value={5}>Top 5</option>
          <option value={10}>Top 10</option>
          <option value={20}>Top 20</option>
        </select>
      </div>

      <div className="w-full h-96">
                <ResponsiveContainer width="100%" height="100%">
                    <BarChart
                        data={data}
                        layout="vertical"
                        margin={{ left: 0, right: 25 }}
                    >
                        <XAxis
                            type="number"
                            tick={{ fill: "#777", fontSize: 11 }}
                            axisLine={false}
                            tickLine={false}
                            allowDecimals={false}
                            domain={[0, 'dataMax + 20']}
                            tickCount={5}
                        />
                        <YAxis
                            type="category"
                            dataKey="name"
                            width={100}
                            tick={{ fill: "#aaa", fontSize: 11 }}
                        />
                        <Tooltip
                            contentStyle={{ backgroundColor: "#1f1f1f", border: "none" }}
                            labelStyle={{ color: "#fff" }}
                        />
                        <Bar
                            dataKey="Messages"
                            fill="#ddaafdff"
                            radius={[0, 4, 4, 0]}
                            barSize={25}
                        />
                    </BarChart>
                </ResponsiveContainer>
        </div>

    </div>
  );
}