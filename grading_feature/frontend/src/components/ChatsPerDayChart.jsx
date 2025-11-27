import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid } from "recharts";

export default function ChatsPerDayChart({ data }) {
  return (
    <div className="bg-[#1a1a1a] border border-gray-800 rounded-xl p-6 shadow-sm">
      <h2 className="text-gray-200 text-lg font-semibold mb-4">
        Chats Per Day
      </h2>

      <div className="w-full h-96">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={data}>
            <CartesianGrid stroke="#333" strokeDasharray="3 3" />
            <XAxis dataKey="date" tick={{ fill: "#aaa", fontSize: 7}} />
            <YAxis tick={{ fill: "#aaa" }} />
            <Tooltip 
              contentStyle={{ backgroundColor: "#1f1f1f", border: "none" }}
              labelStyle={{ color: "#fff" }}
            />
            <Line 
              type="monotone" 
              dataKey="chats" 
              stroke="#A78BFA" 
              strokeWidth={3}
              dot={{ r: 4, fill: "#A78BFA" }}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
