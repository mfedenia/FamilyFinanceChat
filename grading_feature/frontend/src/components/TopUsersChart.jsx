import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer } from 'recharts';


export default function TopUsersChart({ users }) {
    // Top Users Visualization
    function getTopUsers(users, limit = 5) {
        const result = users.map(student => ({
            name: student.name,
            Chats: student.chats?.length || 0
        }));
        return result.sort((a, b) => b.Chats - a.Chats).slice(0, limit); // Descending order
    }
    const data = getTopUsers(users)


    // If no data, show placeholder but keep same size
    if (!data || data.length === 0 || data.every(u => u.Chats === 0)) {
        return (
            <div className="bg-[#1a1a1a] border border-gray-800 rounded-xl p-6 shadow-sm">

                {/* Title stays */}
                <h2 className="text-gray-200 text-lg font-semibold mb-2">
                    Top Active Students (by # Chats)
                </h2>

                {/* Same height as the real chart */}
                <div className="w-full h-96 flex items-center justify-center">
                    <p className="text-gray-400">No student activity to display</p>
                </div>

            </div>
        );
    }
    return (
        <div className="bg-[#1a1a1a] border border-gray-800 rounded-xl p-6 shadow-sm">
            <h2 className="text-gray-200 text-lg font-semibold mb-2">
                Top Active Students (by # Chats)
            </h2>
            <div className="w-full h-96">
                <ResponsiveContainer width="100%" height="100%">
                    <BarChart
                        data={data}
                        layout="vertical"
                        margin={{ left: 0, right: 25 }}
                    >
                        <XAxis type="number" tick={{ fill: "#aaa" }} />
                        <YAxis
                            type="category"
                            dataKey="name"
                            width={100}
                            tick={{ fill: "#aaa", fontSize: 13 }}
                        />
                        <Tooltip
                            contentStyle={{ backgroundColor: "#1f1f1f", border: "none" }}
                            labelStyle={{ color: "#fff" }}
                        />
                        <Bar
                            dataKey="Chats"
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