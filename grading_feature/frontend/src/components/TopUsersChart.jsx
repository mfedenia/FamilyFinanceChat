import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer } from 'recharts';


export default function TopUsersChart({data}) {

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
                        dataKey="num_chats" 
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