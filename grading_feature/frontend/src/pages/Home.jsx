import axios, { Axios } from "axios";
import { useEffect, useState } from "react";
import { Navigate, useNavigate } from "react-router-dom";
import Pagination from "../components/Pagination";
import MetricCard from "../components/MetricCard";
import TopUsersChart from "../components/TopUsersChart";
import ChatsPerDayChart from "../components/ChatsPerDayChart";


export default function Home() {
    
    const[users, setUsers] = useState([]);
    const navigate = useNavigate();
    
    // Search
    const [searchTerm, setSearchTerm] = useState(""); 
    
    // Pagination 
    const [currentPage, setCurrentPage] = useState(1)

    // Get all users
    useEffect(() => {
        axios.get("http://localhost:9500/users")
            .then(res => setUsers(res.data))
            .catch(err => console.error(err))
    }, []);

    // Vars for Metric Cards 
    const totalStudents = users.length
    const totalChats = users.reduce((sum, student) => sum + student.chats.length, 0)
    const totalMessages = users.reduce((total, student) => {
        return total + student.chats.reduce((sum, chat) => {
            return sum + chat.message_pairs.length * 2;
        }, 0);
    }, 0);

    // Search function
    const filteredUsers = users.filter(u => {
        const name = u.name?.toLowerCase() || "";
        const email = u.email?.toLowerCase() || "";    
        const search = searchTerm.toLowerCase();

        return (
            name.includes(search) ||
            email.includes(search) 
        );
    });

    // Pagination Vars and Calculation
    const rowsPerPage = 8
    const idxOfLastUser = currentPage * rowsPerPage
    const idxOfFirstUser = idxOfLastUser - rowsPerPage
    const currentUsers = filteredUsers.slice(idxOfFirstUser, idxOfLastUser)

    // Top Users Visualization
    function getTopUsers(data, limit=5) {
         const result = data.map(student => ({
            name: student.name,
            num_chats: student.chats?.length || 0
        }));
        return result.sort((a,b) => b.num_chats - a.num_chats).slice(0, limit); // Descending order
    }
    const top10 = getTopUsers(users)

    // Daily Chats Visualization
    function getChatsPerDay(users) {
        const counts = {}

        users.forEach(student=>{
            student.chats.forEach(chat => {
                if (chat.message_pairs.length == 0) return;
                
                // Get the Last message timestamp in the chat
                const lastMsg = chat.message_pairs[chat.message_pairs.length - 1];
                const [date] = lastMsg.timestamp.split(" ") // Get the first thing

                counts[date] = (counts[date] || 0) + 1;
            });
        });

        return Object.keys(counts)
            .sort((a,b) => new Date(a) - new Date(b)) // chronological order
            .map(date =>({
                date, 
                chats: counts[date]
            }));
    } 

    const chatsPerDay = getChatsPerDay(users);
    console.log(chatsPerDay)

    return (

        <div className="mt-8">
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-6 mx-5 mb-8">
                <MetricCard title="Total Students" value={totalStudents} />
                <MetricCard title="Total Chats" value={totalChats} />
                <MetricCard title="Total Messages" value={totalMessages} />
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mx-5 mb-8">
                <TopUsersChart data={top10} />
                <ChatsPerDayChart data={chatsPerDay} />

            </div>

            <input
                type="text"
                placeholder="Search by name or email..."
                className="mb-4 w-full px-4 py-2 rounded-full 
                        bg-[#161b22] border border-white/10 
                        focus:outline-none focus:ring-2 focus:ring-blue-500/50 
                        text-white placeholder-gray-500"
                value={searchTerm}
                onChange={(e)=> setSearchTerm(e.target.value)}
            />
            {filteredUsers.length == 0 && (
                <p className="p-3">No users found</p>
            )}

            {filteredUsers.length > 0 && ( 
                <table className="min-w-full bg-[#0d1117] border border-white/10 rounded-lg shadow-sm">
                    <thead className="bg-[#161b22] text-gray-300">
                        <tr>
                            <th className="text-left p-3 border-b border-white/10 font-medium">Name</th>
                            <th className="text-left p-3 border-b border-white/10 font-medium">Email</th>
                            <th className="text-left p-3 border-b border-white/10 font-medium">Join Date</th>
                            <th className="text-left p-3 border-b border-white/10 font-medium"># Chats</th>
                            <th className="text-left p-3 border-b border-white/10 font-medium">Action</th>
                        </tr>
                    </thead>
                    
                    <tbody>

                        {currentUsers.map((u) => (
                            <tr key={u.user_id} className="hover:bg-gray-800 transition">
                                <td className="p-3">{u.name}</td>
                                <td className="p-3">{u.email}</td>
                                <td className="p-3">{u.join_date.split(" ")[0]}</td>
                                <td className="p-3">{u.chats.length}</td>
                                <td className="p-3">
                                    <button
                                        onClick={() => navigate(`/user/${u.user_id}`)}
                                        className="px-4 py-1 rounded-full border border-white/10 
                                                bg-[#21262d] hover:bg-[#30363d] text-white transition"
                                    >
                                        View
                                    </button>

                                </td>
                            </tr>
                        ))}
                    </tbody>
                </table>
            )}
            <Pagination 
                currentPage={currentPage}
                setCurrentPage={setCurrentPage}
                totalItems={filteredUsers.length}
                rowsPerPage={rowsPerPage}
            />
        </div>

    )

}
