import axios, { Axios } from "axios";
import { useEffect, useState } from "react";
import { Navigate, useNavigate } from "react-router-dom";
import Pagination from "../components/Pagination";

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

    return (

        <div>
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
