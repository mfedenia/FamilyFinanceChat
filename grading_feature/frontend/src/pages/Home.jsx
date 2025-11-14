import axios, { Axios } from "axios";
import { useEffect, useState } from "react";
import { Navigate, useNavigate } from "react-router-dom";

export default function Home() {
    const[users, setUsers] = useState([]);
    const navigate = useNavigate();
    const [searchTerm, setSearchTerm] = useState(""); 

    useEffect(() => {
        axios.get("http://localhost:8080/users")
            .then(res => setUsers(res.data))
            .catch(err => console.error(err))
    }, []);

    console.log(users);

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

    return (

        <div>
            <input
                type="text"
                placeholder="Search by name or email..."
                className="mb-4 px-3 py-2 w-full rounded bg-gray-800 border border-gray-700
                           focus:outline-none focus:ring-2 focus:ring-blue-500"
                value={searchTerm}
                onChange={(e)=> setSearchTerm(e.target.value)}
            />
            {filteredUsers.length == 0 && (
                    <p className="p-3">No users found</p>
            )}

            {filteredUsers.length > 0 && ( 
            <table className="min-w-full bg-gray-900 border border-gray-700 rounded-lg">
                <thead className="bg-gray-800">
                    <tr>
                        <th className="text-left p-3 border-b border-gray-700">Name</th>
                        <th className="text-left p-3 border-b border-gray-700">Email</th>
                        <th className="text-left p-3 border-b border-gray-700"># Chats</th>
                        <th className="text-left p-3 border-b border-gray-700">Role</th>
                        <th className="text-left p-3 border-b border-gray-700">Action</th>
                    </tr>
                </thead>
                
                <tbody>

                    {filteredUsers.map((u) => (
                        <tr key={u.user_id} className="hover:bg-gray-800 transition">
                            <td className="p-3">{u.name}</td>
                            <td className="p-3">{u.email}</td>
                            <td className="p-3">{u.chats.length}</td>
                            <td className="p-3">{u.role}</td>
                            <td className="p-3">
                                <button
                                    onClick={() => navigate(`/user/${u.user_id}`)}
                                    className="px-3 py-1 bg-blue-600 text-white rounded-md hover:bg-blue-700"
                                >
                                    View
                                </button>
                            </td>
                        </tr>
                    ))}
                </tbody>
            </table>
            )}
        </div>



    )


}
