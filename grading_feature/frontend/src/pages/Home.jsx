import axios, { Axios } from "axios";
import { useEffect, useState } from "react";
import { Navigate, useNavigate } from "react-router-dom";

export default function Home() {
    const[users, setUsers] = useState([]);
    const navigate = useNavigate();

    useEffect(() => {
        axios.get("http://localhost:8080/users")
            .then(res => setUsers(res.data))
            .catch(err => console.error(err))
    }, []);

    console.log(users);


    return (

        <div>
            <h1 className="text-3xl font-bold mb-6">All Students</h1>
            <table className="min-w-full bg-gray-900 border border-gray-700 rounded-lg">
                <thead className="bg-gray-800">
                    <tr>
                        <th className="text-left p-3 border-b border-gray-700">Name</th>
                        <th className="text-left p-3 border-b border-gray-700">Email</th>
                        <th className="text-left p-3 border-b border-gray-700"># Chats</th>
                        <th className="text-left p-3 border-b border-gray-700">Action</th>
                    </tr>
                </thead>
                <tbody>
                    {users.map((u) => (
                        <tr key={u.user_id} className="hover:bg-gray-800 transition">
                             <td className="p-3">{u.name}</td>
                            <td className="p-3">{u.email}</td>
                            <td className="p-3">{u.chats.length}</td>
                            <td className="p-3 text-center">
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
        </div>



    )


}
