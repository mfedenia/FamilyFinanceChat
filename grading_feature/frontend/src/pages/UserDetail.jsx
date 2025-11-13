import { useEffect, useState } from "react";
import { useParams, useNavigate } from "react-router-dom";
import axios from "axios";

export default function UserDetail() {
  const { userId } = useParams();
  const navigate = useNavigate();
  const [user, setUser] = useState(null);
  
  useEffect((u) => {
    axios.get(`http://localhost:8080/user/${userId}`)
         .then(res => setUser(res.data))
         .catch(err => console.log(err))
  }, [userId]);

  if (!user) return <p>Loading...</p>;

  return (
    <div>
      <button 
        onClick={() => navigate(-1)}
        className="text-blue-400 hover:underline mb-4"
      >
        ← Back
      </button>

      <h1 className="">
        {user.name} ({user.email})
      </h1>

      <h2 className="text-xl font-semibold mb-3">Chat Sessions</h2>

      <table className="min-w-full bg-gray-900 border border-gray-700 rounded-lg">
        <thead className="bg-gray-800">
          <tr>
            <th className="">Title</th>
            <th className="">Action</th>
          </tr>
        </thead>
        
        <tbody>
          {user.chats.map((chat, index) => (
            <tr key={index} className="hover:bg-gray-800 transition">
              <td className="p-3 text-center">{chat.title}</td>

              <td className="p-3 text-center">
                <button 
                  onClick={() => navigate(`/chat/${user.user_id}/${encodeURIComponent(chat.title)}`)}
                  className="px-3 py-1 bg-blue-600 text-white rounded-md hover:bg-blue-700"
                >
                  View Chat
                </button>
              </td>
            </tr>
          ))}
        </tbody>
      </table> 
    </div>
  )
}
