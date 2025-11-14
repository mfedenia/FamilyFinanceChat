import { useEffect, useState } from "react";
import { useParams, useNavigate } from "react-router-dom";
import axios from "axios";

export default function UserDetail() {
  const { userId } = useParams();
  const navigate = useNavigate();
  const [user, setUser] = useState(null);
  const [selectedChat, setSelectedChat] = useState(null);

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
                  onClick={() => setSelectedChat(chat)}
                  className="px-3 py-1 bg-blue-600 text-white rounded-md hover:bg-blue-700"
                >
                  View Chat
                </button>
              </td>
            </tr>
          ))}
        </tbody>
      </table> 
      {/* DRAWER */}
      <div className={`fixed top-0 right-0 h-full w-[45%] bg-gray-900 border-l border-gray-700 shadow-xl transform transition-transform duration-300 ${selectedChat ? "translate-x-0" : "translate-x-full"}`}>
        {selectedChat && (
          <div className="p-5 flex flex-col h-full">
            <div className="flex justify-between items-center border-b border-gray-700 pb-3 mb-4">
              <h2 className="text-xl font-bold">{selectedChat.title}</h2>
              <h2 className="text-xl font-bold">{user.email}</h2>
              <button
                onClick={() => setSelectedChat(null)}
                className="text-gray-400 hover:text-gray-200 text-xl font-bold"
              >
                ✕
              </button>
            </div>

            <div> 
              {selectedChat.message_pairs.map((pair, idx) => ( 
                  <div key={idx} className="bg-gray-800 p-4 rounded-lg border border-gray-700">                     
                    <p className="text-blue-300 font-semibold">Question:</p>
                    <p className="mb3">{pair.question}</p>
                    <p className="text-blue-300 font-semibold">Answer:</p>
                    <p className="mb3">{pair.timestamp}</p>
                  </div>
            
            
              ))}
            </div>  
          </div>
        )}
      </div>
    </div>
    
  )
}
