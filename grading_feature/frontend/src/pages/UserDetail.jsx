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


      <h2 className="text-xl font-semibold mb-3">Chat Sessions</h2>

      <table className="min-w-full bg-gray-900 border border-gray-700 rounded-lg">
        <thead className="bg-gray-800">
          <tr>
            <th className="text-left p-3 border-b border-gray-700">Title</th>
            <th className="text-left p-3 border-b border-gray-700">Last Interacted</th>
            <th className="text-left p-3 border-b border-gray-700">Action</th>
          </tr>
        </thead>
        
        <tbody>
          {user.chats.map((chat, index) => (
            <tr key={index} className="hover:bg-gray-800 transition">
              <td className="p-3">{chat.title}</td>
              <td className="p-3">{chat.message_pairs[chat.message_pairs.length - 1].timestamp.split(" ")[0]}</td>
              <td className="p-3">
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
      <div className={`fixed top-14 right-0 h-[calc(100%-56px)] w-[45%] bg-gray-900 border-l border-gray-700 shadow-xl transform transition-transform duration-300 ${selectedChat ? "translate-x-0" : "translate-x-full"}`}>
        {selectedChat && (
              
              <div className="flex flex-col h-full">

                {/* HEADER */}
                <div className="p-5 flex justify-between items-center border-b border-gray-700 bg-gray-800">
                  <h2 className="text-xl font-semibold">{selectedChat.title}</h2>
                  <h2 className="text-xl font-semibold">{user.email}</h2>
                  <button
                    onClick={() => setSelectedChat(null)}
                    className="text-gray-400 hover:text-gray-200 text-xl font-bold"
                  >
                    ✕
                  </button>
                </div>
            
                {/* SCROLLABLE CONTENT */}
                <div className="flex-1 overflow-y-auto p-4 space-y-4">
                   {selectedChat.message_pairs.map((pair, idx) => (
                    <div key={idx} className="space-y-2">

                      {/* QUESTION BUBBLE (LEFT) */}
                      <div className="flex justify-start">
                        <div className="bg-blue-600 text-white px-4 py-2 rounded-xl max-w-[75%] shadow">
                          <p className="text-xs text-gray-300 font-semibold mb-1">Student</p>
                          <p>{pair.question}</p>
                          <p className="text-[10px] text-gray-400 mt-1">
                            {pair.timestamp}
                          </p>
                        </div>
                      </div>

                      {/* ANSWER BUBBLE (RIGHT) */}
                      <div className="flex justify-end">
                        <div className="bg-gray-600 text-white px-4 py-2 rounded-xl max-w-[75%] shadow">
                          <p className="text-xs text-blue-200 font-semibold mb-1">Chatbot</p>
                          <p>{pair.answer}</p>
                          <p className="text-[10px] text-blue-100 mt-1">
                            {pair.timestamp}
                          </p>
                        </div>
                      </div>

                    </div>
                  ))}
                </div>
              </div>
          )}
      </div>
    </div>
)
}
