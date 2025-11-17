import { useEffect, useState } from "react";
import { useParams, useNavigate } from "react-router-dom";
import axios from "axios";
import Pagination from "../components/Pagination";

export default function UserDetail() {
  const { userId } = useParams();
  const navigate = useNavigate();

  // Set a specific user when clicked
  const [user, setUser] = useState(null);
  
  // Set chat when title is clicked
  const [selectedChat, setSelectedChat] = useState(null);

  // Pagination
  const [currentPage, setCurrentPage] = useState(1);
  
  useEffect((u) => {
    axios.get(`http://localhost:8080/user/${userId}`)
         .then(res => setUser(res.data))
         .catch(err => console.log(err))
  }, [userId]);

  // Loading
  if (!user) return <p>Loading...</p>;
  
  // Pagination variables and calculations
  const rowsPerPage = 8;
  const indexOfLast = currentPage * rowsPerPage;
  const indexOfFirst = indexOfLast - rowsPerPage;
  const currentChats = user.chats.slice(indexOfFirst, indexOfLast)

  return (
    <div>
      <button 
        onClick={() => navigate(-1)}
          className="inline-flex items-center gap-2 
             px-3 py-1.5 rounded-md 
             bg-[#161b22] border border-white/10 
             text-gray-300 hover:bg-[#1c2128] 
             transition"
      >
        ← Back
      </button>


      <h2 className="text-xl font-semibold mb-3">Chat Sessions</h2>

      <table className="min-w-full bg-[#0d1117] border border-white/10 rounded-lg shadow-sm">
        <thead className="bg-[#161b22] text-gray-300">
          <tr>
            <th className="text-left p-3 border-b border-white/10 font-medium">Title</th>
            <th className="text-left p-3 border-b border-white/10 font-medium">Last Interacted</th>
            <th className="text-left p-3 border-b border-white/10 font-medium">Action</th>
          </tr>
        </thead>
        
        <tbody>
          {currentChats.map((chat, index) => (
            <tr key={index} className="hover:bg-[#161b22]/50 transition">
              <td className="p-3">{chat.title}</td>
              <td className="p-3">{chat.message_pairs[chat.message_pairs.length - 1].timestamp.split(" ")[0]}</td>
              <td className="p-3">
                <button 
                  onClick={() => setSelectedChat(chat)}
                  className="bg-[#161b22] text-white px-4 py-2 rounded-2xl max-w-[75%] shadow border border-white/10"
                > 
                  View Chat
                </button>
              </td>
            </tr>
          ))}
        </tbody>
      </table> 
      
      <Pagination
      currentPage={currentPage}
      setCurrentPage={setCurrentPage}
      totalItems={user.chats.length}
      rowsPerPage={rowsPerPage}
      />
      
      {/* DRAWER */}
      <div className={`fixed top-14 right-0 h-[calc(100%-56px)] w-[45%] bg-gray-900 border-l border-gray-700 shadow-xl transform transition-transform duration-300 ${selectedChat ? "translate-x-0" : "translate-x-full"}`}>
        {selectedChat && (
              
              <div className="flex flex-col h-full">

                {/* HEADER */}
                <div className="p-5 flex justify-between items-center border-b border-white/10 bg-[#161b22]">
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
                        <div className="bg-[#21262d] text-white px-4 py-2 rounded-2xl 
                                        max-w-[75%] shadow border border-white/10">
                          <p className="text-xs text-gray-300 font-semibold mb-1">Student</p>
                          <p>{pair.question}</p>
                          <p className="text-[10px] text-gray-400 mt-1">
                            {pair.timestamp}
                          </p>
                        </div>
                      </div>

                      {/* ANSWER BUBBLE (RIGHT) */}
                      <div className="flex justify-end">
                        <div className="bg-[#161b22] text-white px-4 py-2 rounded-2xl 
                                        max-w-[75%] shadow border border-white/10">
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
