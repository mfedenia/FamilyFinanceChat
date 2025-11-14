import axios from "axios";
export default function Navbar() {
    const handleRefresh = async () => {
        try {
            await axios.get("http://localhost:8080/refresh")
            window.location.reload()
        } catch (err) {
            console.error("Refresh Failed: ", err)
            alert("Backend non responsive")
        }  
    };

    return (
        <div className="w-full h-14 bg-gray-900 border-b border-gray-700 flex items-center justify-between px-6 fixed top-0 left-0 z-50 shadow-md">
            <h1 className="text-3xl font-semibold text-gray-100">Finance Chatbot Dashboard</h1>    

            <button 
                onClick={handleRefresh} 
                className="px-4 py-1 bg-blue-600 hover:bg-blue-700 text-white rounded-lg transition"
            >
                Refresh Data
            </button>
        </div>
    );
}