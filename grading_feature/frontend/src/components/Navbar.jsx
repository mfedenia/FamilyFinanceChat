import axios from "axios";
import { NavLink } from "react-router-dom";

export default function Navbar() {
    const handleRefresh = async () => {
        try {
            await axios.get("/refresh");
            window.location.reload();
        } catch (err) {
            alert("Backend not responsive");
        }
    };

    return (
        <div className="w-full h-14 bg-[#0d1117] border-b border-white/10
                        flex items-center justify-between px-6 fixed top-0 left-0 z-50">

            <div className="flex items-center gap-6">
                <h1 className="text-2xl font-semibold text-white">
                    Finance Chatbot Dashboard
                </h1>

                <div className="flex items-center gap-2 text-sm">
                    <NavLink
                        to="/"
                        className={({ isActive }) =>
                            `px-3 py-1 rounded-full border border-white/10 transition ${
                                isActive ? "bg-blue-700 text-white" : "bg-[#21262d] text-gray-200 hover:bg-[#30363d]"
                            }`
                        }
                    >
                        Grading
                    </NavLink>
                    <NavLink
                        to="/scoring"
                        className={({ isActive }) =>
                            `px-3 py-1 rounded-full border border-white/10 transition ${
                                isActive ? "bg-blue-700 text-white" : "bg-[#21262d] text-gray-200 hover:bg-[#30363d]"
                            }`
                        }
                    >
                        Scoring
                    </NavLink>
                </div>
            </div>

            <button
                onClick={handleRefresh}
                className="px-4 py-1 rounded-full border border-white/10 
                           bg-[#21262d] hover:bg-[#30363d] text-white 
                           transition shadow-sm"
            >
                Refresh Data
            </button>
        </div>
    );
}
