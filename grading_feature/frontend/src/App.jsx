import {BrowserRouter, Routes, Route, useLocation} from 'react-router-dom'
import Home from './pages/Home'
import UserDetail from './pages/UserDetail'
import Scoring from './pages/Scoring'
import StudentFeedback from './pages/StudentFeedback'
import Navbar from './components/Navbar'

function AppContent() {
    const location = useLocation()
    const isStudentPage = location.pathname.startsWith('/student/feedback')

    return (
        <div className="dark min-h-screen bg-[#0d1117] text-[#e6edf3]">
            {!isStudentPage && <Navbar />}
            <div className={isStudentPage ? "px-6 py-6" : "pt-16 px-6"}>
                <Routes>
                    <Route path="/" element={<Home />} />
                    <Route path="/user/:userId" element={<UserDetail />} />
                    <Route path="/scoring" element={<Scoring />} />
                    <Route path="/student/feedback/:userId" element={<StudentFeedback />} />
                </Routes>
            </div>
        </div>
    )
}

export default function App() {
    return (
        <BrowserRouter>
            <AppContent />
        </BrowserRouter>
    );
}
