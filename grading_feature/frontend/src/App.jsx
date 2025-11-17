import {BrowserRouter, Routes, Route} from 'react-router-dom'
import Home from './pages/Home'
import UserDetail from './pages/UserDetail'
import Navbar from './components/Navbar'

export default function App() {
    return (
        <BrowserRouter>
            <div className="dark min-h-screen bg-[#0d1117] text-[#e6edf3]">

                <Navbar />

                <div className="pt-16 px-6">
                    <Routes>
                        <Route path="/" element={<Home />} />
                        <Route path="/user/:userId" element={<UserDetail />} />
                    </Routes>
                </div>

            </div>
        </BrowserRouter>
    );
}
