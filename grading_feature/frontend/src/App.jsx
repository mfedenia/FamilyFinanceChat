import {BrowserRouter, Routes, Route} from 'react-router-dom'
import Home from './pages/Home'
import UserDetail from './pages/UserDetail'
import Navbar from './components/Navbar'

export default function App() {
    return (
        <BrowserRouter>
            <div className="dark bg-[#0d0d0d] text-gray-200 min-h-screen">

                <Navbar />

                <div className="pt-16 px-6">
                    <Routes>
                        <Route path='/' element={<Home />} />
                        <Route path='/user/:userId' element={<UserDetail />} />
                    </Routes>
                </div>
            </div>
        </BrowserRouter>
    )
}