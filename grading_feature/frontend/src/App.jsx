import {BrowserRouter, Routes, Route} from 'react-router-dom'
import Home from './pages/Home'
import ChatViewer from './pages/ChatViewer'
import UserDetail from './pages/UserDetail'

export default function App() {
    return (
        <BrowserRouter>
            
            <div className="dark bg-[#0d0d0d] text-gray-200 min-h-screen">
                <Routes>
                    <Route path='/' element={<Home />} />
                    <Route path='/user/:userId' element={<UserDetail />} />
                </Routes>
            </div>
        </BrowserRouter>
    )
}