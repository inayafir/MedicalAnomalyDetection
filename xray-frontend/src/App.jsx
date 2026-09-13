import { Routes, Route } from "react-router-dom";
import Navbar from "./components/Navbar";
import Dashboard from "./pages/Dashboard";
import Upload from "./pages/Upload";
import Studies from "./pages/Studies";
import Analysis from "./pages/Analysis";
import Report from "./pages/Report";

export default function App() {
  return (
    <div className="min-h-screen bg-void">
      <Navbar />
      <main>
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/upload" element={<Upload />} />
          <Route path="/studies" element={<Studies />} />
          <Route path="/analysis/:predictionId" element={<Analysis />} />
          <Route path="/report/:predictionId" element={<Report />} />
        </Routes>
      </main>
    </div>
  );
}
